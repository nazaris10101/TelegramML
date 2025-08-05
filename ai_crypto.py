import os, math, random, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep
from binance.client import Client

from datetime import timedelta
from binance.client import Client
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers

# =========================================================
# Конфіг
# =========================================================
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1DAY
LOOKBACK = 2000         # скільки днів підтягувати (до 2000 у одному запиті)
SEQ_LEN = 120           # довжина історії
PRED_LEN = 7            # прогноз на 7 днів
FOLDS = 3               # walk-forward фолди
ENSEMBLE = 3            # скільки моделей усереднюємо
BATCH_SIZE = 32
EPOCHS = 120            # EarlyStopping зупинить раніше
MODEL_DIR = "models_maxprec"
SCALER_PATH = "scaler_robust.gz"
SKIP_TRAIN = False      # True → пропустити тренування, лише завантажити моделі та зробити прогноз

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# Утиліти
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, k=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / (ma + 1e-9)
    return upper, lower, width

def fetch_ohlcv(limit=LOOKBACK, symbol=SYMBOL, interval=INTERVAL, retries=5):
    """
    Тягне ОСТАННІ `limit` свічок із Binance Spot (за замовч. BTCUSDT, 1d).
    Пагінація назад (через endTime), ретраї, повертає UTC-час і float-значення.
    Колонки: ['timestamp','open','high','low','close','volume'].
    """
    client = Client()
    max_per_req = 1000
    collected = []
    end_time = None  # None → останні свічки; далі рухаємось у минуле

    cols = [
        'open_time','open','high','low','close','volume',
        'close_time','quote_asset_volume','number_of_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ]

    while len(collected) < limit:
        batch_limit = min(max_per_req, limit - len(collected))
        last_err, batch = None, None

        for attempt in range(retries):
            try:
                batch = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=batch_limit,
                    endTime=end_time
                )
                break
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"Помилка завантаження ({e}). Повтор через {wait}s…")
                sleep(wait)

        if last_err is not None:
            raise RuntimeError(f"Не вдалося отримати дані з Binance: {last_err}")
        if not batch:
            break

        # наступний запит — перед першою свічкою поточного батчу
        first_open_time = batch[0][0]  # open_time мс
        end_time = first_open_time - 1

        collected = batch + collected  # додаємо на початок, зберігаючи хронологію
        sleep(0.2)  # маленька пауза, щоб не впертися в ліміти

    if not collected:
        raise ValueError("Порожня відповідь від Binance.")

    df = pd.DataFrame(collected, columns=cols)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)

    df = df[['timestamp','open','high','low','close','volume']].sort_values('timestamp').reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['log_ret'] = np.log(out['close']).diff()
    out['range'] = (out['high'] - out['low']) / (out['open'] + 1e-9)
    out['hlc_mean'] = (out['high'] + out['low'] + out['close']) / 3
    # EMA/SMA
    out['EMA_12'] = out['close'].ewm(span=12, adjust=False).mean()
    out['EMA_26'] = out['close'].ewm(span=26, adjust=False).mean()
    out['SMA_14'] = out['close'].rolling(14).mean()
    out['SMA_50'] = out['close'].rolling(50).mean()
    # Волатильність
    out['vol_10'] = out['log_ret'].rolling(10).std()
    out['vol_20'] = out['log_ret'].rolling(20).std()
    # RSI, MACD, Bollinger
    out['RSI_14'] = rsi(out['close'], 14)
    macd_line, signal_line, hist = macd(out['close'])
    out['MACD'] = macd_line
    out['MACD_sig'] = signal_line
    out['MACD_hist'] = hist
    bb_u, bb_l, bb_w = bollinger(out['close'], 20, 2)
    out['BB_upper'] = bb_u
    out['BB_lower'] = bb_l
    out['BB_width'] = bb_w
    # Обʼємні
    out['vol_z'] = (out['volume'] - out['volume'].rolling(20).mean()) / (out['volume'].rolling(20).std() + 1e-9)
    out['vwap_proxy'] = (out['hlc_mean'] * out['volume']).rolling(5).sum() / (out['volume'].rolling(5).sum() + 1e-9)
    # Лаги close
    out['close_1'] = out['close'].shift(1)
    out['close_3'] = out['close'].shift(3)
    out['close_5'] = out['close'].shift(5)
    out = out.dropna().reset_index(drop=True)
    return out

def to_supervised(arr: np.ndarray, seq_len: int, pred_len: int, close_col_idx: int):
    X, y = [], []
    for i in range(seq_len, len(arr) - pred_len + 1):
        X.append(arr[i-seq_len:i, :])
        y.append(arr[i:i+pred_len, close_col_idx])  # вектор з 7 майбутніх close
    return np.array(X), np.array(y)

class SelfAttention(layers.Layer):
    def __init__(self, d_k=64, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.Wq = self.add_weight(shape=(d, self.d_k), initializer="glorot_uniform", name="Wq")
        self.Wk = self.add_weight(shape=(d, self.d_k), initializer="glorot_uniform", name="Wk")
        self.Wv = self.add_weight(shape=(d, self.d_k), initializer="glorot_uniform", name="Wv")
    def call(self, x):
        Q = tf.matmul(x, self.Wq)
        K = tf.matmul(x, self.Wk)
        V = tf.matmul(x, self.Wv)
        attn_scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.d_k)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        context = tf.matmul(attn_weights, V)
        return context

def build_model(input_timesteps, input_dim, pred_len):
    inputs = layers.Input(shape=(input_timesteps, input_dim))
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="gelu")(inputs)
    x = layers.Conv1D(64, kernel_size=5, padding="causal", activation="gelu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = SelfAttention(d_k=64)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.RepeatVector(pred_len)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.TimeDistributed(layers.Dense(1))(x)

    model = models.Model(inputs, outputs)
    # Фіксований LR (щоб не конфліктувати з колбеками)
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mae")
    return model

def walk_forward_eval(X, Y, folds=FOLDS):
    n = len(X)
    fold_size = n // (folds + 1)
    splits = []
    for f in range(1, folds + 1):
        train_end = fold_size * f
        test_end = fold_size * (f + 1)
        splits.append(((0, train_end), (train_end, test_end)))
    return splits

def inverse_close_only(seq_scaled, scaler: RobustScaler, feature_dim: int, close_idx: int):
    arr = seq_scaled.reshape(-1, 1)
    dummy = np.zeros((arr.shape[0], feature_dim))
    dummy[:, close_idx] = arr[:, 0]
    inv = scaler.inverse_transform(dummy)[:, close_idx]
    return inv.reshape(seq_scaled.shape)

def save_forecast_csv_png(df_feat: pd.DataFrame, fut_prices: np.ndarray, png_name="btc_7day_forecast.png", csv_name="btc_7day_forecast.csv"):
    start_date = df_feat['timestamp'].iloc[-1] + pd.Timedelta(days=1)
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(len(fut_prices))]
    forecast_df = pd.DataFrame({"date": future_dates, "predicted_close_usd": np.round(fut_prices, 2)})
    csv_path = os.path.join(os.path.dirname(__file__), csv_name)
    forecast_df.to_csv(csv_path, index=False)
    print(f"Прогноз збережено у: {csv_path}")
    # Графік: останні 200 днів + 7 прогнозних
    close_hist = df_feat['close'].values
    tail = 200
    hist = close_hist[-tail:]
    plt.figure(figsize=(12,6))
    plt.plot(range(len(hist)), hist, label="Історія (close)")
    plt.plot(range(len(hist), len(hist) + len(fut_prices)), fut_prices, label="Прогноз (7 днів)")
    plt.title(f"{SYMBOL}: 7-денний прогноз (ансамбль, розширені фічі)")
    plt.xlabel("Дні (історія)")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(os.path.dirname(__file__), png_name)
    plt.savefig(png_path, dpi=160, bbox_inches='tight')
    print(f"Графік збережено у: {png_path}")
    plt.show()

# =========================================================
# Основний сценарій
# =========================================================
def main():
    set_seed(42)
    print("1) Завантаження та підготовка даних…")
    raw = fetch_ohlcv(LOOKBACK)
    df = build_features(raw)

    FEATURES = [
        'open','high','low','close','volume',
        'log_ret','range','hlc_mean',
        'EMA_12','EMA_26','SMA_14','SMA_50',
        'vol_10','vol_20','RSI_14',
        'MACD','MACD_sig','MACD_hist',
        'BB_upper','BB_lower','BB_width',
        'vol_z','vwap_proxy','close_1','close_3','close_5'
    ]
    close_idx = FEATURES.index('close')

    data = df[FEATURES].values

    # Масштабування (робимо на train-частині, але зберігаємо для використання)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Scaler завантажено.")
    else:
        scaler = RobustScaler()
        scaler.fit(data[: int(len(data)*0.8) ])
        joblib.dump(scaler, SCALER_PATH)
        print("Scaler навчено і збережено.")

    data_scaled = scaler.transform(data)

    # Послідовності
    X, Y = to_supervised(data_scaled, SEQ_LEN, PRED_LEN, close_idx)
    print(f"X: {X.shape}, Y: {Y.shape}")

    # ---------- Walk-forward оцінка ----------
    splits = walk_forward_eval(X, Y, folds=FOLDS)
    fold_metrics = []

    if not SKIP_TRAIN:
        for fi, ((tr0, tr1), (te0, te1)) in enumerate(splits, 1):
            X_tr, Y_tr = X[tr0:tr1], Y[tr0:tr1]
            X_te, Y_te = X[te0:te1], Y[te0:te1]
            print(f"\n2) Fold {fi}: train={X_tr.shape}, test={X_te.shape}")

            preds_ens = []
            for m in range(ENSEMBLE):
                set_seed(100 + m)
                model = build_model(SEQ_LEN, X.shape[2], PRED_LEN)
                cb = [
                    callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
                ]
                model.fit(
                    X_tr, Y_tr[..., np.newaxis],
                    validation_split=0.1,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb
                )
                preds = model.predict(X_te, verbose=0).squeeze(-1)  # (N, PRED_LEN)
                preds_ens.append(preds)

            preds_mean = np.mean(preds_ens, axis=0)

            # Інверсія масштабу
            Y_te_inv = inverse_close_only(Y_te, scaler, len(FEATURES), close_idx)
            P_te_inv = inverse_close_only(preds_mean, scaler, len(FEATURES), close_idx)

            mae = mean_absolute_error(Y_te_inv.flatten(), P_te_inv.flatten())
            rmse = np.sqrt(mean_squared_error(Y_te_inv.flatten(), P_te_inv.flatten()))
            fold_metrics.append((mae, rmse))
            print(f"Fold {fi} → MAE: {mae:,.2f} USD | RMSE: {rmse:,.2f} USD")

        if fold_metrics:
            mae_mean = np.mean([m[0] for m in fold_metrics])
            rmse_mean = np.mean([m[1] for m in fold_metrics])
            print(f"\nСередні метрики (walk-forward) → MAE: {mae_mean:,.2f} | RMSE: {rmse_mean:,.2f} USD")

        # ---------- Фінальне тренування на всіх даних ----------
        print("\n3) Фінальне тренування ансамблю на всіх даних…")
        final_models = []
        for m in range(ENSEMBLE):
            set_seed(200 + m)
            model = build_model(SEQ_LEN, X.shape[2], PRED_LEN)
            cb = [
                callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
            ]
            model.fit(
                X, Y[..., np.newaxis],
                validation_split=0.1,
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb
            )
            path = os.path.join(MODEL_DIR, f"ensemble_{m}_final.keras")
            model.save(path)
            final_models.append(model)
            print(f"  Модель {m} збережено у {path}")
    else:
        # ---------- Завантажити існуючі моделі ----------
        print("Пропускаємо тренування. Завантажуємо модель(і)…")
        final_models = []
        for m in range(ENSEMBLE):
            path = os.path.join(MODEL_DIR, f"ensemble_{m}_final.keras")
            if os.path.exists(path):
                final_models.append(tf.keras.models.load_model(path, custom_objects={"SelfAttention": SelfAttention}))
        if not final_models:
            raise RuntimeError("Не знайдено збережених моделей. Запусти з SKIP_TRAIN=False для навчання.")

    # ---------- Прогноз на наступні 7 днів ----------
    print("\n4) Прогноз на наступні 7 днів…")
    last_seq = data_scaled[-SEQ_LEN:]  # (SEQ_LEN, F)
    ens_future = []
    for model in final_models:
        fut = model.predict(last_seq[np.newaxis, ...], verbose=0).squeeze()  # (PRED_LEN,)
        ens_future.append(fut)
    fut_mean = np.mean(ens_future, axis=0)

    fut_prices = inverse_close_only(fut_mean, scaler, len(FEATURES), close_idx)
    for i, p in enumerate(fut_prices, 1):
        print(f"День {i}: {p:,.2f} USD")

    # ---------- Збереження у CSV + графік ----------
    save_forecast_csv_png(df[['timestamp','close']].assign(close=df['close']), fut_prices)

if __name__ == "__main__":
    # Запобіжник для старих Mac: вимкнути AVX помилки TensorFlow (опціонально)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
