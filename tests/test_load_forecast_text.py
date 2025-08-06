import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import telegram_bot


def test_load_forecast_uses_predicted_column(tmp_path, monkeypatch):
    df = pd.DataFrame({"predicted_close_usd": [100.0], "other": [1]})
    csv_path = tmp_path / "forecast.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(telegram_bot, "CSV_PATH", csv_path)
    text = telegram_bot.load_forecast_text()
    assert "День 1: $100.00" in text


def test_load_forecast_missing_file(tmp_path, monkeypatch):
    csv_path = tmp_path / "missing.csv"
    monkeypatch.setattr(telegram_bot, "CSV_PATH", csv_path)
    text = telegram_bot.load_forecast_text()
    assert "Файл прогнозу не знайдено" in text
