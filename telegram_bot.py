#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Application,
)

# ================== НАЛАШТУВАННЯ ==================
CSV_PATH = Path("/Users/mihajluknazar/Desktop/btc_7day_forecast.csv")
PNG_PATH = Path("/Users/mihajluknazar/Desktop/btc_7day_forecast.png")
MODEL_SCRIPT = Path("/Users/mihajluknazar/Desktop/ai_crypto.py")
PYTHON_BIN = "/usr/local/bin/python3"          # Інтерпретатор, яким ви запускали модель
TZ = ZoneInfo("Europe/Zurich")                 # Часовий пояс для розсилки

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)

# Пам’ять підписок (проста множина у процесі)
subscribers: set[int] = set()


# ================== УТИЛІТИ ==================
def load_forecast_text(max_days: int = 7) -> str:
    """Завантажити CSV з прогнозом та зібрати текст."""
    if not CSV_PATH.exists():
        return "❌ Файл прогнозу не знайдено."

    try:
        df = pd.read_csv(CSV_PATH)
        # Вибір колонки з прогнозом: спочатку 'predicted_close_usd', потім 'forecast',
        # і лише як запасний варіант беремо перший стовпець
        if "predicted_close_usd" in df.columns:
            col = "predicted_close_usd"
        elif "forecast" in df.columns:
            col = "forecast"
        else:
            col = df.columns[0]
        vals = df[col].tolist()[:max_days]

        lines = ["📈 Прогноз ціни BTC на 7 днів:"]
        for i, v in enumerate(vals, start=1):
            try:
                lines.append(f"День {i}: ${float(v):,.2f}")
            except Exception:
                lines.append(f"День {i}: {v}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Помилка читання CSV: {e}"


async def send_forecast(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Надіслати прогноз і, якщо є, картинку."""
    text = load_forecast_text()
    await context.bot.send_message(chat_id=chat_id, text=text)

    if PNG_PATH.exists():
        try:
            with PNG_PATH.open("rb") as f:
                await context.bot.send_photo(
                    chat_id=chat_id, photo=f, caption="Графік прогнозу (7 днів)"
                )
        except Exception as e:
            await context.bot.send_message(
                chat_id=chat_id, text=f"⚠️ Не вдалося надіслати зображення: {e}"
            )


async def rebuild_forecast() -> tuple[bool, str]:
    """
    Запустити модельний скрипт і дочекатися завершення.
    Повертає (ok, tail_log).
    """
    if not MODEL_SCRIPT.exists():
        return False, f"Скрипт не знайдено: {MODEL_SCRIPT}"

    try:
        def _run():
            return subprocess.run(
                [PYTHON_BIN, str(MODEL_SCRIPT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        result = await asyncio.to_thread(_run)
        ok = (result.returncode == 0)
        # Повертати тільки «хвіст» логу, щоб не засмічувати чат
        return ok, (result.stdout or "")[-2000:]
    except Exception as e:
        return False, f"Виняток: {e}"


# ================== ХЕНДЛЕРИ КОМАНД ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    msg = (
        "👋 Привіт! Я NazarPriceBot.\n\n"
        "Команди:\n"
        "• /predict — показати поточний прогноз (7 днів)\n"
        "• /rebuild — перегенерувати прогноз (запустити модель) і показати результат\n"
        "• /daily_on — щоденна розсилка о 09:00 (Europe/Zurich)\n"
        "• /daily_off — вимкнути щоденну розсилку\n"
    )
    await update.message.reply_text(msg)


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_forecast(update.effective_chat.id, context)


async def rebuild(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ Запускаю модель, зачекай… (~кілька хвилин)")
    ok, tail = await rebuild_forecast()
    if ok:
        await update.message.reply_text("✅ Модель завершила роботу. Надсилаю оновлений прогноз.")
        await send_forecast(update.effective_chat.id, context)
    else:
        await update.message.reply_text("❌ Помилка під час запуску моделі. Лог нижче:")
        # Відправимо як преформатований текст (без MarkdownV2, щоб не екранувати)
        await update.message.reply_text(f"```\n{tail}\n```", parse_mode="Markdown")


async def daily_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Щоденна розсилка всім підписникам."""
    for chat_id in list(subscribers):
        try:
            await send_forecast(chat_id, context)
        except Exception as e:
            logging.warning(f"Надсилання до {chat_id} не вдалося: {e}")


async def daily_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Увімкнути персональну щоденну розсилку 09:00 Europe/Zurich."""
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)

    # Прибрати можливі дублікати jobів
    for job in context.job_queue.get_jobs_by_name(str(chat_id)):
        job.schedule_removal()

    # Щодня о 09:00 за TZ
    context.job_queue.run_daily(
        daily_job,
        time=time(hour=9, minute=0, tzinfo=TZ),
        name=str(chat_id),
    )
    await update.message.reply_text("✅ Щоденна розсилка увімкнена (щодня о 09:00, Europe/Zurich).")


async def daily_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Вимкнути персональну щоденну розсилку."""
    chat_id = update.effective_chat.id
    removed = False
    for job in context.job_queue.get_jobs_by_name(str(chat_id)):
        job.schedule_removal()
        removed = True
    if chat_id in subscribers:
        subscribers.remove(chat_id)
    await update.message.reply_text("🛑 Розсилку вимкнено." if removed else "ℹ️ Активної розсилки не було.")


# ================== ГОЛОВНА ФУНКЦІЯ ==================
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not token.strip():
        raise RuntimeError("Не задано TELEGRAM_BOT_TOKEN у змінних середовища.")

    app: Application = ApplicationBuilder().token(token.strip()).build()

    # Команди
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("rebuild", rebuild))
    app.add_handler(CommandHandler("daily_on", daily_on))
    app.add_handler(CommandHandler("daily_off", daily_off))

    # Глобальна щоденна розсилка (усім, хто в subscribers)
    app.job_queue.run_daily(
        daily_job,
        time=time(hour=9, minute=0, tzinfo=TZ),
        name="global_daily",
    )

    print("🤖 Бот запущено. Очікує команди…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
