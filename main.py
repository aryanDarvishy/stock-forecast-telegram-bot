import asyncio
from telegram import Update
from telegram.ext import ContextTypes, Application, CommandHandler, MessageHandler, filters
from logic import process_text, log_result
from visualization import plot_forecast
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("Нет BOT_TOKEN в переменных окружения")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    await update.message.reply_text(
        "Привет! Я анализирую акции и делаю прогноз (учебно).\n"
        "Отправь сообщение в формате: TICKER AMOUNT\n"
        "Пример: AAPL 1000\n"
        "Если что-то не так — я подскажу формат."
    )
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    text = (update.message.text or "").strip()

    result = await asyncio.to_thread(process_text, text)
    user_id = update.effective_user.id if update.effective_user else None

    if user_id is not None:
        await asyncio.to_thread(log_result, user_id, result)

    if not result.get('ok'):
        await update.message.reply_text(result.get("message", "Ошибка"))
        return
    
    photo = await asyncio.to_thread(plot_forecast, result)
    await update.message.reply_photo(photo=photo)
    await update.message.reply_text(result["message"])

def main():
    
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling()

if __name__ == "__main__":
    main()
