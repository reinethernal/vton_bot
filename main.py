import os
# Suppress verbose logs from TensorFlow and Mediapipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF and mediapipe warnings
os.environ.setdefault("GLOG_minloglevel", "2")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging._warn_preinit_stderr = False

import logging
import asyncio
import json
import gc
import psutil
import cv2
import torch
import sys
from aiogram import Bot, Dispatcher, types, F, __version__ as aiogram_version
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import FSInputFile
from dotenv import load_dotenv
from aiogram.filters import Command
from vton import virtual_try_on

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
UNIFORMS_STR = os.getenv("UNIFORMS", "{}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    logger.error("BOT_TOKEN is not set. Exiting.")
    sys.exit(1)

# Memory monitoring function
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    gpu_memory = ""
    if torch.cuda.is_available():
        gpu_memory = f", GPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
    logger.info(f"Memory usage: {memory_mb:.1f}MB RAM{gpu_memory}")

# Clear memory function
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Parse UNIFORMS from .env
try:
    PRELOADED_UNIFORMS = json.loads(UNIFORMS_STR)
    logger.info(f"Loaded {len(PRELOADED_UNIFORMS)} uniform configurations")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding UNIFORMS from .env: {e}")
    PRELOADED_UNIFORMS = {}

# StatesGroup for FSM
class TryOnProcess(StatesGroup):
    waiting_for_person_photo = State()
    waiting_for_uniform_selection = State()

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

@dp.message(F.photo, ~F.via_bot)
async def handle_initial_person_photo(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is not None:
        await message.reply("Я уже обрабатываю ваш запрос. Пожалуйста, подождите.")
        return

    if not PRELOADED_UNIFORMS:
        await message.reply("Извините, нет доступных форм для примерки. Проверьте конфигурацию.")
        return

    log_memory_usage()
    
    try:
        await message.reply("📸 Получил ваше фото! Сохраняю...")
        
        photo = message.photo[-1]
        person_file_path = f"static/uploads/person_{message.from_user.id}_{photo.file_id}.jpg"
        os.makedirs(os.path.dirname(person_file_path), exist_ok=True)
        await bot.download(photo, destination=person_file_path)
        
        if not os.path.exists(person_file_path):
            await message.reply("❌ Ошибка: не удалось сохранить фото.")
            return
            
        await state.update_data(person_path=person_file_path)
        logger.info(f"Saved person photo: {person_file_path}")

        available_uniforms = [(name, path) for name, path in PRELOADED_UNIFORMS.items() if os.path.exists(path)]
        if not available_uniforms:
            await message.reply("❌ Извините, нет доступных форм для примерки.")
            await state.clear()
            return

        buttons = [InlineKeyboardButton(text=f"👕 {name}", callback_data=f"select_uniform_{name}") for name, _ in available_uniforms]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons[i:i + 2] for i in range(0, len(buttons), 2)])
        
        await message.answer("✅ Фото сохранено! Теперь выберите форму для примерки:", reply_markup=keyboard)
        await state.set_state(TryOnProcess.waiting_for_uniform_selection)

    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await message.reply("❌ Произошла ошибка при обработке фото. Попробуйте еще раз.")
        await state.clear()


@dp.message(Command("rotate"))
async def rotate_person_photo(message: types.Message, state: FSMContext):
    """Rotate the saved user photo."""
    user_data = await state.get_data()
    person_path = user_data.get("person_path")
    if not person_path or not os.path.exists(person_path):
        await message.reply("❌ Нет фотографии для поворота. Сначала отправьте фото.")
        return

    direction = "right"
    parts = message.text.split(maxsplit=1)
    if len(parts) > 1:
        direction = parts[1].lower()
    if direction not in {"right", "left"}:
        await message.reply("❌ Укажите направление: /rotate [right|left]")
        return

    img = cv2.imread(person_path)
    if img is None:
        await message.reply("❌ Ошибка при загрузке фото.")
        return
    code = cv2.ROTATE_90_CLOCKWISE if direction == "right" else cv2.ROTATE_90_COUNTERCLOCKWISE
    rotated = cv2.rotate(img, code)
    cv2.imwrite(person_path, rotated)
    await message.reply("🔄 Фото повернуто.")

@dp.callback_query(F.data.startswith("select_uniform_"), TryOnProcess.waiting_for_uniform_selection)
async def handle_uniform_selection(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer("🔄 Обрабатываю ваш запрос...")
    
    uniform_name = callback_query.data.replace("select_uniform_", "")
    uniform_path = PRELOADED_UNIFORMS.get(uniform_name)

    if not uniform_path or not os.path.exists(uniform_path):
        await callback_query.message.reply("❌ Выбранная форма не найдена.")
        await state.clear()
        return

    user_data = await state.get_data()
    person_path = user_data.get("person_path")
    
    if not person_path or not os.path.exists(person_path):
        await callback_query.message.reply("❌ Фото человека не найдено. Начните сначала.")
        await state.clear()
        return

    logger.info(f"Processing try-on: person={person_path}, uniform={uniform_path}")
    log_memory_usage()

    processing_msg = await callback_query.message.reply(
        f"🎭 Примеряю форму '{uniform_name}'...\n"
        "⏳ Это может занять несколько минут, пожалуйста подождите."
    )

    try:
        clear_memory()
        result_path = await asyncio.to_thread(
            virtual_try_on, person_path, uniform_path
        )
        
        if result_path and os.path.exists(result_path):
            logger.info(f"Try-on successful: {result_path}")
            log_memory_usage()
            input_file = FSInputFile(result_path)
            await callback_query.message.reply_photo(
                input_file,
                caption=f"✨ Готово! Вот результат примерки с формой '{uniform_name}'"
            )
            for file_path in [person_path, result_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
        else:
            await callback_query.message.reply(
                "❌ К сожалению, не удалось создать результат примерки. "
                "Попробуйте с другим фото или другой формой."
            )
            
        try:
            await processing_msg.delete()
        except Exception as e:
            logger.warning(f"Failed to delete processing message: {e}")
            
    except Exception as e:
        logger.error(f"Error during virtual try-on: {e}", exc_info=True)
        try:
            await processing_msg.delete()
        except Exception as e:
            logger.warning(f"Failed to delete processing message: {e}")
        await callback_query.message.reply("❌ Произошла ошибка при обработке. Попробуйте еще раз.")
    finally:
        await state.clear()
        clear_memory()
        log_memory_usage()

async def main():
    logger.info(f"Starting bot with aiogram version {aiogram_version}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
