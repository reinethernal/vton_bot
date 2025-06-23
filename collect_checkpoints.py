import os
import shutil
import logging
import argparse
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Целевая директория для копирования задаётся аргументом
parser = argparse.ArgumentParser(description="Collect model checkpoints")
DEFAULT_TARGET_DIR = os.environ.get("TARGET_DIR", "checkpoints_collected")
parser.add_argument(
    "--target-dir",
    default=DEFAULT_TARGET_DIR,
    help="Directory to copy checkpoints into (or set TARGET_DIR env var)",
)
args = parser.parse_args()

TARGET_DIR = args.target_dir
os.makedirs(TARGET_DIR, exist_ok=True)

# Список исходных путей (из вашего find)
source_paths = [
    "./pytorch-openpose/model/body_pose_model.pth",
    "./pytorch-openpose/model/hand_pose_model.pth",
    "./models/u2net.pth",
    "./models/cloth_segm_u2net_latest.pth",
    "./models/CatVTON/SCHP/exp-schp-201908261155-lip.pth",
    "./models/CatVTON/SCHP/exp-schp-201908301523-atr.pth",
    "./models/u2net_portrait.pth",
    "./models/cloth_seg.pth",
    "./models/gmm_final.pth",
    "./models/seg_final.pth",
    "./venv/lib/python3.10/site-packages/distutils-precedence.pth",  # Пропустим этот файл
    "./pytorch3d/tests/pulsar/reference/nr0000-in.pth",
    "./pytorch3d/tests/pulsar/reference/nr0000-out.pth",
    "./pytorch3d/tests/data/icp_data.pth",
    "./pytorch3d/docs/tutorials/data/camera_graph.pth",
    "./VITON-HD/checkpoints/seg_final.pth",
    "./VITON-HD/checkpoints/gmm_final.pth",
    "./VITON-HD/checkpoints/alias_final.pth"
]

# Фильтрация: исключаем нерелевантные файлы (например, distutils-precedence.pth)
relevant_paths = [path for path in source_paths if not path.endswith("distutils-precedence.pth")]

# Копирование файлов и логирование
for src_path in relevant_paths:
    if os.path.exists(src_path):
        # Сохраняем исходное имя файла
        filename = os.path.basename(src_path)
        target_path = os.path.join(TARGET_DIR, filename)
        
        # Проверяем, не существует ли файл с таким именем, добавляем суффикс если нужно
        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            target_path = os.path.join(TARGET_DIR, f"{name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(src_path, target_path)  # copy2 сохраняет метаданные
        logger.info(f"Copied {src_path} to {target_path}")
    else:
        logger.warning(f"Source file not found: {src_path}")

logger.info(f"Completed. All relevant checkpoints copied to {TARGET_DIR}")
