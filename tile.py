import subprocess
import os
import sys

INPUT_FILE = "/app/data/processed/frontend.geojson"
OUTPUT_FILE = "/app/client/public/data/buildings.pmtiles"

if not os.path.exists(INPUT_FILE):
    print(f"Файл {INPUT_FILE} не найден! Проверь, что он есть в ./data/processed/")
    sys.exit(1)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

command = [
    "tippecanoe",
    "-o", OUTPUT_FILE,
    "--minimum-zoom=10",
    "--maximum-zoom=16",
    "--base-zoom=14",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--force",
    "--progress",    # ← запятая обязательна
    INPUT_FILE
]

print(f"Начинаю варку тайлов из {INPUT_FILE}...")
try:
    subprocess.run(command, check=True)
    print(f"Успех! Файл лежит в {OUTPUT_FILE}")
except subprocess.CalledProcessError as e:
    print(f"Ошибка при работе Tippecanoe: {e}")
    sys.exit(1)