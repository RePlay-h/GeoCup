import subprocess
import os
import sys

INPUT_FILE = "/app/data/processed/frontend.geojson"
OUTPUT_FILE = "/app/client/public/data/buildings.pmtiles"

if not os.path.exists(INPUT_FILE):
    print(f"❌ Файл {INPUT_FILE} не найден!")
    print("Проверь, что data-pipeline отработал и создал frontend.geojson")
    sys.exit(1)

# Создаём папку, если её нет
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Удаляем старый файл, чтобы избежать "Wrong magic number"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"🗑️  Старый файл {OUTPUT_FILE} удалён")

# Современная и более стабильная команда tippecanoe
command = [
    "tippecanoe",
    "-o", OUTPUT_FILE,
    "--minimum-zoom=10",
    "--maximum-zoom=16",
    "--base-zoom=14",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--force",
    "--progress=1",               # ← важно: =1
    "--layer=buildings",
    "--name=buildings",
    "--no-feature-limit",         # позволяет больше объектов
    "--no-tile-size-limit",       # ослабляет агрессивный дроп
    "--simplify-only-low-zooms",  # упрощать геометрию только на низких зумах
    INPUT_FILE
]

print(f"🚀 Начинаю генерацию тайлов из {INPUT_FILE}...")
print(f"Выходной файл: {OUTPUT_FILE}")

try:
    result = subprocess.run(command, check=True, capture_output=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Успех! Файл создан: {OUTPUT_FILE}")
    print(f"   Размер: {size_mb:.1f} MB")
except subprocess.CalledProcessError as e:
    print(f"❌ Ошибка при работе tippecanoe:")
    print(e)
    sys.exit(1)