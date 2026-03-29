import subprocess
import os

def make_tiles():
    input_path = "data/processed/your_file.geojson"
    output_dir = "client/public/data"
    output_path = os.path.join(output_dir, "buildings.pmtiles")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = [
        "tippecanoe",
        "-o", output_path,
        "--force",
        "-zg",
        "--drop-densest-as-needed",
        input_path
    ]

    print(f"Начинаю варку тайлов из {input_path}...")
    subprocess.run(command, check=True)
    print(f" Файл лежит в {output_path}")

if __name__ == "__main__":
    make_tiles()