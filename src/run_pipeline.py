"""Orquesta las etapas del pipeline en orden."""

import runpy
from pathlib import Path

ETAPAS = ["01_generar_videos.py"]


def main() -> None:
    directorio = Path(__file__).resolve().parent
    for etapa in ETAPAS:
        print(f"=== {etapa} ===")
        runpy.run_path(str(directorio / etapa), run_name="__main__")


if __name__ == "__main__":
    main()
