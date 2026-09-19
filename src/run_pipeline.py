"""Orquesta las etapas del pipeline en orden.

Por defecto NO corre el entrenamiento (etapa 02), que toma horas: se asume que
ya existe un checkpoint entrenado y se ejecutan las etapas de evaluacion, video
y graficas. Con --entrenar se incluye tambien el entrenamiento.
"""

import argparse
import runpy
import sys
from pathlib import Path

ETAPA_LABORATORIO = "01_generar_videos.py"
ETAPA_ENTRENAMIENTO = "02_entrenar_dqn.py"
ETAPAS_AGENTE = [
    "03_evaluar_agente.py",
    "04_generar_video_agente.py",
    "05_graficar_curvas.py",
]


def correr(directorio: Path, etapa: str, argv=None) -> None:
    print(f"\n=== {etapa} ===")
    sys.argv = [etapa] + list(argv or [])
    runpy.run_path(str(directorio / etapa), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrenar", action="store_true",
                        help="incluye la etapa 02 de entrenamiento (horas de computo)")
    parser.add_argument("--pasos", type=int, default=4_000_000,
                        help="pasos de entrenamiento si se usa --entrenar")
    parser.add_argument("--baselines", action="store_true",
                        help="regenera los videos del laboratorio 5 (agente aleatorio y de regla simple)")
    args = parser.parse_args()

    directorio = Path(__file__).resolve().parent

    if args.baselines:
        correr(directorio, ETAPA_LABORATORIO)
    if args.entrenar:
        correr(directorio, ETAPA_ENTRENAMIENTO,
               ["--pasos", str(args.pasos), "--etiqueta", "rainbow_v1"])
    for etapa in ETAPAS_AGENTE:
        correr(directorio, etapa)


if __name__ == "__main__":
    main()
