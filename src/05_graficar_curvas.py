"""Etapa 5: genera las curvas de entrenamiento que pide la seccion 2.3 del reporte.

Lee los CSV que deja el entrenamiento y produce, por cada iteracion:
  - puntaje por episodio con su media movil de 50 episodios,
  - perdida media contra pasos de entrenamiento,
  - puntaje de las evaluaciones greedy periodicas.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DIR_ENTREGABLES, DIR_LOGS


def leer_csv(ruta: Path) -> dict:
    with ruta.open() as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return {}
    return {clave: [fila[clave] for fila in filas] for clave in filas[0]}


def media_movil(valores: np.ndarray, ventana: int = 50) -> np.ndarray:
    if len(valores) < ventana:
        ventana = max(1, len(valores))
    nucleo = np.ones(ventana) / ventana
    return np.convolve(valores, nucleo, mode="valid")


def graficar(etiqueta: str, salida: Path) -> None:
    datos = leer_csv(DIR_LOGS / f"{etiqueta}_episodios.csv")
    if not datos:
        print(f"Sin datos de episodios para {etiqueta}")
        return

    pasos = np.array([float(x) for x in datos["paso"]])
    puntajes = np.array([float(x) for x in datos["puntaje"]])
    perdidas = np.array([float(x) if x else np.nan for x in datos["perdida_media"]])

    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.2))

    ejes[0].plot(pasos, puntajes, alpha=0.25, color="tab:blue", linewidth=0.8,
                 label="puntaje por episodio")
    if len(puntajes) >= 2:
        suavizado = media_movil(puntajes, 50)
        ejes[0].plot(pasos[len(pasos) - len(suavizado):], suavizado,
                     color="tab:red", linewidth=2, label="media movil (50 ep.)")
    ejes[0].set_xlabel("pasos de entorno")
    ejes[0].set_ylabel("puntaje")
    ejes[0].set_title(f"{etiqueta}: recompensa de entrenamiento")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(pasos, perdidas, color="tab:orange", linewidth=1)
    ejes[1].set_xlabel("pasos de entorno")
    ejes[1].set_ylabel("entropia cruzada (C51)")
    ejes[1].set_title("perdida media")
    ejes[1].grid(alpha=0.3)

    evaluaciones = leer_csv(DIR_LOGS / f"{etiqueta}_eval.csv")
    if evaluaciones:
        pe = np.array([float(x) for x in evaluaciones["paso"]])
        prom = np.array([float(x) for x in evaluaciones["promedio"]])
        maxi = np.array([float(x) for x in evaluaciones["maximo"]])
        ejes[2].plot(pe, prom, "o-", color="tab:green", label="promedio")
        ejes[2].plot(pe, maxi, "s--", color="tab:purple", alpha=0.7, label="maximo")
        ejes[2].legend()
    ejes[2].set_xlabel("pasos de entorno")
    ejes[2].set_ylabel("puntaje")
    ejes[2].set_title("evaluacion greedy periodica")
    ejes[2].grid(alpha=0.3)

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=140)
    print(f"Curvas escritas en {salida}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etiquetas", nargs="*", default=None,
                        help="iteraciones a graficar; por defecto todas las de logs/")
    args = parser.parse_args()

    etiquetas = args.etiquetas
    if not etiquetas:
        etiquetas = sorted(
            p.name.replace("_episodios.csv", "")
            for p in DIR_LOGS.glob("*_episodios.csv")
        )
    for etiqueta in etiquetas:
        graficar(etiqueta, DIR_ENTREGABLES / "figuras" / f"curvas_{etiqueta}.png")


if __name__ == "__main__":
    main()
