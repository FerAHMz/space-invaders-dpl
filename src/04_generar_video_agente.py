"""Etapa 4: graba el video entregable del agente entrenado jugando una partida.

Usa el mismo entorno de evaluacion (misma configuracion de wrappers y politica
greedy) que la etapa 3, para que el video sea respaldo fiel del puntaje que se
reporta. Por defecto graba la mejor de varias partidas.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agente import AgenteRainbow, elegir_dispositivo
from ale_utils import fijar_semillas
from config import (
    CHECKPOINT_FINAL,
    DIR_ENTREGABLES,
    DIR_VIDEOS,
    MAX_STEPS_EVALUACION,
    RAIZ,
    SEMILLA_EVALUACION,
)
from wrappers import crear_entorno_evaluacion


def grabar_partidas(
    ruta_modelo: Path,
    carpeta: Path,
    prefijo: str,
    n_episodios: int,
    semilla: int,
    dispositivo: str = None,
) -> dict:
    dev = elegir_dispositivo(dispositivo)
    agente = AgenteRainbow.cargar(ruta_modelo, dev)

    carpeta.mkdir(parents=True, exist_ok=True)
    env = crear_entorno_evaluacion(video_folder=str(carpeta), name_prefix=prefijo)
    episodios = []
    try:
        for i in range(n_episodios):
            obs, _ = env.reset(seed=semilla + i)
            env.action_space.seed(semilla + i)
            total, pasos, terminated, truncated = 0.0, 0, False, False
            while pasos < MAX_STEPS_EVALUACION and not (terminated or truncated):
                accion = agente.actuar(obs, entrenando=False)
                obs, recompensa, terminated, truncated, _ = env.step(accion)
                total += float(recompensa)
                pasos += 1
            episodios.append({
                "episodio": i,
                "semilla": semilla + i,
                "pasos": pasos,
                "recompensa_total": total,
                "video": f"{prefijo}-episode-{i}.mp4",
            })
            print(f"episodio {i}: {total:.0f} puntos -> {prefijo}-episode-{i}.mp4")
    finally:
        # Indispensable: RecordVideo solo vuelca el ultimo .mp4 al cerrar.
        env.close()

    mejor = max(episodios, key=lambda e: e["recompensa_total"])
    return {"episodios": episodios, "mejor": mejor}


def tabla_markdown(episodios: list, mejor: dict) -> str:
    """Arma la tabla de resultados lista para pegar en el reporte.

    Junta en una sola vista el puntaje de cada episodio y el video que lo
    respalda, que es exactamente lo que pide el enunciado: el video debe ser
    evidencia del puntaje reportado.
    """
    import statistics

    puntajes = [e["recompensa_total"] for e in episodios]
    filas = ["| Episodio | Semilla | Puntaje | Pasos | Video |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for e in episodios:
        marca = "**" if e is mejor else ""
        filas.append(
            f"| {e['episodio']} | {e['semilla']} | {marca}{e['recompensa_total']:.0f}{marca} "
            f"| {e['pasos']} | `{e['video']}` |"
        )
    desviacion = statistics.pstdev(puntajes) if len(puntajes) > 1 else 0.0
    filas.append("")
    filas.append(
        f"**Promedio {statistics.mean(puntajes):.1f} · "
        f"Maximo {max(puntajes):.0f} · Desviacion {desviacion:.1f}**"
    )
    return "\n".join(filas)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=CHECKPOINT_FINAL)
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=SEMILLA_EVALUACION)
    parser.add_argument("--prefijo", type=str, default="agente-entrenado-space-invaders")
    parser.add_argument("--dispositivo", type=str, default=None)
    args = parser.parse_args()

    fijar_semillas(args.semilla)
    resultado = grabar_partidas(
        args.modelo, DIR_VIDEOS, args.prefijo, args.episodios, args.semilla, args.dispositivo
    )

    salida = DIR_ENTREGABLES / "video_agente.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    mejor = resultado["mejor"]

    tabla = tabla_markdown(resultado["episodios"], mejor)
    ruta_tabla = DIR_ENTREGABLES / "tabla_resultados.md"
    ruta_tabla.write_text(tabla + "\n")
    print()
    print(tabla)
    print()
    print(f"Tabla escrita en {ruta_tabla.relative_to(RAIZ)}")
    print(f"\nMejor partida: {mejor['recompensa_total']:.0f} puntos en {mejor['video']}")
    print(f"Metadatos en {salida.relative_to(RAIZ)}")
    return resultado


if __name__ == "__main__":
    main()
