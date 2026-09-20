"""Etapa 3: evalua el agente entrenado con la configuracion exacta de la competencia.

Corre EPISODIOS_EVALUACION episodios con politica greedy sobre
ALE/SpaceInvaders-v5 y reporta el puntaje de cada uno, su promedio y el maximo
(que es la metrica del ranking). El preprocesamiento sale del checkpoint, de
modo que no puede desalinearse con el que se uso al entrenar.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agente import AgenteRainbow, elegir_dispositivo
from ale_utils import fijar_semillas
from config import (
    CHECKPOINT_FINAL,
    DIR_ENTREGABLES,
    RAIZ,
    EPISODIOS_EVALUACION,
    MAX_STEPS_EVALUACION,
    SEMILLA_EVALUACION,
)
from wrappers import crear_entorno_evaluacion


def evaluar_agente(
    ruta_modelo: Path,
    n_episodios: int = EPISODIOS_EVALUACION,
    semilla: int = SEMILLA_EVALUACION,
    max_pasos: int = MAX_STEPS_EVALUACION,
    dispositivo: str = None,
) -> dict:
    dev = elegir_dispositivo(dispositivo)
    agente = AgenteRainbow.cargar(ruta_modelo, dev)

    env = crear_entorno_evaluacion()
    episodios = []
    try:
        for i in range(n_episodios):
            obs, _ = env.reset(seed=semilla + i)
            env.action_space.seed(semilla + i)
            total, pasos, terminated, truncated = 0.0, 0, False, False
            while pasos < max_pasos and not (terminated or truncated):
                accion = agente.actuar(obs, entrenando=False)
                obs, recompensa, terminated, truncated, _ = env.step(accion)
                total += float(recompensa)
                pasos += 1
            episodios.append({
                "episodio": i,
                "semilla": semilla + i,
                "pasos": pasos,
                "recompensa_total": total,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            print(f"episodio {i}: {total:.0f} puntos en {pasos} pasos")
    finally:
        env.close()

    puntajes = [e["recompensa_total"] for e in episodios]
    # Ruta relativa a la raiz del repo: el JSON se versiona y una ruta absoluta
    # solo tiene sentido en la maquina que lo genero.
    try:
        ruta_reportada = str(Path(ruta_modelo).resolve().relative_to(RAIZ))
    except ValueError:
        ruta_reportada = str(ruta_modelo)
    return {
        "modelo": ruta_reportada,
        "preprocesamiento": getattr(agente, "preprocesamiento", {}),
        "metadatos_entrenamiento": getattr(agente, "metadatos", {}),
        "episodios": episodios,
        "promedio": float(np.mean(puntajes)),
        "maximo": float(np.max(puntajes)),
        "minimo": float(np.min(puntajes)),
        "desviacion": float(np.std(puntajes)),
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=CHECKPOINT_FINAL)
    parser.add_argument("--episodios", type=int, default=EPISODIOS_EVALUACION)
    parser.add_argument("--semilla", type=int, default=SEMILLA_EVALUACION)
    parser.add_argument("--dispositivo", type=str, default=None)
    parser.add_argument("--salida", type=Path, default=DIR_ENTREGABLES / "evaluacion.json")
    args = parser.parse_args()

    fijar_semillas(args.semilla)
    resultado = evaluar_agente(
        args.modelo, args.episodios, args.semilla, dispositivo=args.dispositivo
    )

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(
        f"\npromedio {resultado['promedio']:.1f} | maximo {resultado['maximo']:.0f} "
        f"| desviacion {resultado['desviacion']:.1f}"
    )
    print(f"Resultados escritos en {args.salida}")
    return resultado


if __name__ == "__main__":
    main()
