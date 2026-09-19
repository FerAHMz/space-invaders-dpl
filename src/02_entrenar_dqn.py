"""Etapa 2: entrena el agente Rainbow-lite sobre ALE/SpaceInvaders-v5.

Cada corrida se identifica con una etiqueta y deja tres rastros en disco:
  logs/<etiqueta>_episodios.csv   una fila por episodio de entrenamiento
  logs/<etiqueta>_eval.csv        una fila por evaluacion greedy periodica
  modelos/<etiqueta>.pt           ultimo checkpoint (pesos + preprocesamiento)
  modelos/<etiqueta>_mejor.pt     checkpoint de la mejor evaluacion vista

El entrenamiento es reanudable: --reanudar retoma los pesos y el contador de
pasos del ultimo checkpoint. El replay buffer NO se guarda (ocupa gigabytes),
asi que al reanudar se vuelve a llenar con la politica ya aprendida, que es
mejor que la aleatoria del arranque original.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agente import AgenteRainbow, elegir_dispositivo
from ale_utils import fijar_semillas
from config import (
    CAPACIDAD_REPLAY,
    DIR_LOGS,
    DIR_MODELOS,
    FRECUENCIA_ENTRENAMIENTO,
    FRECUENCIA_TARGET,
    INICIO_APRENDIZAJE,
    PER_BETA_INICIAL,
    SEMILLA,
)
from wrappers import crear_entorno_entrenamiento, crear_entorno_evaluacion


def evaluar(agente: AgenteRainbow, n_episodios: int, semilla: int, max_pasos: int = 30000) -> dict:
    """Corre episodios con politica greedy (sin ruido) y sin clipping de recompensa."""
    env = crear_entorno_evaluacion()
    puntajes = []
    try:
        for i in range(n_episodios):
            obs, _ = env.reset(seed=semilla + i)
            env.action_space.seed(semilla + i)
            total, pasos, listo = 0.0, 0, False
            while not listo and pasos < max_pasos:
                accion = agente.actuar(obs, entrenando=False)
                obs, recompensa, terminated, truncated, _ = env.step(accion)
                total += float(recompensa)
                pasos += 1
                listo = terminated or truncated
            puntajes.append(total)
    finally:
        env.close()
    return {
        "puntajes": puntajes,
        "promedio": float(np.mean(puntajes)),
        "maximo": float(np.max(puntajes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasos", type=int, default=2_000_000,
                        help="pasos de agente a EJECUTAR en esta invocacion (cada uno son 4 frames)")
    parser.add_argument("--hasta", type=int, default=None,
                        help="objetivo ABSOLUTO de pasos. A diferencia de --pasos, no se suma a "
                             "lo ya entrenado: entrenar hasta 12M sigue siendo 12M aunque la "
                             "corrida se reanude varias veces. Es lo que hay que usar cuando la "
                             "sesion se puede cortar (Colab), porque ademas mantiene estable el "
                             "calendario de beta de PER entre reanudaciones.")
    parser.add_argument("--etiqueta", type=str, default="rainbow",
                        help="identificador de la iteracion; nombra logs y checkpoints")
    parser.add_argument("--dispositivo", type=str, default=None, help="cuda | mps | cpu")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--eval-cada", type=int, default=100_000,
                        help="pasos entre evaluaciones greedy")
    parser.add_argument("--eval-episodios", type=int, default=3)
    parser.add_argument("--guardar-cada", type=int, default=50_000)
    parser.add_argument("--capacidad-replay", type=int, default=CAPACIDAD_REPLAY,
                        help="transiciones del replay buffer. El valor por defecto esta "
                             "dimensionado para 16 GB de RAM; con 32 GB o mas conviene subirlo "
                             "a 1000000, que es el del paper de Rainbow: un buffer mas grande "
                             "conserva experiencia mas antigua y diversa, lo que reduce el "
                             "sobreajuste a la politica reciente. Cuesta ~7 GB de RAM.")
    parser.add_argument("--inicio-aprendizaje", type=int, default=INICIO_APRENDIZAJE,
                        help="pasos de recoleccion antes del primer gradiente")
    parser.add_argument("--reanudar", action="store_true",
                        help="retoma desde modelos/<etiqueta>.pt")
    args = parser.parse_args()

    fijar_semillas(args.semilla)
    torch.manual_seed(args.semilla)

    dispositivo = elegir_dispositivo(args.dispositivo)
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    env = crear_entorno_entrenamiento(semilla=args.semilla)
    n_acciones = env.action_space.n

    checkpoint = DIR_MODELOS / f"{args.etiqueta}.pt"
    mejor_checkpoint = DIR_MODELOS / f"{args.etiqueta}_mejor.pt"

    agente = AgenteRainbow(n_acciones, dispositivo, capacidad_replay=args.capacidad_replay)
    paso_inicial, mejor_promedio = 0, -float("inf")
    if args.reanudar and checkpoint.exists():
        punto = torch.load(checkpoint, map_location=dispositivo, weights_only=False)
        agente.red.load_state_dict(punto["red"])
        agente.optimizador.load_state_dict(punto["optimizador"])
        agente.sincronizar_objetivo()
        paso_inicial = int(punto["metadatos"].get("paso", 0))
        mejor_promedio = float(punto["metadatos"].get("mejor_promedio", -float("inf")))
        print(f"Reanudando {args.etiqueta} desde el paso {paso_inicial:,}")

    csv_episodios = DIR_LOGS / f"{args.etiqueta}_episodios.csv"
    csv_eval = DIR_LOGS / f"{args.etiqueta}_eval.csv"
    if not csv_episodios.exists():
        csv_episodios.write_text("paso,episodio,puntaje,pasos_episodio,perdida_media,pasos_por_segundo\n")
    if not csv_eval.exists():
        csv_eval.write_text("paso,promedio,maximo,puntajes\n")

    obs, _ = env.reset(seed=args.semilla)
    inicio_episodio = True
    ultimos_puntajes = deque(maxlen=50)
    perdidas = deque(maxlen=1000)
    episodio = 0
    puntaje_partida = 0.0
    pasos_partida = 0
    t0 = time.time()
    pasos_totales = args.hasta if args.hasta is not None else paso_inicial + args.pasos
    if pasos_totales <= paso_inicial:
        print(f"Nada que hacer: ya se entrenaron {paso_inicial:,} pasos (objetivo {pasos_totales:,})")
        env.close()
        return
    print(f"Entrenando del paso {paso_inicial:,} al {pasos_totales:,} en {dispositivo}")

    for paso in range(paso_inicial, pasos_totales):
        accion = agente.actuar(obs, entrenando=True)
        obs_sig, recompensa, terminated, truncated, info = env.step(accion)

        # Se guarda el frame mas reciente de la pila: el buffer reconstruye la
        # pila completa al muestrear (ver replay.py).
        agente.buffer.agregar(
            np.asarray(obs)[-1], accion, float(recompensa), terminated, inicio_episodio
        )
        inicio_episodio = False

        puntaje_partida += float(recompensa)
        pasos_partida += 1
        obs = obs_sig

        if terminated or truncated:
            # Con EpisodicLife esto ocurre al perder cada vida; el puntaje real
            # de la partida completa lo reporta RecordEpisodeStatistics en
            # info["episode"], que vive antes del clipping de recompensa.
            if "episode" in info:
                puntaje_real = float(info["episode"]["r"])
                ultimos_puntajes.append(puntaje_real)
                episodio += 1
                velocidad = (paso - paso_inicial + 1) / (time.time() - t0)
                with csv_episodios.open("a") as f:
                    csv.writer(f).writerow([
                        paso + 1, episodio, puntaje_real, int(info["episode"]["l"]),
                        round(float(np.mean(perdidas)), 5) if perdidas else "",
                        round(velocidad, 1),
                    ])
                if episodio % 5 == 0:
                    print(
                        f"paso {paso + 1:>9,} | episodio {episodio:>5} | "
                        f"puntaje {puntaje_real:>6.0f} | media50 {np.mean(ultimos_puntajes):>7.1f} | "
                        f"perdida {np.mean(perdidas) if perdidas else 0:>6.3f} | "
                        f"{velocidad:>5.1f} pasos/s"
                    )
            obs, _ = env.reset()
            inicio_episodio = True
            puntaje_partida, pasos_partida = 0.0, 0

        if paso >= args.inicio_aprendizaje and paso % FRECUENCIA_ENTRENAMIENTO == 0:
            # beta sube linealmente de PER_BETA_INICIAL a 1: al principio se
            # tolera el sesgo del muestreo priorizado a cambio de aprender mas
            # rapido, y al final se corrige por completo.
            progreso = (paso - args.inicio_aprendizaje) / max(1, pasos_totales - args.inicio_aprendizaje)
            beta = PER_BETA_INICIAL + (1.0 - PER_BETA_INICIAL) * min(1.0, progreso)
            perdida = agente.aprender(beta)
            if perdida is not None:
                perdidas.append(perdida)

        if paso > 0 and paso % FRECUENCIA_TARGET == 0:
            agente.sincronizar_objetivo()

        if paso > 0 and paso % args.guardar_cada == 0:
            agente.guardar(checkpoint, {"paso": paso, "mejor_promedio": mejor_promedio,
                                        "etiqueta": args.etiqueta})

        if paso > 0 and paso % args.eval_cada == 0 and paso >= args.inicio_aprendizaje:
            resultado = evaluar(agente, args.eval_episodios, semilla=10_000 + paso)
            with csv_eval.open("a") as f:
                csv.writer(f).writerow([
                    paso, resultado["promedio"], resultado["maximo"],
                    json.dumps(resultado["puntajes"]),
                ])
            print(f">>> EVAL paso {paso:,}: promedio {resultado['promedio']:.1f} "
                  f"maximo {resultado['maximo']:.0f} {resultado['puntajes']}")
            if resultado["promedio"] > mejor_promedio:
                mejor_promedio = resultado["promedio"]
                agente.guardar(mejor_checkpoint, {"paso": paso, "mejor_promedio": mejor_promedio,
                                                  "etiqueta": args.etiqueta,
                                                  "eval": resultado})
                print(f">>> nuevo mejor modelo guardado en {mejor_checkpoint.name}")

    agente.guardar(checkpoint, {"paso": pasos_totales, "mejor_promedio": mejor_promedio,
                                "etiqueta": args.etiqueta})
    env.close()
    print(f"Entrenamiento terminado. Checkpoint en {checkpoint}")


if __name__ == "__main__":
    main()
