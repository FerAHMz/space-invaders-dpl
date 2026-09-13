"""Funciones reutilizables para crear entornos de ALE, correr agentes y grabar video."""

from pathlib import Path
from typing import Callable, Optional

import ale_py
import gymnasium as gym
import numpy as np

from config import COLOR_JUGADOR, FILAS_INVASORES, FILAS_JUGADOR

# Registra los entornos ALE/* en el catálogo de Gymnasium. Sin esta llamada,
# gym.make("ALE/SpaceInvaders-v5") levanta un NamespaceNotFound.
gym.register_envs(ale_py)


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas de numpy y random para que las corridas sean reproducibles."""
    import random

    random.seed(semilla)
    np.random.seed(semilla)


def crear_entorno(
    nombre_entorno: str,
    video_folder: Optional[str] = None,
    name_prefix: str = "video",
    episode_trigger: Optional[Callable[[int], bool]] = None,
    render_mode: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    """Crea un entorno de Gymnasium, opcionalmente envuelto para grabar video.

    Args:
        nombre_entorno: id registrado en Gymnasium (ej. "ALE/SpaceInvaders-v5").
        video_folder: carpeta destino de los .mp4. Si es None no se graba.
        name_prefix: prefijo de los archivos de video generados.
        episode_trigger: función episodio -> bool que decide qué episodios grabar.
            Por defecto se graban todos.
        render_mode: modo de render. Al grabar video se fuerza "rgb_array",
            que es el único que entrega frames a RecordVideo.
        **kwargs: parámetros extra del entorno (frameskip, obs_type,
            repeat_action_probability, full_action_space, ...).

    Returns:
        El entorno listo para usar; si hay grabación, envuelto en RecordVideo.
    """
    if video_folder is not None:
        render_mode = "rgb_array"

    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=episode_trigger if episode_trigger is not None else (lambda ep: True),
            disable_logger=True,
        )

    return env


def agente_aleatorio(observation, env: gym.Env) -> int:
    """Política baseline: ignora la observación y muestrea del espacio de acción."""
    return env.action_space.sample()


def _columna_media(mascara: np.ndarray) -> Optional[float]:
    """Retorna la columna promedio de los píxeles activos de una máscara, o None si está vacía."""
    columnas = np.nonzero(mascara)[1]
    if columnas.size == 0:
        return None
    return float(columnas.mean())


def agente_regla_simple(observation, env: gym.Env) -> int:
    """Política heurística: alinea el cañón con los invasores y dispara.

    Localiza el cañón por su color verde en las filas inferiores del frame y el
    bloque de invasores en la banda central, y se mueve disparando hacia el lado
    donde están los invasores. Si no logra localizar alguno de los dos (por
    ejemplo con observaciones en RAM o preprocesadas a escala de grises),
    retrocede a disparar en el lugar.
    """
    accion_por_defecto = 1  # FIRE

    frame = np.asarray(observation)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return accion_por_defecto

    banda_jugador = frame[FILAS_JUGADOR[0] : FILAS_JUGADOR[1]]
    mascara_jugador = np.all(banda_jugador == np.array(COLOR_JUGADOR), axis=2)
    x_jugador = _columna_media(mascara_jugador)

    banda_invasores = frame[FILAS_INVASORES[0] : FILAS_INVASORES[1]]
    mascara_invasores = banda_invasores.any(axis=2)
    x_invasores = _columna_media(mascara_invasores)

    if x_jugador is None or x_invasores is None:
        return accion_por_defecto

    diferencia = x_invasores - x_jugador
    if abs(diferencia) <= 3:
        return accion_por_defecto
    return 4 if diferencia > 0 else 5  # RIGHTFIRE / LEFTFIRE


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: Callable,
    max_steps: int = 10000,
    seed: Optional[int] = None,
) -> dict:
    """Corre un episodio completo y retorna sus métricas.

    El loop avanza hasta que el entorno reporta terminated o truncated, o hasta
    agotar max_steps.

    Returns:
        dict con pasos, recompensa_total (return del episodio), terminated y truncated.
    """
    observation, _ = env.reset(seed=seed)

    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    while pasos < max_steps and not (terminated or truncated):
        accion = funcion_agente(observation, env)
        observation, recompensa, terminated, truncated, _ = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
