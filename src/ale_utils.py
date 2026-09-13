"""Funciones reutilizables para crear entornos de ALE, correr agentes y grabar video."""

from pathlib import Path
from typing import Callable, Optional

import ale_py
import gymnasium as gym
import numpy as np

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
