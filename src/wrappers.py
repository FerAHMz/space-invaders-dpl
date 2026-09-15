"""Wrappers de preprocesamiento del entorno Atari para el agente de RL.

El pipeline de observacion replica el de Mnih et al. (2015) con los ajustes
que pide Space Invaders, y es EXACTAMENTE el mismo en entrenamiento y en
evaluacion: cualquier diferencia entre ambos invalida los pesos aprendidos.

    ALE/SpaceInvaders-v5 (frameskip=1, sticky=0.25)
      -> NoopReset            estado inicial aleatorio
      -> AtariPreprocessing   frameskip=4 con max-pool, gris, 84x84
      -> FireReset            aprieta FIRE al empezar para lanzar la partida
      -> EpisodicLife         cada vida es un episodio para el aprendizaje
      -> ClipReward           recompensa recortada a su signo (solo entreno)
      -> FrameStackObservation(4)
"""

from typing import Optional

import ale_py
import gymnasium as gym
import numpy as np

from config import (
    ENTORNO_SPACE_INVADERS,
    FRAMES_APILADOS,
    FRAME_SKIP,
    NOOP_MAX,
    TAMANO_FRAME,
)

gym.register_envs(ale_py)


class FireReset(gym.Wrapper):
    """Aprieta FIRE despues de cada reset.

    Space Invaders se queda congelado en la pantalla de inicio hasta que se
    envia FIRE. Sin este wrapper un agente que aun no aprendio a disparar
    puede gastar miles de pasos sin que el juego arranque siquiera.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        significados = env.unwrapped.get_action_meanings()
        assert significados[1] == "FIRE", "Se esperaba FIRE en el indice 1"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLife(gym.Wrapper):
    """Trata la perdida de una vida como fin de episodio para el aprendizaje.

    El agente recibe terminated=True al perder una vida, de modo que el valor
    bootstrapeado del siguiente estado no se propaga a traves de la muerte y el
    agente aprende que morir es malo. El entorno real NO se reinicia: el reset
    siguiente solo continua la partida, asi que el puntaje que se acumula
    afuera sigue siendo el de la partida completa de tres vidas.

    Este wrapper se usa unicamente en entrenamiento. En evaluacion se quita,
    porque ahi el episodio es la partida completa tal como la mide la
    competencia.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.vidas = 0
        self.fin_real = True

    def step(self, action):
        obs, recompensa, terminated, truncated, info = self.env.step(action)
        self.fin_real = terminated or truncated
        vidas = self.env.unwrapped.ale.lives()
        if 0 < vidas < self.vidas:
            # Se perdio una vida pero la partida sigue: fin de episodio para el
            # agente, no para el emulador.
            terminated = True
        self.vidas = vidas
        return obs, recompensa, terminated, truncated, info

    def reset(self, **kwargs):
        if self.fin_real:
            obs, info = self.env.reset(**kwargs)
        else:
            # Avanza un paso NOOP para salir del frame de muerte sin perder el
            # estado del emulador.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.vidas = self.env.unwrapped.ale.lives()
        return obs, info


def crear_entorno_entrenamiento(
    semilla: Optional[int] = None,
    clip_reward: bool = True,
    episodic_life: bool = True,
    render_mode: Optional[str] = None,
    video_folder: Optional[str] = None,
    name_prefix: str = "agente",
) -> gym.Env:
    """Crea el entorno con el preprocesamiento completo.

    Args:
        semilla: semilla del entorno y de su espacio de accion.
        clip_reward: recorta la recompensa a {-1, 0, 1}. True en entrenamiento
            (estabiliza el gradiente entre invasores de 5 y naves de 200 puntos),
            False en evaluacion para que el return sea el puntaje real.
        episodic_life: activa EpisodicLife. Solo en entrenamiento.
        render_mode: "rgb_array" cuando se va a grabar o mostrar.
        video_folder: si no es None, envuelve en RecordVideo.
        name_prefix: prefijo de los .mp4 generados.
    """
    if video_folder is not None:
        render_mode = "rgb_array"

    # frameskip=1: el salto de frames y el max-pooling los hace
    # AtariPreprocessing. repeat_action_probability queda en 0.25 (defecto v5).
    env = gym.make(
        ENTORNO_SPACE_INVADERS,
        frameskip=1,
        render_mode=render_mode,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if video_folder is not None:
        from pathlib import Path

        Path(video_folder).mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=NOOP_MAX,
        frame_skip=FRAME_SKIP,
        screen_size=TAMANO_FRAME,
        terminal_on_life_loss=False,  # lo maneja EpisodicLife, que si permite continuar
        grayscale_obs=True,
        scale_obs=False,  # se deja en uint8: el replay buffer ocupa 4x menos memoria
    )
    env = FireReset(env)

    if episodic_life:
        env = EpisodicLife(env)
    if clip_reward:
        env = gym.wrappers.ClipReward(env, -1.0, 1.0)

    env = gym.wrappers.FrameStackObservation(env, FRAMES_APILADOS)

    if semilla is not None:
        env.reset(seed=semilla)
        env.action_space.seed(semilla)

    return env


def crear_entorno_evaluacion(
    semilla: Optional[int] = None,
    video_folder: Optional[str] = None,
    name_prefix: str = "agente-entrenado",
) -> gym.Env:
    """Entorno de evaluacion: mismo preprocesamiento, sin clipping ni vidas episodicas.

    Es la configuracion con la que se mide el puntaje de la competencia, por lo
    que el return del episodio es literalmente el marcador de la partida.
    """
    return crear_entorno_entrenamiento(
        semilla=semilla,
        clip_reward=False,
        episodic_life=False,
        video_folder=video_folder,
        name_prefix=name_prefix,
    )
