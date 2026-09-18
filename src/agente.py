"""Agente Rainbow-lite: Double DQN + Dueling + PER + n-step + NoisyNet + C51."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ADAM_EPS,
    ATOMOS,
    CAPACIDAD_REPLAY,
    GAMMA,
    MAX_GRAD_NORM,
    N_STEP,
    PER_ALPHA,
    TAMANO_BATCH,
    TASA_APRENDIZAJE,
    V_MAX,
    V_MIN,
)
from modelos import RainbowCNN
from replay import BufferPriorizado


def elegir_dispositivo(preferido: Optional[str] = None) -> torch.device:
    """Escoge el mejor dispositivo disponible: CUDA, luego MPS (Apple Silicon), luego CPU."""
    if preferido:
        return torch.device(preferido)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AgenteRainbow:
    def __init__(
        self,
        n_acciones: int,
        dispositivo: torch.device,
        capacidad_replay: int = CAPACIDAD_REPLAY,
        n_step: int = N_STEP,
        gamma: float = GAMMA,
        tasa_aprendizaje: float = TASA_APRENDIZAJE,
        batch: int = TAMANO_BATCH,
        atomos: int = ATOMOS,
    ):
        self.n_acciones = n_acciones
        self.dispositivo = dispositivo
        self.gamma = gamma
        self.n_step = n_step
        self.batch = batch
        self.atomos = atomos

        self.red = RainbowCNN(n_acciones, atomos).to(dispositivo)
        # Red objetivo: copia congelada que provee el target del error TD. Sin
        # ella el objetivo se mueve con cada gradiente y el entrenamiento diverge.
        self.red_objetivo = RainbowCNN(n_acciones, atomos).to(dispositivo)
        self.red_objetivo.load_state_dict(self.red.state_dict())
        self.red_objetivo.eval()
        for p in self.red_objetivo.parameters():
            p.requires_grad_(False)

        # eps de Adam mas grande que el de PyTorch (1e-8): con la perdida
        # distribucional los gradientes son pequenos y el eps por defecto
        # produce pasos inestables. Es el valor que usa el paper de Rainbow.
        self.optimizador = torch.optim.Adam(
            self.red.parameters(), lr=tasa_aprendizaje, eps=ADAM_EPS
        )

        self.buffer = BufferPriorizado(capacidad_replay, n_step, gamma, PER_ALPHA)

        self.soportes = torch.linspace(V_MIN, V_MAX, atomos, device=dispositivo)
        self.delta_z = (V_MAX - V_MIN) / (atomos - 1)

    # ------------------------------------------------------------------ accion

    @torch.no_grad()
    def actuar(self, observacion: np.ndarray, entrenando: bool = True) -> int:
        """Selecciona la accion de mayor Q esperado.

        No hay epsilon-greedy: la exploracion la aporta el ruido de las capas
        NoisyLinear, que solo esta activo cuando la red esta en modo train.
        """
        self.red.train(entrenando)
        obs = torch.as_tensor(np.asarray(observacion), device=self.dispositivo).unsqueeze(0)
        return int(self.red.valores_q(obs).argmax(dim=1).item())

    # ---------------------------------------------------------------- aprender

    def aprender(self, beta: float) -> Optional[float]:
        """Un paso de gradiente sobre un batch priorizado. Retorna la perdida."""
        if len(self.buffer) < self.batch + self.n_step + 8:
            return None

        (obs, acciones, retornos, obs_sig, terminados, pesos, indices, pasos) = (
            self.buffer.muestrear(self.batch, beta)
        )

        dev = self.dispositivo
        obs_t = torch.as_tensor(obs, device=dev)
        obs_sig_t = torch.as_tensor(obs_sig, device=dev)
        acciones_t = torch.as_tensor(acciones, device=dev)
        retornos_t = torch.as_tensor(retornos, device=dev)
        no_terminal = torch.as_tensor(~terminados, device=dev).float()
        pesos_t = torch.as_tensor(pesos, device=dev)
        pasos_t = torch.as_tensor(pasos, device=dev).float()

        # Ruido nuevo en cada paso de aprendizaje, tanto en la red online como
        # en la objetivo (Rainbow las desacopla).
        self.red.muestrear_ruido()
        self.red_objetivo.muestrear_ruido()

        self.red.train()
        log_p = self.red(obs_t)  # (B, A, atomos)
        log_p_a = log_p[range(self.batch), acciones_t]

        with torch.no_grad():
            # Double DQN: la ACCION del siguiente estado la elige la red online
            # y su VALOR lo estima la red objetivo. Separar ambas cosas corta el
            # sesgo optimista del max, que en Space Invaders hacia divergir los
            # valores Q en las pruebas iniciales.
            self.red_objetivo.train()  # activa el ruido de sus capas NoisyLinear
            mejores = self.red.valores_q(obs_sig_t).argmax(dim=1)
            p_sig = self.red_objetivo(obs_sig_t).exp()[range(self.batch), mejores]

            # Proyeccion categorica del target (C51): Tz = R_n + gamma^pasos * z,
            # recortado al soporte [V_MIN, V_MAX] y repartido entre los dos
            # atomos vecinos.
            descuento = (self.gamma**pasos_t) * no_terminal
            tz = retornos_t.unsqueeze(1) + descuento.unsqueeze(1) * self.soportes.unsqueeze(0)
            tz = tz.clamp(V_MIN, V_MAX)
            b = (tz - V_MIN) / self.delta_z
            inferior = b.floor().long()
            superior = b.ceil().long()
            # Cuando b cae exacto sobre un atomo, floor == ceil y toda la masa
            # se perderia; se desplazan los indices para conservarla.
            inferior[(superior > 0) & (inferior == superior)] -= 1
            superior[(inferior < self.atomos - 1) & (inferior == superior)] += 1

            objetivo = torch.zeros_like(p_sig)
            objetivo.scatter_add_(1, inferior, p_sig * (superior.float() - b))
            objetivo.scatter_add_(1, superior, p_sig * (b - inferior.float()))

        # Entropia cruzada entre la distribucion objetivo y la predicha; es la
        # divergencia KL salvo una constante. Es tambien el error TD que usa PER.
        perdidas = -(objetivo * log_p_a).sum(dim=1)
        perdida = (perdidas * pesos_t).mean()

        self.optimizador.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(self.red.parameters(), MAX_GRAD_NORM)
        self.optimizador.step()

        self.buffer.actualizar_prioridades(
            indices, perdidas.detach().cpu().numpy()
        )
        return float(perdida.item())

    def sincronizar_objetivo(self) -> None:
        self.red_objetivo.load_state_dict(self.red.state_dict())

    # ------------------------------------------------------------ persistencia

    def guardar(self, ruta: Path, metadatos: Optional[dict] = None) -> None:
        """Guarda pesos, optimizador y configuracion de preprocesamiento.

        Los metadatos incluyen los parametros de los wrappers para que la
        evaluacion pueda reconstruir el entorno exactamente igual.
        """
        from config import (
            FRAMES_APILADOS,
            FRAME_SKIP,
            NOOP_MAX,
            TAMANO_FRAME,
        )

        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "red": self.red.state_dict(),
                "optimizador": self.optimizador.state_dict(),
                "n_acciones": self.n_acciones,
                "atomos": self.atomos,
                "v_min": V_MIN,
                "v_max": V_MAX,
                "preprocesamiento": {
                    "entorno": "ALE/SpaceInvaders-v5",
                    "frameskip_env": 1,
                    "frame_skip": FRAME_SKIP,
                    "tamano_frame": TAMANO_FRAME,
                    "frames_apilados": FRAMES_APILADOS,
                    "noop_max": NOOP_MAX,
                    "grayscale": True,
                    "repeat_action_probability": 0.25,
                },
                "metadatos": metadatos or {},
            },
            ruta,
        )

    @classmethod
    def cargar(cls, ruta: Path, dispositivo: torch.device) -> "AgenteRainbow":
        punto = torch.load(ruta, map_location=dispositivo, weights_only=False)
        agente = cls(punto["n_acciones"], dispositivo, capacidad_replay=1024)
        agente.red.load_state_dict(punto["red"])
        agente.red.eval()
        agente.metadatos = punto.get("metadatos", {})
        agente.preprocesamiento = punto.get("preprocesamiento", {})
        return agente
