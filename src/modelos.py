"""Red neuronal del agente: CNN dueling distribucional con capas NoisyLinear."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ATOMOS, FRAMES_APILADOS, SIGMA_INICIAL, V_MAX, V_MIN


class NoisyLinear(nn.Module):
    """Capa densa con ruido parametrico factorizado (Fortunato et al., 2018).

    Sustituye a la exploracion epsilon-greedy: el ruido vive en los pesos, asi
    que el agente explora de forma consistente dentro de un episodio (una
    "estrategia" ruidosa completa) en vez de dar pasos aleatorios sueltos. La
    magnitud del ruido se aprende, de modo que la red puede reducir por si sola
    la exploracion en los estados que ya domina; no hay que programar un
    calendario de epsilon a mano.
    """

    def __init__(self, entradas: int, salidas: int, sigma_inicial: float = SIGMA_INICIAL):
        super().__init__()
        self.entradas = entradas
        self.salidas = salidas
        self.sigma_inicial = sigma_inicial

        self.weight_mu = nn.Parameter(torch.empty(salidas, entradas))
        self.weight_sigma = nn.Parameter(torch.empty(salidas, entradas))
        self.register_buffer("weight_epsilon", torch.empty(salidas, entradas))

        self.bias_mu = nn.Parameter(torch.empty(salidas))
        self.bias_sigma = nn.Parameter(torch.empty(salidas))
        self.register_buffer("bias_epsilon", torch.empty(salidas))

        self.reiniciar_parametros()
        self.muestrear_ruido()

    def reiniciar_parametros(self) -> None:
        cota = 1.0 / math.sqrt(self.entradas)
        self.weight_mu.data.uniform_(-cota, cota)
        self.bias_mu.data.uniform_(-cota, cota)
        self.weight_sigma.data.fill_(self.sigma_inicial / math.sqrt(self.entradas))
        self.bias_sigma.data.fill_(self.sigma_inicial / math.sqrt(self.entradas))

    @staticmethod
    def _ruido_escalado(tamano: int, device) -> torch.Tensor:
        x = torch.randn(tamano, device=device)
        return x.sign() * x.abs().sqrt()

    def muestrear_ruido(self) -> None:
        """Remuestrea el ruido factorizado: epsilon_ij = f(eps_i) * f(eps_j)."""
        eps_entrada = self._ruido_escalado(self.entradas, self.weight_mu.device)
        eps_salida = self._ruido_escalado(self.salidas, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_salida.outer(eps_entrada))
        self.bias_epsilon.copy_(eps_salida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            peso = self.weight_mu + self.weight_sigma * self.weight_epsilon
            sesgo = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # En evaluacion se usa la media: politica greedy y determinista.
            peso, sesgo = self.weight_mu, self.bias_mu
        return F.linear(x, peso, sesgo)


class RainbowCNN(nn.Module):
    """Red dueling distribucional sobre la observacion apilada de 4x84x84.

    Torre convolucional identica a la de Mnih et al. (2015), que es la que
    corresponde a una entrada de 84x84: 32 filtros 8x8 stride 4 (reduce a 20x20
    y captura el movimiento grueso de la formacion de invasores), 64 filtros
    4x4 stride 2 (a 9x9) y 64 filtros 3x3 stride 1 (a 7x7), con ReLU. Aplana a
    3136 caracteristicas.

    Sobre esa torre van dos cabezas NoisyLinear:
      - valor V(s), con ATOMOS salidas;
      - ventaja A(s,a), con acciones * ATOMOS salidas;
    y se recombinan como Q = V + (A - media_a A). La separacion importa en
    Space Invaders porque la mayoria de los estados tienen un valor parecido
    sin importar la accion (todo el tiempo conviene disparar), y la cabeza de
    valor puede aprenderlo sin tener que estimar las seis acciones por separado.

    La salida no es un escalar por accion sino una distribucion categorica de
    ATOMOS atomos sobre el retorno (C51, Bellemare et al., 2017). Aprender la
    distribucion completa en vez de solo su media da un gradiente mucho mas
    informativo y es el ingrediente individual de Rainbow con mayor impacto.
    """

    def __init__(self, n_acciones: int, atomos: int = ATOMOS):
        super().__init__()
        self.n_acciones = n_acciones
        self.atomos = atomos
        self.register_buffer("soportes", torch.linspace(V_MIN, V_MAX, atomos))

        self.convoluciones = nn.Sequential(
            nn.Conv2d(FRAMES_APILADOS, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        caracteristicas = 3136  # 64 * 7 * 7

        self.valor_oculto = NoisyLinear(caracteristicas, 512)
        self.valor_salida = NoisyLinear(512, atomos)
        self.ventaja_oculta = NoisyLinear(caracteristicas, 512)
        self.ventaja_salida = NoisyLinear(512, n_acciones * atomos)

    def muestrear_ruido(self) -> None:
        for modulo in self.modules():
            if isinstance(modulo, NoisyLinear):
                modulo.muestrear_ruido()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Retorna log-probabilidades de forma (batch, n_acciones, atomos)."""
        # La observacion llega en uint8 0-255; se normaliza aqui y no en un
        # wrapper para que el replay buffer guarde bytes en vez de floats.
        x = obs.float().div_(255.0)
        x = self.convoluciones(x)

        valor = self.valor_salida(F.relu(self.valor_oculto(x))).view(-1, 1, self.atomos)
        ventaja = self.ventaja_salida(F.relu(self.ventaja_oculta(x)))
        ventaja = ventaja.view(-1, self.n_acciones, self.atomos)

        logits = valor + ventaja - ventaja.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2)

    def valores_q(self, obs: torch.Tensor) -> torch.Tensor:
        """Valor esperado de cada accion: Q(s,a) = sum_i z_i * p_i(s,a)."""
        return (self.forward(obs).exp() * self.soportes).sum(dim=2)
