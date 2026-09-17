"""Buffer de repeticion de experiencias priorizado, con n-step y frames comprimidos.

Dos decisiones de implementacion que condicionan todo lo demas:

1. **Frames comprimidos.** Guardar la observacion apilada completa (4x84x84) de
   s y s' para cada transicion cuesta ~56 KB por transicion: 300 000
   transiciones serian casi 17 GB. En vez de eso se guarda UN solo frame de
   84x84 por paso en un arreglo circular y la pila de 4 se reconstruye al
   muestrear leyendo los indices i-3..i. El costo baja a ~2.1 GB, que si cabe
   en memoria.

2. **n-step al muestrear.** El retorno de n pasos no se acumula al insertar
   sino que se calcula en el momento del muestreo a partir de la ventana
   [i, i+n). Asi el buffer guarda la trayectoria cruda y n puede cambiarse
   entre iteraciones sin reconstruir nada.
"""

from typing import Tuple

import numpy as np

from config import FRAMES_APILADOS, TAMANO_FRAME


class ArbolSuma:
    """Arbol de sumas sobre un arreglo de prioridades.

    Permite muestrear un indice con probabilidad proporcional a su prioridad en
    O(log n) y actualizar una prioridad en O(log n). Con capacidad de cientos de
    miles de transiciones, recalcular la suma total en cada muestreo (O(n))
    dominaria el tiempo de entrenamiento.
    """

    def __init__(self, capacidad: int):
        self.capacidad = capacidad
        # El descenso por el arbol asume que cada nivel tiene el doble de nodos
        # que el anterior, asi que las hojas se rellenan hasta la siguiente
        # potencia de 2. Las hojas de relleno quedan en prioridad 0 y nunca
        # salen muestreadas.
        self.hojas = 1
        while self.hojas < capacidad:
            self.hojas *= 2
        self.arbol = np.zeros(2 * self.hojas, dtype=np.float64)

    def actualizar(self, indices: np.ndarray, prioridades: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64) + self.hojas
        self.arbol[indices] = prioridades
        # Sube por los niveles recomputando cada padre a partir de sus dos hijos.
        indices = np.unique(indices // 2)
        while indices.size and indices[0] >= 1:
            self.arbol[indices] = self.arbol[2 * indices] + self.arbol[2 * indices + 1]
            if indices[0] == 1:
                break
            indices = np.unique(indices // 2)

    @property
    def total(self) -> float:
        return float(self.arbol[1])

    def prioridad(self, indices: np.ndarray) -> np.ndarray:
        return self.arbol[np.asarray(indices, dtype=np.int64) + self.hojas]

    def buscar(self, valores: np.ndarray) -> np.ndarray:
        """Para cada valor en [0, total) retorna el indice de hoja correspondiente."""
        indices = np.ones(len(valores), dtype=np.int64)
        valores = valores.copy()
        while indices[0] < self.hojas:
            izquierda = 2 * indices
            ir_derecha = valores > self.arbol[izquierda]
            valores = np.where(ir_derecha, valores - self.arbol[izquierda], valores)
            indices = izquierda + ir_derecha.astype(np.int64)
        return np.clip(indices - self.hojas, 0, self.capacidad - 1)


class BufferPriorizado:
    """Replay buffer con muestreo priorizado por el error TD."""

    def __init__(
        self,
        capacidad: int,
        n_step: int,
        gamma: float,
        alpha: float,
        frames_apilados: int = FRAMES_APILADOS,
    ):
        self.capacidad = capacidad
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.k = frames_apilados

        self.frames = np.zeros((capacidad, TAMANO_FRAME, TAMANO_FRAME), dtype=np.uint8)
        self.acciones = np.zeros(capacidad, dtype=np.int64)
        self.recompensas = np.zeros(capacidad, dtype=np.float32)
        self.terminales = np.zeros(capacidad, dtype=bool)
        # Marca el primer paso de cada episodio: la pila de 4 frames no puede
        # cruzar hacia el episodio anterior.
        self.inicios = np.zeros(capacidad, dtype=bool)

        self.cursor = 0
        self.tamano = 0
        self.arbol = ArbolSuma(capacidad)
        self.prioridad_maxima = 1.0

        self._descuentos = (gamma ** np.arange(n_step)).astype(np.float32)

    def __len__(self) -> int:
        return self.tamano

    def agregar(
        self,
        frame: np.ndarray,
        accion: int,
        recompensa: float,
        terminal: bool,
        inicio_episodio: bool,
    ) -> None:
        """Guarda un paso. `frame` es el frame MAS RECIENTE del estado observado."""
        i = self.cursor
        self.frames[i] = frame
        self.acciones[i] = accion
        self.recompensas[i] = recompensa
        self.terminales[i] = terminal
        self.inicios[i] = inicio_episodio

        # Una transicion nueva entra con la prioridad maxima vista para
        # garantizar que se muestree al menos una vez antes de ser descartada.
        self.arbol.actualizar(np.array([i]), np.array([self.prioridad_maxima**self.alpha]))

        self.cursor = (self.cursor + 1) % self.capacidad
        self.tamano = min(self.tamano + 1, self.capacidad)

    def _indices_validos(self, indices: np.ndarray) -> np.ndarray:
        """Descarta indices cuya ventana [i-k+1, i+n] no esta completamente escrita.

        Los datos validos ocupan el rango circular [cursor - tamano, cursor - 1].
        Se deja un margen de un paso en cada extremo porque la posicion del
        cursor contiene simultaneamente el dato mas viejo y el mas nuevo cuando
        el buffer ya dio la vuelta.
        """
        mas_viejo = (self.cursor - self.tamano) % self.capacidad
        valido = np.ones(len(indices), dtype=bool)

        for desplazamiento in range(-self.k + 1, self.n_step + 1):
            pos = (indices + desplazamiento) % self.capacidad
            antiguedad = (pos - mas_viejo) % self.capacidad
            valido &= (antiguedad >= 1) & (antiguedad <= self.tamano - 2)

        # La pila de k frames hacia atras no puede cruzar el inicio de un
        # episodio: mezclaria pantallas de dos partidas distintas.
        for desplazamiento in range(1, self.k):
            pos = (indices - desplazamiento) % self.capacidad
            valido &= ~self.inicios[pos]

        return valido

    def _apilar(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruye la observacion apilada (batch, k, 84, 84) terminada en cada indice."""
        desplazamientos = np.arange(-self.k + 1, 1)
        posiciones = (indices[:, None] + desplazamientos[None, :]) % self.capacidad
        return self.frames[posiciones]

    def muestrear(self, batch: int, beta: float) -> Tuple[np.ndarray, ...]:
        """Muestrea un batch priorizado.

        Returns:
            obs, acciones, retornos_n, obs_siguientes, terminados, pesos_is, indices
        """
        indices = np.empty(0, dtype=np.int64)
        intentos = 0
        while len(indices) < batch:
            valores = np.random.uniform(0.0, self.arbol.total, size=batch * 2)
            candidatos = self.arbol.buscar(valores)
            candidatos = candidatos[self._indices_validos(candidatos)]
            indices = np.concatenate([indices, candidatos])
            intentos += 1
            if intentos > 50:
                raise RuntimeError("El buffer no logro juntar un batch valido")
        indices = indices[:batch]

        obs = self._apilar(indices)

        # Retorno de n pasos, cortado en el primer terminal de la ventana.
        pasos = (indices[:, None] + np.arange(self.n_step)[None, :]) % self.capacidad
        recompensas = self.recompensas[pasos]
        terminales = self.terminales[pasos]
        # vivo[:, j] indica que ningun paso anterior a j fue terminal.
        vivo = np.concatenate(
            [np.ones((batch, 1), dtype=bool), ~terminales[:, :-1]], axis=1
        )
        vivo = np.cumprod(vivo, axis=1).astype(np.float32)
        retornos = (recompensas * self._descuentos[None, :] * vivo).sum(axis=1)

        termino = terminales.any(axis=1)
        # Cuantos pasos reales se avanzaron antes de terminar (para el descuento
        # del bootstrap): n completo si no hubo terminal.
        pasos_reales = np.where(termino, terminales.argmax(axis=1) + 1, self.n_step)

        indices_siguientes = (indices + pasos_reales) % self.capacidad
        obs_siguientes = self._apilar(indices_siguientes)

        # Pesos de importance sampling: corrigen el sesgo que introduce muestrear
        # no uniformemente. beta va de PER_BETA_INICIAL a 1 durante el entrenamiento.
        prioridades = self.arbol.prioridad(indices)
        probabilidades = prioridades / self.arbol.total
        pesos = (self.tamano * probabilidades) ** (-beta)
        pesos = (pesos / pesos.max()).astype(np.float32)

        return (
            obs,
            self.acciones[indices],
            retornos.astype(np.float32),
            obs_siguientes,
            termino,
            pesos,
            indices,
            pasos_reales.astype(np.int64),
        )

    def actualizar_prioridades(self, indices: np.ndarray, errores: np.ndarray) -> None:
        """Reasigna prioridades a partir del error TD absoluto de cada transicion."""
        prioridades = np.abs(errores) + 1e-6
        self.prioridad_maxima = max(self.prioridad_maxima, float(prioridades.max()))
        self.arbol.actualizar(indices, prioridades**self.alpha)
