# Codebook — ALE/SpaceInvaders-v5

Este laboratorio no usa un dataset tabular: los "datos" son las observaciones,
acciones y recompensas que produce el entorno. Este codebook documenta esas
señales y los parámetros con los que se generan.

## Entorno

| Parámetro | Valor por defecto en v5 | Descripción |
| --- | --- | --- |
| `game` | `space_invaders` | ROM de Atari 2600 emulada por Stella. |
| `frameskip` | `4` | Frames de emulación por paso del agente; la acción se repite en todos. |
| `repeat_action_probability` | `0.25` | Sticky actions: probabilidad de repetir la acción anterior en lugar de la enviada. |
| `full_action_space` | `False` | Usa solo las 6 acciones mínimas del juego en vez de las 18 del joystick. |
| `obs_type` | `rgb` | Alternativas: `ram` (128 bytes) y `grayscale` (210x160). |
| `max_num_frames_per_episode` | `108000` | Corte por tiempo del episodio (30 min de juego a 60 Hz). |

## Espacio de observación

| Variante | Tipo | Forma | Rango | Descripción |
| --- | --- | --- | --- | --- |
| `rgb` (defecto) | `Box` `uint8` | `(210, 160, 3)` | 0–255 | Frame de pantalla del emulador. |
| `grayscale` | `Box` `uint8` | `(210, 160)` | 0–255 | Mismo frame en un solo canal. |
| `ram` | `Box` `uint8` | `(128,)` | 0–255 | Memoria completa de la consola Atari 2600. |
| Con `AtariPreprocessing` | `Box` `uint8` | `(84, 84)` | 0–255 | Escala de grises, recortado y redimensionado. |
| Con `FrameStackObservation(k=4)` | `Box` `uint8` | `(4, 84, 84)` | 0–255 | k frames consecutivos apilados para dar información de movimiento. |

Comparación: `CartPole-v1` entrega `Box(4,)` en `float32` con posición,
velocidad, ángulo y velocidad angular; es decir, el estado ya viene factorizado.
En Atari la observación es perceptual: el agente debe extraer de los píxeles
dónde está el cañón, los invasores y los proyectiles.

## Espacio de acción

`Discrete(6)`, con el siguiente significado (`env.unwrapped.get_action_meanings()`):

| Índice | Acción | Efecto |
| --- | --- | --- |
| 0 | `NOOP` | No hacer nada; el cañón se queda quieto. |
| 1 | `FIRE` | Disparar sin moverse. |
| 2 | `RIGHT` | Mover el cañón a la derecha. |
| 3 | `LEFT` | Mover el cañón a la izquierda. |
| 4 | `RIGHTFIRE` | Mover a la derecha y disparar en el mismo paso. |
| 5 | `LEFTFIRE` | Mover a la izquierda y disparar en el mismo paso. |

Con `full_action_space=True` el espacio pasa a `Discrete(18)` (el conjunto común
del joystick de Atari), aunque las 12 acciones extra no tienen efecto distinto en
este juego.

## Recompensa

La recompensa de cada paso es el incremento del marcador del juego original
durante los frames que abarca ese paso. En Space Invaders los invasores valen
entre 5 y 30 puntos según la fila que ocupan y la nave nodriza vale entre 50 y
200. No hay penalización negativa por perder una vida: la señal es no negativa y
dispersa. El *return* del episodio que reporta `ejecutar_episodio` es la suma sin
descuento de esas recompensas, es decir, el puntaje final de la partida.

`AtariPreprocessing` puede recortar la recompensa a su signo
(`clip_reward=True`), lo que estabiliza el entrenamiento pero rompe la
equivalencia entre return y puntaje; por eso las métricas reportadas se toman del
entorno sin recorte.

## Terminación

- `terminated = True` cuando el jugador pierde sus tres vidas.
- `truncated = True` cuando se alcanza `max_num_frames_per_episode` o el límite
  `max_steps` del loop propio.

## Métricas registradas por episodio (`entregables/metricas.json`)

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `episodio` | int | Índice del episodio dentro de la corrida. |
| `pasos` | int | Pasos del agente sobrevividos (cada uno equivale a 4 frames). |
| `recompensa_total` | float | Return del episodio, igual al puntaje final. |
| `terminated` | bool | El episodio terminó por fin de juego. |
| `truncated` | bool | El episodio se cortó por límite de pasos o frames. |
| `videos` | lista | Rutas relativas de los `.mp4` grabados en la corrida. |
