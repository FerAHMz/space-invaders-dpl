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

---

# Proyecto 2 — Señales del agente entrenado

El Proyecto 2 reutiliza el mismo entorno del laboratorio, pero la observación
que ve la red no es el frame crudo: pasa por una cadena de wrappers. Esta
sección documenta esa transformación y las señales nuevas que produce el
entrenamiento.

## Cadena de preprocesamiento

Se aplica en este orden, y es **idéntica en entrenamiento y en evaluación**
(`src/wrappers.py`). Cualquier diferencia entre ambas invalidaría los pesos.

| # | Wrapper | Parámetros | Por qué |
| --- | --- | --- | --- |
| 0 | `gym.make(..., frameskip=1)` | sticky = 0.25 | Se desactiva el frameskip del entorno para que lo haga `AtariPreprocessing`, que además aplica max-pooling. Las sticky actions se dejan en 0.25 porque es el valor con el que se evalúa. |
| 1 | `RecordEpisodeStatistics` | — | Registra el puntaje **real** de la partida antes de cualquier recorte. |
| 2 | `AtariPreprocessing` | `noop_max=30`, `frame_skip=4`, `screen_size=84`, gris | Reduce `(210,160,3)` a `(84,84)` en `uint8`: 24x menos datos por frame. El max-pooling sobre los 2 últimos frames es crítico en Space Invaders porque los láseres parpadean en frames alternos y sin él desaparecen de la observación. |
| 3 | `FireReset` | — | Space Invaders se queda congelado hasta recibir `FIRE`; sin esto un agente que aún no aprende a disparar gasta miles de pasos sin que la partida arranque. |
| 4 | `EpisodicLife` | solo entrenamiento | Marca `terminated=True` al perder una vida sin reiniciar el emulador, para que el valor del estado siguiente no se propague a través de la muerte. |
| 5 | `ClipReward` | solo entrenamiento, `[-1, 1]` | Un invasor vale 5 y la nave nodriza 200; sin recorte el gradiente queda dominado por la nave nodriza. |
| 6 | `FrameStackObservation` | `k=4` | Un frame solo no dice hacia dónde se mueven invasores y proyectiles. Cuatro frames apilados hacen el estado aproximadamente markoviano. |

Observación final: `Box(0, 255, (4, 84, 84), uint8)`. La normalización a `[0,1]`
la hace la red en su `forward`, no un wrapper, para que el replay buffer guarde
bytes en vez de floats (4x menos memoria).

## Recompensa: análisis de la señal

- **Densidad**: moderadamente dispersa. Hay recompensa solo al destruir un
  invasor; entre disparo e impacto pasan varios pasos y muchos disparos fallan.
  En el agente aleatorio la recompensa es cero en más del 95 % de los pasos.
- **Rango**: no negativa y muy desbalanceada (5, 10, 15, 20, 25, 30 por invasor
  según la fila; 50–200 por la nave nodriza). De ahí el clipping en
  entrenamiento.
- **Reward shaping**: no se aplicó. La señal de castigo por morir la aporta
  `EpisodicLife` a través del corte del bootstrap, que es equivalente a un
  shaping implícito sin introducir sesgo en la política óptima.

## Variables registradas durante el entrenamiento

`logs/<etiqueta>_episodios.csv` — una fila por partida completa:

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `paso` | int | Paso de agente global (cada uno = 4 frames de emulación). |
| `episodio` | int | Índice de la partida dentro de la corrida. |
| `puntaje` | float | Marcador real de la partida, **sin** clipping. |
| `pasos_episodio` | int | Duración de la partida en frames de emulación. |
| `perdida_media` | float | Entropía cruzada C51 promediada sobre los últimos 1000 pasos de gradiente. |
| `pasos_por_segundo` | float | Velocidad de la corrida, para estimar costo de cómputo. |

`logs/<etiqueta>_eval.csv` — una fila por evaluación greedy periódica:

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `paso` | int | Paso de entrenamiento en que se evaluó. |
| `promedio` | float | Puntaje promedio de los episodios de evaluación. |
| `maximo` | float | Mejor episodio, que es la métrica del ranking de la competencia. |
| `puntajes` | JSON | Lista con el puntaje de cada episodio. |

`entregables/evaluacion.json` — evaluación final con la configuración exacta de
la competencia (5 episodios, política greedy), incluyendo el diccionario
`preprocesamiento` leído del checkpoint para poder auditar que coincide con el
de entrenamiento.

## Contenido del checkpoint (`modelos/*.pt`)

| Clave | Descripción |
| --- | --- |
| `red` | `state_dict` de la red online. |
| `optimizador` | Estado de Adam, para poder reanudar el entrenamiento. |
| `n_acciones`, `atomos`, `v_min`, `v_max` | Parámetros para reconstruir la arquitectura. |
| `preprocesamiento` | Configuración de wrappers (frame skip, tamaño, apilado, sticky). |
| `metadatos` | Paso alcanzado, etiqueta de la iteración y última evaluación. |
