"""Rutas, identificadores de entorno y constantes compartidas por el pipeline."""

from pathlib import Path

# Raíz del repositorio, calculada desde este archivo para que los scripts
# funcionen sin importar desde qué directorio se invoquen.
RAIZ = Path(__file__).resolve().parent.parent

DIR_ENTREGABLES = RAIZ / "entregables"
DIR_VIDEOS = DIR_ENTREGABLES / "videos"
ARCHIVO_METRICAS = DIR_ENTREGABLES / "metricas.json"

# Entorno objetivo del laboratorio. La variante v5 usa frameskip=4 y
# repeat_action_probability=0.25 (sticky actions) por defecto.
ENTORNO_SPACE_INVADERS = "ALE/SpaceInvaders-v5"

# Entorno de control de bajo nivel usado como punto de comparación.
ENTORNO_CARTPOLE = "CartPole-v1"

# Semilla única para numpy, random y los reset de los entornos.
SEMILLA = 42

# Corte de seguridad del loop de episodio: un episodio de Space Invaders
# termina mucho antes, pero evita loops infinitos si un wrapper falla.
MAX_STEPS = 10000

# Colores RGB del frame de Space Invaders usados por el agente de regla simple.
# El cañón del jugador es verde y ocupa las filas inferiores de la pantalla.
COLOR_JUGADOR = (50, 132, 50)
FILAS_JUGADOR = (185, 196)

# Banda vertical donde viven los invasores; excluye los escudos (que comparten
# el color del jugador) y el marcador superior.
FILAS_INVASORES = (30, 150)


# --------------------------------------------------------------------------
# Proyecto 2: entrenamiento del agente de Reinforcement Learning
# --------------------------------------------------------------------------

DIR_MODELOS = RAIZ / "modelos"
DIR_LOGS = RAIZ / "logs"
CHECKPOINT_FINAL = DIR_MODELOS / "rainbow_space_invaders.pt"

# Preprocesamiento. El entorno se crea con frameskip=1 para que el salto de
# frames lo haga AtariPreprocessing, que ademas aplica max-pooling sobre los
# dos ultimos frames: en Space Invaders los laseres y algunos invasores
# parpadean en frames alternos y sin ese max-pooling desaparecen de la
# observacion. repeat_action_probability se deja en su valor por defecto (0.25)
# porque es el que tendra el entorno el dia de la evaluacion.
TAMANO_FRAME = 84
FRAME_SKIP = 4
FRAMES_APILADOS = 4
NOOP_MAX = 30  # pasos NOOP aleatorios al inicio para diversificar el estado inicial

# Arquitectura y algoritmo (Rainbow-lite: Double + Dueling + PER + n-step +
# NoisyNet + distribucional C51).
ATOMOS = 51
V_MIN = -10.0
V_MAX = 10.0
SIGMA_INICIAL = 0.5  # ruido inicial de las capas NoisyLinear

# Hiperparametros de entrenamiento.
GAMMA = 0.99
N_STEP = 3
TASA_APRENDIZAJE = 6.25e-5
ADAM_EPS = 1.5e-4
# batch=64 y no 32: en MPS el costo de un paso de aprendizaje esta dominado por
# el lanzamiento de kernels, no por la aritmetica, asi que 64 muestras cuestan
# casi lo mismo que 32 (41.6 vs 43.9 pasos/s medidos) y duplican las muestras
# vistas por paso de entorno.
TAMANO_BATCH = 64
CAPACIDAD_REPLAY = 300_000
INICIO_APRENDIZAJE = 20_000
FRECUENCIA_ENTRENAMIENTO = 4
FRECUENCIA_TARGET = 8_000
PER_ALPHA = 0.5
PER_BETA_INICIAL = 0.4
MAX_GRAD_NORM = 10.0

# Evaluacion (identica a la de la competencia: 5 episodios, politica greedy).
EPISODIOS_EVALUACION = 5
SEMILLA_EVALUACION = 2026
