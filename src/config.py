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
