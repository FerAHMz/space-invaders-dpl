# Laboratorio 5 — Agentes en el Arcade Learning Environment (ALE)

Infraestructura mínima para conectar un agente a un entorno Atari 2600 a través de
Gymnasium + ALE, ejecutar episodios completos y grabar video de las partidas.
El juego objetivo es `ALE/SpaceInvaders-v5`. En este laboratorio no se entrena
ningún agente: se validan las piezas (creación de entorno, política, loop de
episodio y grabación) que después servirán de base para entrenar.

## Estructura

```
src/                      módulo y etapas del pipeline
  config.py               rutas, identificadores de entorno y constantes
  ale_utils.py            funciones reutilizables para interactuar con ALE
  01_generar_videos.py    etapa 1: graba los videos entregables y sus métricas
  run_pipeline.py         orquesta las etapas del pipeline
notebooks/                notebook del laboratorio (investigación y resultados)
entregables/videos/       videos .mp4 generados
entregables/metricas.json métricas por episodio de cada video
codebook.md               descripción de observaciones, acciones y recompensa
```

## Entorno de ejecución

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Las ROMs de Atari vienen incluidas en `ale-py` desde la versión 0.10, por lo que
no hay que descargarlas aparte.

## Ejecución

```bash
python src/run_pipeline.py          # genera todos los videos entregables
python src/01_generar_videos.py     # solo la etapa de grabación
```

Para el notebook:

```bash
jupyter notebook notebooks/laboratorio5_ale_space_invaders.ipynb
```

## Contenido del notebook

1. Configuración e importaciones, semillas fijas.
2. El Arcade Learning Environment: qué es, Stella y las variantes de un mismo juego.
3. Espacios de observación y acción de `ALE/SpaceInvaders-v5` frente a `CartPole-v1`.
4. Observación en RAM y wrappers `AtariPreprocessing` / `FrameStackObservation`.
5. Módulo de funciones: `crear_entorno`, `agente_aleatorio`, `agente_regla_simple`,
   `ejecutar_episodio`, `generar_video_agente`.
6. Generación de los videos y reporte de pasos sobrevividos y recompensa total.
7. Comparación agente aleatorio vs. agente de regla simple y discusión.
