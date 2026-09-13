"""Etapa 1: graba los videos entregables y escribe las métricas de cada episodio."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ale_utils import (
    agente_aleatorio,
    agente_regla_simple,
    fijar_semillas,
    generar_video_agente,
)
from config import (
    ARCHIVO_METRICAS,
    DIR_VIDEOS,
    RAIZ,
    ENTORNO_SPACE_INVADERS,
    MAX_STEPS,
    SEMILLA,
)

# Videos a generar: el agente aleatorio es el entregable obligatorio y el de
# regla simple sirve de comparación.
CORRIDAS = [
    {
        "name_prefix": "aleatorio-space-invaders",
        "funcion_agente": agente_aleatorio,
        "n_episodios": 1,
    },
    {
        "name_prefix": "regla-simple-space-invaders",
        "funcion_agente": agente_regla_simple,
        "n_episodios": 1,
    },
]


def main() -> dict:
    fijar_semillas(SEMILLA)
    DIR_VIDEOS.mkdir(parents=True, exist_ok=True)

    resultados = {}
    for corrida in CORRIDAS:
        resultado = generar_video_agente(
            ENTORNO_SPACE_INVADERS,
            corrida["funcion_agente"],
            video_folder=str(DIR_VIDEOS),
            name_prefix=corrida["name_prefix"],
            n_episodios=corrida["n_episodios"],
            max_steps=MAX_STEPS,
            seed=SEMILLA,
        )
        # Se guardan rutas relativas a la raíz para que el JSON sea portable.
        resultado["videos"] = [str(Path(v).relative_to(RAIZ)) for v in resultado["videos"]]
        resultados[corrida["name_prefix"]] = resultado

        for episodio in resultado["episodios"]:
            print(
                f"{corrida['name_prefix']} | episodio {episodio['episodio']}: "
                f"{episodio['pasos']} pasos, recompensa total {episodio['recompensa_total']:.0f}"
            )

    ARCHIVO_METRICAS.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"Métricas escritas en {ARCHIVO_METRICAS}")
    return resultados


if __name__ == "__main__":
    main()
