# Laboratorio 03: clasificación de género y estimación de edad

Proyecto del Taller 3 de Deep Learning sobre UTKFace. Implementa un pipeline
multitarea reproducible para:

- clasificar la etiqueta binaria de género de UTKFace;
- estimar la edad mediante regresión;
- comparar modelos clásicos, MLP, CNN y transferencia de aprendizaje;
- ejecutar ablaciones bajo un protocolo común;
- desplegar el modelo seleccionado mediante Streamlit.

Las etiquetas binarias de UTKFace representan anotaciones del dataset, no la
identidad de género de una persona. El sistema es educativo, puede reproducir
sesgos demográficos y no debe utilizarse para tomar decisiones sobre personas.

## Estado del proyecto

- 5 estrategias implementadas.
- 20 experimentos completados: 5 configuraciones base y 15 ablaciones.
- 23.705 imágenes válidas.
- Split fijo: 16.593 entrenamiento, 3.555 validación y 3.557 prueba.
- Semilla: 42.
- 11 pruebas automatizadas aprobadas.
- Aplicación Streamlit funcional y preparada para Community Cloud.
- Informe IEEE y figuras disponibles en `informe/`.

Resultados principales:

| Criterio | Experimento | Resultado |
|---|---|---:|
| Mejor accuracy de género | `resnet_finetuning_unfreeze_more` | 0,9255 |
| Mejor F1 de género | `resnet_finetuning_unfreeze_more` | 0,9255 |
| Mejor MAE de edad | `resnet_finetuning_lambda_high` | 5,8480 años |
| Mejor RMSE de edad | `resnet_finetuning_lambda_high` | 8,2030 años |
| Mejor R² de edad | `resnet_finetuning_lambda_high` | 0,8232 |

La aplicación despliega `resnet_finetuning_lambda_high`, seleccionado por su
equilibrio entre ambas tareas. Su accuracy de género es 0,9165 y su mejor
checkpoint corresponde a la época 4.

## Estructura

```text
.
├── .env.example
├── environment.yml
├── requirements.txt
├── main.py
├── Frontend/
│   ├── streamlit_main.py
│   ├── streamlit_app.py
│   ├── ui_styles.py
│   ├── requirements.txt
│   ├── models/
│   │   └── best_model.pt
│   └── assets/
│       ├── fondo.png
│       ├── github-logo.png
│       └── results/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data/
│   ├── baselines/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── scripts/
│   └── generate_visual_examples.py
├── tests/
├── artifacts/
│   ├── checkpoints/
│   ├── experiments/
│   ├── plots/
│   ├── reports/
│   └── splits/
├── informe/
│   ├── boceto_informe3.ltx
│   ├── referencias.bib
│   └── images/
└── docs/
    └── Laboratorio_03__Redes_convolucionales_final.pdf
```

Los archivos `mlp_todo.py`, `resnet_todo.py` y `classical_todo.py` conservan
el nombre de la plantilla educativa, pero contienen implementaciones
funcionales.

## Estrategias experimentales

| ID | Estrategia | Configuraciones |
|---|---|---:|
| E1 | PCA + GaussianNB/Ridge | 3 |
| E2 | MLP multitarea | 4 |
| E3 | CNN simple multitarea | 5 |
| E4 | ResNet18 congelada | 4 |
| E5 | ResNet18 con fine-tuning | 4 |

Catálogo completo:

```bash
python main.py --list
```

Las ablaciones estudian componentes PCA, dropout, aumentación, ponderación de
edad, learning rate y cantidad de bloques residuales descongelados.

## Dataset y preprocesamiento

Se utiliza UTKFace `Aligned & Cropped Faces`. Los nombres siguen el formato:

```text
edad_genero_raza_fecha.jpg
```

El parser extrae edad y género, valida los nombres e ignora entradas
inválidas. El código interpreta:

```text
edad   -> valor continuo
género -> 0 o 1, según la anotación de UTKFace
```

Preprocesamiento neuronal:

- redimensionamiento a 224x224;
- conversión a tensor RGB;
- normalización con estadísticas de ImageNet;
- volteo horizontal y `ColorJitter` leve solo en entrenamiento.

Validación, prueba e inferencia usan transformaciones deterministas. El
baseline clásico convierte las imágenes a escala de grises y 64x64 antes de
PCA.

## Pérdida y métricas

Los modelos neuronales comparten representación y utilizan dos cabezas:

```text
CrossEntropyLoss(género)
    + lambda_age * SmoothL1Loss(edad)
```

Métricas de género:

- accuracy;
- precisión, recall y F1 ponderados;
- matriz de confusión.

Métricas de edad:

- MAE;
- RMSE;
- R²;
- MAE y soporte por rango etario;
- edad real frente a predicha;
- distribución de residuos.

También se registran tiempo de entrenamiento, parámetros entrenables,
historial por época, predicciones individuales y checkpoint seleccionado.

## Instalación

### Conda

```bash
conda env create -f environment.yml
conda activate lab03-dl-2026-01
cp .env.example .env
```

Para actualizar un entorno existente:

```bash
conda env update -f environment.yml --prune
```

### Entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows, la activación depende de la terminal utilizada.

## Configuración

Configura `.env` antes de entrenar:

```dotenv
UTKFACE_DIR=/ruta/a/UTKFace
ARTIFACTS_DIR=artifacts
CNN_CHECKPOINT=Frontend/models/best_model.pt

SEED=42
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=10
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
LAMBDA_AGE=0.01
NUM_WORKERS=0
CPU_THREADS=4
MAX_IMAGES=0
DEVICE=auto
```

`CPU_THREADS` limita PyTorch, OpenMP, MKL, OpenBLAS y NumExpr para evitar
saturación en servidores compartidos. `MAX_IMAGES` permite ejecutar pruebas
rápidas con un subconjunto; el valor 0 usa todas las imágenes.

## Ejecución experimental

Ejecutar la CNN base:

```bash
python main.py --experiment cnn_base
```

Seleccionar varios experimentos:

```bash
python main.py \
  --experiment cnn_base \
  --experiment cnn_no_dropout
```

Ejecutar las 20 configuraciones implementadas:

```bash
python main.py --ablations
```

`--all` también ejecuta todos los experimentos implementados:

```bash
python main.py --all
```

El entrenamiento muestra progreso por época y batch. Cada modelo neuronal
guarda el checkpoint con menor pérdida total de validación.

## Artefactos

Los archivos generados se guardan bajo `ARTIFACTS_DIR`:

```text
artifacts/
├── checkpoints/<experimento>/
├── experiments/<experimento>/
│   ├── result.json
│   ├── evaluation.json
│   ├── predictions.csv
│   ├── training_history.csv
│   └── training_history.json
├── plots/
├── reports/
└── splits/utkface_split.json
```

Los experimentos clásicos no tienen historial por épocas. Los reportes
principales incluyen:

```text
all_experiments_comparison.csv
all_experiments_comparison.md
e1_classical_ablations.*
e2_mlp_ablations.*
e3_cnn_ablations.*
e4_resnet_frozen_ablations.*
e5_resnet_finetuning_ablations.*
environment.json
run_metadata.json
run_progress.json
```

`run_progress.json` se actualiza después de cada experimento para conservar los
resultados ya completados ante una interrupción.

## Ejemplos cualitativos

La figura de aciertos y errores se genera sin reentrenar:

```bash
python scripts/generate_visual_examples.py \
  --predictions artifacts/experiments/resnet_finetuning_lambda_high/predictions.csv \
  --dataset /ruta/a/UTKFace \
  --output informe/images/resnet_final_visual_examples.png \
  --seed 42
```

El script produce una grilla 2x3 a 300 DPI con:

- un acierto por género con error de edad bajo;
- ambos sentidos de error de género;
- dos errores grandes de edad, priorizando personas de 60 años o más.

## Aplicación Streamlit

La aplicación local utiliza el checkpoint final incluido en el repositorio.
No necesita UTKFace ni volver a entrenar:

```bash
streamlit run Frontend/streamlit_main.py
```

También es válido:

```bash
streamlit run Frontend/streamlit_app.py
```

Funciones principales:

- carga de archivo y captura por cámara;
- detección y recorte del rostro;
- preprocesamiento equivalente al conjunto de prueba;
- predicción de género, confianza estimada y edad;
- comparación resumida de los experimentos;
- créditos y enlace al repositorio.

### Streamlit Community Cloud

Configuración recomendada:

```text
Repository: Menderin/Taller-3-Deep-Learning
Branch: main
Main file path: Frontend/streamlit_main.py
Python: 3.10
Secrets: ninguno
```

`Frontend/requirements.txt` contiene únicamente las dependencias necesarias
para la aplicación. El checkpoint final y los gráficos del frontend están
versionados, por lo que el despliegue no depende de `.env` ni de artefactos
externos.

## Informe

El informe se encuentra en:

```text
informe/boceto_informe3.ltx
informe/referencias.bib
informe/images/
```

Para Overleaf, sube el contenido de `informe/` como raíz del proyecto y
selecciona `boceto_informe3.ltx` como documento principal.

## Pruebas

```bash
python -m pytest
```

La suite contiene 11 pruebas para:

- parser de UTKFace;
- formas de salida de la CNN;
- catálogo experimental;
- métricas multitarea;
- reportes y persistencia de artefactos;
- selección de ejemplos cualitativos.

## Reproducibilidad

La ejecución completa registrada utilizó:

```text
Python: 3.10.20
PyTorch: 2.12.1+cu130
torchvision: 0.27.1+cu130
CUDA: 13.0
GPU: NVIDIA RTX A5000
Seed: 42
```

El repositorio no incluye el dataset, `.env`, agentes internos ni los
artefactos completos descargados desde el servidor.
