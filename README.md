# Laboratorio 03: clasificación de género y regresión de edad

Este proyecto entrega una estructura educativa y modular para trabajar con
UTKFace usando PyTorch. La implementación funcional corresponde a una CNN
simple multitarea con dos salidas:

- Clasificación binaria de género según las etiquetas de UTKFace.
- Regresión de edad.

La CNN incluye ejemplos de ablaciones y el proyecto contiene implementaciones
base para las cinco estrategias del laboratorio, listas para validarse,
ejecutarse y compararse con el mismo protocolo.

> Las etiquetas binarias de UTKFace describen la anotación del dataset y no la
> identidad de género de una persona. El modelo puede reproducir sesgos y no
> debe usarse para tomar decisiones sobre personas.

## Estructura del proyecto

```text
.
├── .env.example
├── environment.yml
├── README.md
├── requirements.txt
├── main.py
├── Frontend/
│   └── streamlit_main.py
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data/
│   │   ├── parser.py
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── base.py
│   │   ├── cnn.py
│   │   ├── mlp_todo.py
│   │   └── resnet_todo.py
│   ├── baselines/
│   │   └── classical_todo.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── experiment_runner.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── reporter.py
│   └── inference/
│       ├── face_detector.py
│       ├── predictor.py
│       └── streamlit_app.py
├── tests/
└── artifacts/
    ├── checkpoints/
    ├── reports/
    ├── plots/
    └── splits/
```

## Componentes principales

| Componente | Responsabilidad |
|---|---|
| `main.py` | Orquestar entrenamiento, evaluación y reportes. |
| `Frontend/streamlit_main.py` | Orquestar la aplicación de inferencia. |
| `src/config.py` | Leer `.env` y centralizar hiperparámetros. |
| `src/data/` | Leer UTKFace, extraer etiquetas y crear particiones reproducibles. |
| `src/models/cnn.py` | Implementar la CNN multitarea entregada. |
| `src/training/` | Ejecutar el ciclo manual de entrenamiento con PyTorch. |
| `src/evaluation/` | Calcular métricas, generar gráficos y exportar reportes. |
| `src/inference/` | Detectar rostros y usar el checkpoint de la CNN. |

No se usa PyTorch Lightning. El ciclo de entrenamiento muestra directamente
`zero_grad`, `backward` y `step` para que el flujo de PyTorch sea observable.
La única excepción es E1, porque PCA y los estimadores clásicos se implementan
con scikit-learn.

## Preparación del ambiente

Se recomienda usar Conda para crear el ambiente completo desde un solo archivo:

```bash
conda env create -f environment.yml
conda activate lab03-dl-2026-01
cp .env.example .env
```

Para actualizar un ambiente existente después de modificar `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

También es posible usar un ambiente virtual estándar con Python 3.10 o
superior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Descarga la versión **Aligned & Cropped Faces** de UTKFace y configura la ruta
en `.env`:

```dotenv
UTKFACE_DIR=/ruta/a/UTKFace
```

La carpeta debe contener archivos con nombres como:

```text
25_1_2_20170116174525125.jpg
```

El parser interpreta el primer valor como edad y el segundo como género.

## Flujo de trabajo

1. `AppConfig` carga la ruta del dataset y los hiperparámetros desde `.env`.
2. `UTKFaceDataModule` descubre imágenes válidas y crea una partición
   reproducible de entrenamiento, validación y prueba.
3. Las transformaciones aleatorias se aplican solamente en entrenamiento.
4. `ExperimentRunner` construye la CNN, la pérdida multitarea y el optimizador.
5. `MultiTaskTrainer` entrena con PyTorch y guarda el mejor checkpoint según la
   pérdida de validación.
6. `MultiTaskEvaluator` calcula métricas separadas para género y edad.
7. `MetricsReporter` registra todos los experimentos, incluidos los pendientes.

La pérdida total es:

```text
CrossEntropyLoss(género) + lambda_age * SmoothL1Loss(edad)
```

## Ejecución

Mostrar el catálogo completo:

```bash
python3 main.py --list
```

Entrenar la CNN base:

```bash
python3 main.py --experiment cnn_base
```

Ejecutar las configuraciones base y las ablaciones implementadas de todas las
estrategias:

```bash
python3 main.py --ablations
```

Ejecutar todos los experimentos actualmente implementados:

```bash
python3 main.py --all
```

También se pueden seleccionar varios experimentos:

```bash
python3 main.py \
  --experiment cnn_base \
  --experiment cnn_no_dropout
```

## Estrategias y ablaciones

| ID | Estrategia | Estado inicial |
|---|---|---|
| E1 | PCA + modelos clásicos | `IMPLEMENTADO` |
| E2 | MLP multitarea | `IMPLEMENTADO` |
| E3 | CNN simple multitarea | `IMPLEMENTADO` |
| E4 | ResNet18 congelada | `IMPLEMENTADO` |
| E5 | ResNet18 fine-tuning | `IMPLEMENTADO` |

En esta estructura, E6 no es un modelo independiente. E6 representa el
análisis transversal de ablaciones que debe aplicarse a E1, E2, E3, E4 y E5.

La CNN incluye estas variantes funcionales:

| Experimento | Componente modificado |
|---|---|
| `cnn_base` | Ninguno |
| `cnn_no_augmentation` | Elimina aumentación de datos |
| `cnn_no_dropout` | Usa `dropout=0.0` |
| `cnn_lambda_low` | Reduce `lambda_age` |
| `cnn_lambda_high` | Aumenta `lambda_age` |

### Regla obligatoria para los alumnos

**Las ablaciones deben probarse en todos los experimentos implementados, no
solo en la CNN entregada.**

Para cada estrategia E1 a E5 se debe:

1. Ejecutar una configuración base.
2. Seleccionar al menos dos componentes relevantes para analizar.
3. Modificar un solo componente por ablación.
4. Mantener la misma semilla, partición y protocolo de evaluación.
5. Reportar resultados favorables y desfavorables.
6. Discutir el efecto sobre género, edad, costo computacional y sobreajuste.

El catálogo deja ejemplos de ablaciones pendientes para E1, E2, E4 y E5.

## Reportes y métricas

Cada ejecución genera una fila para todos los modelos y algoritmos del
catálogo. Los estados posibles son:

- `COMPLETADO`: se ejecutó y tiene métricas.
- `NO_IMPLEMENTADO`: todavía debe ser completado por los alumnos.
- `NO_EJECUTADO`: existe, pero no fue seleccionado en esta ejecución.
- `ERROR`: se intentó ejecutar, pero ocurrió un problema.

Los reportes principales se guardan en:

```text
artifacts/reports/
├── all_experiments_comparison.csv
├── all_experiments_comparison.md
├── e1_classical_ablations.csv
├── e2_mlp_ablations.csv
├── e3_cnn_ablations.csv
├── e4_resnet_frozen_ablations.csv
├── e5_resnet_finetuning_ablations.csv
└── environment.json
```

Las métricas de género son accuracy, precision, recall, F1 y matriz de
confusión. Las métricas de edad son MAE, RMSE, R² y MAE por rango etario.
También se registran parámetros entrenables, tiempo de entrenamiento, curvas
de pérdida y gráfico de edad real versus predicha.

## Aplicación Streamlit

Primero entrena `cnn_base` o configura `CNN_CHECKPOINT` en `.env` para apuntar
a otro checkpoint de la CNN:

```bash
python3 main.py --experiment cnn_base
streamlit run Frontend/streamlit_main.py
```

La aplicación permite subir una imagen o capturarla con la cámara, detecta la
cara más grande mediante OpenCV, aplica el preprocesamiento de evaluación y
muestra género, confianza y edad estimada.

## Trabajo pendiente para los alumnos

- Validar el ambiente y la ruta real de UTKFace en `.env`.
- Ejecutar y analizar ablaciones para **cada** estrategia.
- Comparar métricas, curvas, errores visuales y costo computacional.
- Discutir sesgos, limitaciones del dataset y uso responsable del modelo.
- Redactar el informe final con tablas, gráficos y conclusiones.

## Pruebas

```bash
python3 -m pytest
```

Las pruebas verifican el parser, las formas de salida de la CNN, las métricas y
la generación de reportes. No reemplazan la evaluación experimental sobre
UTKFace.
