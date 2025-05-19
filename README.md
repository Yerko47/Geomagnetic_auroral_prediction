# Proyecto de Predicción de Índices Aurorales Electrojet

## Descripción General

Este proyecto tiene como objetivo predecir los índices aurorales electrojet (como AE, AL, AU) utilizando datos de viento solar y campos magnéticos interplanetarios (OMNI). Implementa un flujo de trabajo completo de machine learning, desde la ingesta y preprocesamiento de datos hasta el entrenamiento, validación cruzada, selección y evaluación de diversos modelos de redes neuronales profundas.

El sistema está diseñado para ser configurable a través de un archivo `config.yaml` y argumentos de línea de comandos, permitiendo flexibilidad en la experimentación con diferentes arquitecturas de modelos, preprocesamiento de datos y estrategias de entrenamiento.

## Características Principales

* **Ingesta de Datos Configurable**: Carga y procesa datos OMNI en formato CDF de un rango de años especificado.
* **Preprocesamiento Avanzado**: Incluye limpieza de datos, manejo de valores atípicos/faltantes, y escalado de características.
* **Selección de Eventos de Tormenta**: Capacidad para enfocar el análisis en periodos alrededor de tormentas geomagnéticas.
* **Ingeniería de Características**: Creación de secuencias temporales con retardo (`delay_steps`) para alimentar modelos secuenciales.
* **Múltiples Arquitecturas de Modelos**:
    * Red Neuronal Artificial (ANN/MLP)
    * Red Neuronal Convolucional 1D (CNN)
    * Red de Memoria a Corto-Largo Plazo (LSTM) bidireccional
    * Unidad Recurrente Cerrada (GRU) bidireccional
    * Red Convolucional Temporal (TCNN)
    * Transformer (basado en encoder)
* **Entrenamiento y Evaluación Robustos**:
    * División de datos en conjuntos de desarrollo y prueba final (hold-out).
    * Validación Cruzada Temporal (`TimeSeriesSplit`) para la selección de hiperparámetros (ej. `delay_length`).
    * Métricas de regresión detalladas (RMSE, R-Score, D² Absolute Error, D² Tweedie Score).
    * Mecanismo de Parada Temprana (`EarlyStopping`).
    * Optimización con schedulers de tasa de aprendizaje.
* **Configuración Flexible**:
    * Parámetros gestionados centralmente a través de `config.yaml`.
    * Posibilidad de sobrescribir parámetros de configuración mediante argumentos de línea de comandos (`cli.py`).
* **Visualización de Resultados**: Generación de plots para:
    * Análisis exploratorio de datos (series temporales, correlaciones).
    * Curvas de entrenamiento y validación (pérdida, métricas).
    * Comparación del rendimiento de modelos/configuraciones.
    * Comparación de valores reales vs. predichos para eventos de tormenta.
    * GIFs animados de la evolución temporal de las predicciones.
* **Estructura de Proyecto Organizada**: Rutas de archivos y directorios gestionadas por `paths.py`.

## Estructura del Proyecto (Simplificada)
```text
auroral_prediction_project/
├── config/
│   └── config.yaml             # Archivo de configuración principal
├── data/
│   ├── raw/                    # Datos crudos OMNI (CDF) y feather procesado
│   └── processed/              # Datos procesados (ej. lista de tormentas)
├── models/
│   ├── results_csv/            # CSVs con métricas y resultados de predicciones
│   └── *.pt                    # Modelos PyTorch entrenados guardados
├── plots/                      # Directorio raíz para todos los gráficos generados
│   ├── historic/
│   ├── stadistics/
│   ├── training/
│   └── testing/
├── src/                        # Código fuente del proyecto (o directamente en raíz)
│   ├── main.py                 # Script principal para ejecutar el flujo de trabajo
│   ├── data_processing.py      # Módulos de procesamiento de datos
│   ├── models.py               # Definiciones de arquitecturas de modelos
│   ├── model_training.py       # Lógica de entrenamiento y evaluación
│   ├── config_loader.py        # Carga de configuración
│   ├── paths.py                # Gestión de rutas y estructura
│   ├── plot.py                 # Funciones de ploteo
│   ├── cli.py                  # Interfaz de línea de comandos (opcional)
│   └── ...                     # Otros módulos
├── notebooks/                  # Jupyter notebooks para experimentación (opcional)
├── tests/                      # Pruebas unitarias (opcional)
└── README.md                   # Este archivo

