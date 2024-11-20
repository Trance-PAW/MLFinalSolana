# MLFinalSolana
Proyecto Final MIC - Machine Learning
# Maestria en Ingenieria en Computación de la Facultad de Ingenieria de la UACH

## Proyecto de Machine Learning

- Clase: Machine Learning  
- Alumno: Jesús Manuel Muñoz Larguero  
- Matrícula: 207054


## Predicción del Precio de Solana con SVM y Streamlit

Este proyecto utiliza un modelo de **Support Vector Machine (SVM)** para predecir si el precio de **Solana** subirá o bajará en las próximas 5 horas. El modelo se entrena utilizando indicadores técnicos como **EMA, RSI, MACD** y **SAR**. Además, se utiliza una aplicación de **Streamlit** para proporcionar una interfaz simple donde los usuarios pueden ingresar valores y obtener predicciones del modelo.

## Contenido del Proyecto

- **`solana_historical_data.csv`**: Conjunto de datos histórico utilizado para entrenar el modelo.
- **`train_model.py`**: Script para entrenar el modelo SVM y guardar el pipeline en un archivo `.sav`.
- **`app.py`**: Aplicación de **Streamlit** para cargar el modelo guardado y hacer predicciones en tiempo real.
- **`best_pipeline_svm.sav`**: Archivo que contiene el pipeline del modelo SVM entrenado con los mejores hiperparámetros.
- **`README.md`**: Este archivo de documentación.

## Características del Proyecto

1. **Entrenamiento del Modelo SVM**: 
   - Se entrena un modelo SVM utilizando GridSearchCV para encontrar los mejores hiperparámetros.
   - Los indicadores técnicos calculados incluyen: **EMA (15, 20, 50)**, **RSI**, **MACD**, y **SAR**.
   - El pipeline completo del modelo se guarda en el archivo `best_pipeline_svm.sav`.

2. **Aplicación en Streamlit**:
   - La aplicación permite a los usuarios ingresar los valores de los indicadores técnicos y obtener predicciones del modelo entrenado.
   - Se carga el modelo guardado desde el archivo `best_pipeline_svm.sav` para hacer las predicciones.

## Instalación

Sigue los siguientes pasos para instalar y ejecutar el proyecto localmente:

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu_usuario/nombre_repositorio.git
cd nombre_repositorio
```

### 2. Crear un entorno virtual (opcional, pero recomendado)

```bash
python -m venv env
source env/bin/activate  # En Windows usa: env\Scripts\activate
```
### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```
Claro, te ayudaré a crear un archivo README.md en formato para GitHub. Este archivo proporcionará información clave sobre tu proyecto, cómo configurarlo, y cómo usarlo.

Aquí tienes un ejemplo básico del README.md que puedes ajustar según lo necesites:

markdown

# Predicción del Precio de Solana con SVM y Streamlit

Este proyecto utiliza un modelo de **Support Vector Machine (SVM)** para predecir si el precio de **Solana** subirá o bajará en las próximas 5 horas. El modelo se entrena utilizando indicadores técnicos como **EMA, RSI, MACD** y **SAR**. Además, se utiliza una aplicación de **Streamlit** para proporcionar una interfaz simple donde los usuarios pueden ingresar valores y obtener predicciones del modelo.

## Contenido del Proyecto

- **`solana_historical_data.csv`**: Conjunto de datos histórico utilizado para entrenar el modelo.
- **`train_model.py`**: Script para entrenar el modelo SVM y guardar el pipeline en un archivo `.sav`.
- **`app.py`**: Aplicación de **Streamlit** para cargar el modelo guardado y hacer predicciones en tiempo real.
- **`best_pipeline_svm.sav`**: Archivo que contiene el pipeline del modelo SVM entrenado con los mejores hiperparámetros.
- **`README.md`**: Este archivo de documentación.

## Características del Proyecto

1. **Entrenamiento del Modelo SVM**: 
   - Se entrena un modelo SVM utilizando GridSearchCV para encontrar los mejores hiperparámetros.
   - Los indicadores técnicos calculados incluyen: **EMA (15, 20, 50)**, **RSI**, **MACD**, y **SAR**.
   - El pipeline completo del modelo se guarda en el archivo `best_pipeline_svm.sav`.

2. **Aplicación en Streamlit**:
   - La aplicación permite a los usuarios ingresar los valores de los indicadores técnicos y obtener predicciones del modelo entrenado.
   - Se carga el modelo guardado desde el archivo `best_pipeline_svm.sav` para hacer las predicciones.

## Instalación

Sigue los siguientes pasos para instalar y ejecutar el proyecto localmente:

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu_usuario/nombre_repositorio.git
cd nombre_repositorio
```
2. Crear un entorno virtual (opcional, pero recomendado)

```bash

python -m venv env
source env/bin/activate  # En Windows usa: env\Scripts\activate
```
3. Instalar las dependencias

```bash
pip install -r requirements.txt
```
Asegúrate de tener instaladas las siguientes dependencias:

    streamlit
    pandas
    joblib
    sklearn
    talib
    matplotlib
    seaborn

4. Ejecutar la Aplicación en Streamlit

```bash
streamlit run app.py
```
Estructura del Proyecto
```bash
├── app.py                    # Código de la aplicación Streamlit
├── solana_historical_data.csv # Conjunto de datos utilizado
├── train_model.py             # Código para entrenar el modelo SVM
├── best_pipeline_svm.sav      # Pipeline guardado del modelo SVM
├── README.md                  # Documentación del proyecto
└── requirements.txt           # Dependencias del proyecto
```
Cómo entrenar el modelo

Si quieres entrenar el modelo desde cero:

    Asegúrate de que el archivo solana_historical_data.csv esté disponible.
    Ejecuta el siguiente comando para entrenar el modelo SVM y guardar el pipeline:
```bash
python train_model.py
```
El modelo entrenado se guardará como best_pipeline_svm.sav.

# Enlace de streamlit  
```bash
https://solanamlmic.streamlit.app/
```
