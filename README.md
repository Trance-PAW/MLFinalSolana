# MLFinalSolana
Proyecto Final MIC - Machine Learning
# Maestria en Ingenieria en Computación de la Facultad de Ingenieria de la UACH

## Proyecto Final de Machine Learning

- Clase: Machine Learning  
- Alumno: Jesús Manuel Muñoz Larguero  
- Matrícula: 207054


## Predicción del Precio de Solana con LSTM y Streamlit

Este proyecto utiliza un modelo de **Long Short-Term Memory (LSTM)** para predecir si el precio de **Solana** subirá o bajará en las próximas 5 horas. El modelo se entrena utilizando indicadores técnicos como **RSI**, **MACD** y **ATR**. Además, se utiliza una aplicación de **Streamlit** para proporcionar una interfaz simple donde los usuarios pueden ingresar valores y obtener predicciones del modelo.

## Contenido del Proyecto

- **`solana_historical_data.csv`**: Conjunto de datos histórico utilizado para entrenar el modelo.
- **`app.py`**: Aplicación de **Streamlit** para cargar el modelo guardado y hacer predicciones en tiempo real.
- **`README.md`**: Este archivo de documentación.

## Características del Proyecto

1. **Entrenamiento del Modelo LSTM**: 
   - Se entrena un modelo LSTM utilizando Keras Tuner con Randomized Search para encontrar los mejores hiperparámetros.
   - Los indicadores técnicos calculados incluyen: **RSI**, **MACD**, y **ATR**.
 

2. **Aplicación en Streamlit**:
   - La aplicación permite a los usuarios ingresar los valores de los indicadores técnicos y obtener predicciones del modelo entrenado.
  
## Instalación

Sigue los siguientes pasos para instalar y ejecutar el proyecto localmente:

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu_usuario/nombre_repositorio.git(https://github.com/Trance-PAW/MLFinalSolana)
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
- **`README.md`**: Este archivo de documentación.


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

# Enlace de streamlit  
```bash
https://solanamlmic.streamlit.app
```
