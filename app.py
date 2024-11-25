import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
import streamlit as st

# Configuración para TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        st.write("Memoria del GPU configurada para crecimiento gradual.")
    except Exception as e:
        st.error(f"Error al configurar el GPU: {e}")
else:
    st.write("No se encontró un dispositivo GPU. El entrenamiento se realizará en la CPU.")

# Streamlit UI
st.title("Predicción de Precios de Solana con LSTM")
st.write("Este modelo utiliza un LSTM para predecir precios futuros de Solana basados en datos históricos o introducidos por el usuario.")

# Cargar modelo entrenado
# Registrar la función 'mse' explícitamente
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Cargar modelo entrenado
try:
    model = load_model("lstm_model_solana.h5", custom_objects={"mse": mse})
    st.write("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Normalizador (asegúrate de que este sea el mismo que usaste durante el entrenamiento)
features = ['close', 'RSI', 'MACD', 'MACD_signal', 'ATR']
scaler = MinMaxScaler()

# Ajustar el escalador con datos de ejemplo (deberías usar los datos reales del entrenamiento)
dummy_data = np.array([
    [10, 0, -1, -1, 0.5],  # Valores mínimos simulados
    [250, 100, 1, 1, 10]   # Valores máximos simulados
])
scaler.fit(dummy_data)  # Ajustar el escalador con los rangos de valores originales

# Función para denormalizar
def denormalize(scaled_data, scaler, feature_index):
    """
    Denormalizar datos escalados utilizando un MinMaxScaler.
    """
    dummy = np.zeros((scaled_data.shape[0], len(scaler.scale_)))
    dummy[:, feature_index] = scaled_data
    denormalized_data = scaler.inverse_transform(dummy)
    return denormalized_data[:, feature_index]

# Entrada del usuario
st.header("Entrada Manual de Indicadores")
price = st.number_input("Precio actual de Solana (USD)", min_value=0.0, value=200.0, step=0.01)
rsi = st.number_input("RSI", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
macd = st.number_input("MACD", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
macd_signal = st.number_input("MACD Signal", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
atr = st.number_input("ATR", min_value=0.0, value=1.0, step=0.1)

# Botón para predecir
if st.button("Predecir"):
    try:
        # Normalizar los valores introducidos
        user_input = np.array([[price, rsi, macd, macd_signal, atr]])
        normalized_input = scaler.transform(user_input)

        # Crear la secuencia de entrada
        n_steps_in = 720  # 15 días de datos (30 minutos cada paso)
        sequence = np.tile(normalized_input, (n_steps_in, 1))  # Replicar los valores para completar la secuencia
        sequence = sequence.reshape(1, n_steps_in, len(features))  # Darle forma para que sea compatible con LSTM

        # Hacer la predicción
        prediction = model.predict(sequence)

        # Denormalizar la predicción
        future_prices = denormalize(prediction[0], scaler, features.index('close'))

        # Mostrar los resultados
        st.subheader("Predicciones para las próximas horas")
        for i, price in enumerate(future_prices, start=1):
            st.write(f"Hora {i}: {price:.2f} USD")
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")


