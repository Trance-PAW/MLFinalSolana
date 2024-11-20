import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import seaborn as sns
import streamlit as st
import gdown
import os

# Configuración para TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memoria del GPU configurada para crecimiento gradual.")
    except Exception as e:
        print(f"Ocurrió un error al configurar el GPU: {e}")
else:
    print("No se encontró un dispositivo GPU. El entrenamiento se realizará en la CPU.")

# Streamlit UI
st.title("Predicción de Precios de Solana con LSTM")
st.write("Este modelo utiliza un LSTM para predecir precios futuros de Solana basados en datos históricos.")

def download_dataset():
    file_id = "TU_FILE_ID"  # Reemplaza con el ID real del archivo
    output_path = "solana_historical_data.csv"
    gdown.download(f"https://drive.google.com/file/d/1x8kLHMPbYKFskn4n9gTJE_Wp0Ei-87cE/view?usp=sharing", output_path, quiet=False)
    return output_path



# Cargar el CSV con tolerancia a errores
try:
    solana_data = pd.read_csv(data_path, on_bad_lines='skip')
    print("Archivo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")

# Revisar si todas las filas tienen la misma cantidad de columnas
with open(data_path, 'r') as file:
    lines = file.readlines()

# Detectar problemas de longitud
column_counts = [len(line.split(',')) for line in lines]
if len(set(column_counts)) > 1:
    print("Se encontraron líneas con un número inconsistente de columnas:")
    for i, count in enumerate(column_counts):
        if count != column_counts[0]:
            print(f"Línea {i + 1} tiene {count} columnas.")
            
            
# Verificar estructura del dataset
st.write("Primeras filas del dataset:")
st.write(solana_data.head())
st.write("Resumen del dataset:")
st.write(solana_data.info())

# Procesar datos
try:
    solana_data['timestamp'] = pd.to_datetime(solana_data['timestamp'])
    solana_data.sort_values('timestamp', inplace=True)
    filtered_data = solana_data[solana_data['timestamp'] >= solana_data['timestamp'].max() - pd.DateOffset(months=2)]
    filtered_data.fillna(method='ffill', inplace=True)
    filtered_data.fillna(method='bfill', inplace=True)
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    st.stop()

# Cálculo de indicadores técnicos
try:
    rsi_window = 14
    def compute_rsi(data, window):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    filtered_data['RSI'] = compute_rsi(filtered_data['close'], window=rsi_window)
    ema12 = filtered_data['close'].ewm(span=12, adjust=False).mean()
    ema26 = filtered_data['close'].ewm(span=26, adjust=False).mean()
    filtered_data['MACD'] = ema12 - ema26
    filtered_data['MACD_signal'] = filtered_data['MACD'].ewm(span=9, adjust=False).mean()
    filtered_data['TR'] = np.maximum(filtered_data['high'] - filtered_data['low'], 
                                     np.maximum(abs(filtered_data['high'] - filtered_data['close'].shift(1)),
                                                abs(filtered_data['low'] - filtered_data['close'].shift(1))))
    filtered_data['ATR'] = filtered_data['TR'].rolling(window=rsi_window).mean()
    filtered_data.dropna(inplace=True)
except Exception as e:
    st.error(f"Error al calcular los indicadores técnicos: {e}")
    st.stop()

# Normalización
try:
    features = ['close', 'RSI', 'MACD', 'MACD_signal', 'ATR']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(filtered_data[features])
except Exception as e:
    st.error(f"Error al normalizar los datos: {e}")
    st.stop()

# Crear secuencias para el LSTM
try:
    def create_sequences(data, n_steps_in, n_steps_out):
        X, y = [], []
        for i in range(len(data) - n_steps_in - n_steps_out + 1):
            X.append(data[i:i + n_steps_in])
            y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])  # Columna 'close'
        return np.array(X), np.array(y)

    n_steps_in = 720  # 15 días de datos (30 minutos cada paso)
    n_steps_out = 6  # 3 horas
    X, y = create_sequences(scaled_features, n_steps_in, n_steps_out)

    split_ratio = 0.8
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
except Exception as e:
    st.error(f"Error al crear las secuencias: {e}")
    st.stop()

# Construcción del modelo LSTM
try:
    model = Sequential([
        LSTM(200, activation='tanh', return_sequences=True, input_shape=(n_steps_in, len(features))),
        Dropout(0.3),
        LSTM(100, activation='tanh'),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(n_steps_out)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    st.write("Modelo LSTM creado.")
except Exception as e:
    st.error(f"Error al construir el modelo LSTM: {e}")
    st.stop()

# Entrenamiento
try:
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    st.write("Entrenamiento completado.")
except Exception as e:
    st.error(f"Error durante el entrenamiento: {e}")
    st.stop()

# Guardar el modelo
try:
    model.save("lstm_model.h5")
    st.write("Modelo LSTM guardado como 'lstm_model.h5'.")
except Exception as e:
    st.error(f"Error al guardar el modelo: {e}")
    st.stop()

# Predicciones y evaluación
try:
    predictions = model.predict(X_test)
    def denormalize(scaled_data, scaler, feature_index):
        dummy = np.zeros((scaled_data.shape[0], scaler.n_features_in_))
        dummy[:, feature_index] = scaled_data
        return scaler.inverse_transform(dummy)[:, feature_index]

    predicted_close = denormalize(predictions.flatten(), scaler, features.index('close'))
    real_close = denormalize(y_test.flatten(), scaler, features.index('close'))

    mae = mean_absolute_error(real_close, predicted_close)
    rmse = np.sqrt(mean_squared_error(real_close, predicted_close))
    r2 = r2_score(real_close, predicted_close)

    st.write(f"MAE: {mae}")
    st.write(f"RMSE: {rmse}")
    st.write(f"R² Score: {r2}")
except Exception as e:
    st.error(f"Error durante las predicciones o evaluación: {e}")
    st.stop()

# Gráficos
try:
    plt.figure(figsize=(12, 6))
    plt.plot(real_close[:100], label='Real', color='blue')
    plt.plot(predicted_close[:100], label='Predicción', color='orange')
    plt.title('Predicción vs Valores Reales (Primeras 100 muestras)')
    plt.xlabel('Índice')
    plt.ylabel('Precio')
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error al generar los gráficos: {e}")
