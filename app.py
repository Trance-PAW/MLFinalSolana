#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:14:00 2024

@author: trancepaw
"""
import streamlit as st
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
import os
import requests

# Configuración para el uso del GPU y optimización de memoria
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Título de la aplicación
st.title("Predicción del Precio de Solana con LSTM")



# Descargar el archivo desde Google Drive si no existe
def download_dataset():
    dataset_url = "https://drive.google.com/file/d/1x8kLHMPbYKFskn4n9gTJE_Wp0Ei-87cE/view?usp=sharing"  # Reemplaza con el ID del archivo de Google Drive
    local_filename = "solana_historical_data.csv"
    if not os.path.exists(local_filename):
        with requests.get(dataset_url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_filename

# Cargar el dataset
data_path = download_dataset()
solana_data = pd.read_csv(data_path)
st.write("Dataset cargado correctamente desde Google Drive.")

# Procesar datos
solana_data['timestamp'] = pd.to_datetime(solana_data['timestamp'])
solana_data.sort_values('timestamp', inplace=True)
filtered_data = solana_data[solana_data['timestamp'] >= solana_data['timestamp'].max() - pd.DateOffset(months=2)]
filtered_data.fillna(method='ffill', inplace=True)
filtered_data.fillna(method='bfill', inplace=True)

# Cálculo de indicadores técnicos
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

filtered_data['RSI'] = compute_rsi(filtered_data['close'])
ema12 = filtered_data['close'].ewm(span=12, adjust=False).mean()
ema26 = filtered_data['close'].ewm(span=26, adjust=False).mean()
filtered_data['MACD'] = ema12 - ema26
filtered_data['MACD_signal'] = filtered_data['MACD'].ewm(span=9, adjust=False).mean()
filtered_data['TR'] = np.maximum(filtered_data['high'] - filtered_data['low'], 
                                 np.maximum(abs(filtered_data['high'] - filtered_data['close'].shift(1)),
                                            abs(filtered_data['low'] - filtered_data['close'].shift(1))))
filtered_data['ATR'] = filtered_data['TR'].rolling(window=14).mean()
filtered_data.dropna(inplace=True)

# Normalización
features = ['close', 'RSI', 'MACD', 'MACD_signal', 'ATR']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(filtered_data[features])

# Crear secuencias
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:i + n_steps_in])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])  # Columna 'close'
    return np.array(X), np.array(y)

n_steps_in = 720  # 15 días de datos
n_steps_out = 3   # Predicción de 3 horas
X, y = create_sequences(scaled_features, n_steps_in, n_steps_out)

# Dividir en conjuntos de entrenamiento y prueba
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Crear o cargar modelo
model_path = "lstm_model_solana.h5"
if not os.path.exists(model_path):
    st.write("Entrenando un nuevo modelo LSTM...")
    model = Sequential([
        LSTM(200, activation='tanh', return_sequences=True, input_shape=(n_steps_in, len(features))),
        Dropout(0.3),
        LSTM(100, activation='tanh'),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(n_steps_out)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    model.save(model_path)
else:
    st.write("Cargando modelo preentrenado...")
    model = load_model(model_path)

# Predicciones
predictions = model.predict(X_test)
def denormalize(scaled_data, scaler, feature_index):
    dummy = np.zeros((scaled_data.shape[0], scaler.n_features_in_))
    dummy[:, feature_index] = scaled_data
    return scaler.inverse_transform(dummy)[:, feature_index]

predicted_close = denormalize(predictions.flatten(), scaler, features.index('close'))
real_close = denormalize(y_test.flatten(), scaler, features.index('close'))

# Calcular métricas
mae = mean_absolute_error(real_close, predicted_close)
rmse = np.sqrt(mean_squared_error(real_close, predicted_close))
r2 = r2_score(real_close, predicted_close)

st.write(f"### Métricas del modelo LSTM:")
st.write(f"- **MAE**: {mae:.2f}")
st.write(f"- **RMSE**: {rmse:.2f}")
st.write(f"- **R² Score**: {r2:.2f}")

# Gráficas
st.write("### Predicción vs Valores Reales")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(real_close[:100], label='Real', color='blue')
ax.plot(predicted_close[:100], label='Predicción', color='orange')
ax.set_title('Predicción vs Valores Reales (Primeras 100 muestras)')
ax.set_xlabel('Índice')
ax.set_ylabel('Precio')
ax.legend()
st.pyplot(fig)

# Predicciones futuras
last_sequence = np.expand_dims(scaled_features[-n_steps_in:], axis=0)
future_predictions = model.predict(last_sequence)
future_predictions_denorm = denormalize(future_predictions[0], scaler, features.index('close'))

st.write(f"### Predicciones futuras (en USD):")
st.write(f"- **Próxima hora**: {future_predictions_denorm[0]:.2f} USD")
st.write(f"- **Segunda hora**: {future_predictions_denorm[1]:.2f} USD")
st.write(f"- **Tercera hora**: {future_predictions_denorm[2]:.2f} USD")
