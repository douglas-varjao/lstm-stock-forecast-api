import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout



#====== Config ======#
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
LOOKBACK_DAYS = 60
MODEL_PATH = 'models/lstm_model_petr4.keras'
SCALER_PATH = 'models/scaler_petr4.pkl'

def train_model():
    #Garantir que a pasta models existe
    if not os.path.exists('models'):
        os.makedirs('models')

    #Coleta de dados
    print(f"Baixando dados de {SYMBOL}...")
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if df.empty:
        print(f"ERRO: Não foi possível baixar dados para {SYMBOL}. Verifique sua conexão ou o Ticker.")
        return # Para a execução se não houver dados

    # Filtrando apenas o fechamento
    data = df[['Close']].values

    #Pré-processamento e Normalização
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for x in range(LOOKBACK_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-LOOKBACK_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train =np.array(x_train), np.array(y_train)

    #Reshape para 3 [amostras, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    #arquitetura do modelo LSTM
    
    model = Sequential([
        LSTM(units= 50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    #treinameto
    print("Treinando o modelo...")
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1, shuffle=False)

    #Calculo de métricas de avaliação
    print("Calculando métricas de avaliação...")
    predictions = model.predict(x_train, verbose=0)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))

    mae = mean_absolute_error(y_train_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_train_rescaled, predictions_rescaled))
    mape = np.mean(np.abs((y_train_rescaled - predictions_rescaled) / y_train_rescaled)) * 100

    print(f"=== Métricas de performance ===")
    print((f"MAE:{mae:.2f}"))
    print((f"RMSE:{rmse:.2f}"))
    print((f"MAPE:{mape:.2f}%"))

      #salvamento de artefatos
    print("salvando modelo e scaler...")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Sucesso! Tudo pronto para o deploy")

if __name__ == "__main__":
    train_model()