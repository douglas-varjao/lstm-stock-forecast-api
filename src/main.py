import os
#OS AJUSTES DEVEM VIRT ANTES DO IMPORT DO TENSORFLOW PARA EVITAR O ERRO CUDA 303
#0 - todos os logs, 1 - filtro de avisos, 2 - apenas erros, 3 - sem logs
os.environ['TF_CPP__MIN_LOG_LEVEL'] = '3'
#Força o uso da cpu e ignora a busca por gpu/drivers da nvidia
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI, HTTPException
import tensorflow as tf
import joblib
import numpy as np
import uvicorn



app = FastAPI(title="API de previsão PETR4 - Tech Challenge")

#caminho dos arquivos
MODEL_PATH = 'models/lstm_model_petr4.keras'
SCALER_PATH = 'models/scaler_petr4.pkl'


#carregaando os artefatos

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    raise FileNotFoundError("Modelo ou Scaler não encontrdos na pasta models/")

@app.get("/")
def read_root():
    return {"message": "API de previsão da Petrobras est ONLINE!", "status": "Pronta para previsões."}

@app.post("/predict")
async def predict(data: list[float]):
    if len(data) != 60:
        #validação obrigatoria: o modelo precisaa do lookback de 60 dias
        raise HTTPException(status_code=400, detail="A entrada deve conter exatamente 60 valores.")
    try:
        #Transformar lista em array numpy e redimensionar/ normalizar
        input_array = np.array(data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)

        #reshape para o formato 3DD da LSTM: 1 amostra 60 timesteps 1 feature
        input_reshaped = input_scaled.reshape(1, 60, 1)

        #realizar a predição
        prediction_scaled = model.predict(input_reshaped)

        #desfazer a normalização para obter o valor real (R$)
        prediction_real = scaler.inverse_transform(prediction_scaled)

        return{"predicted_price": float(prediction_real[0][0]),
               "moeda": "BRL"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a predição:{str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)