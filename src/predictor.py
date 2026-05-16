import os

# Configurações de silenciamento do TensorFlow
#OS AJUSTES DEVEM VIR ANTES DO IMPORT DO TENSORFLOW PARA EVITAR O ERRO CUDA 303
#0 - todos os logs, 1 - filtro de avisos, 2 - apenas erros, 3 - sem logs
#Força o uso da cpu e ignora a busca por gpu/drivers da nvidia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import joblib
import tensorflow as tf
import numpy as np
from flask import Blueprint, request, jsonify
from auth import auth

predictor_blueprint = Blueprint('predictor', __name__)

#Caminho dos artefatos
MODEL_PATH = 'models/lstm_model_petr4.keras'
SCALER_PATH = 'models/scaler_petr4.pkl'

#carregamento seguro
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modelo ou scaler não encontrado. Certifique-se de treinar o modelo primeiro.")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    print(f"Erro ao carregar modelo ou scaler: {e}")
    model, scaler = None, None

@predictor_blueprint.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    """
    Realiza a previsão do preço da ação PETR4.SA
    ---
    tags:
      - Predição LSTM
    description: Envie uma lista com os últimos 60 preços de fechamento para prever o próximo dia.
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: array
          items:
            type: number
          example: [25.4, 25.6, 25.8, 26.0, 26.1, 26.3, 26.5, 26.7, 26.9, 27.0, 27.1, 27.3, 27.5, 27.6, 27.8, 28.0, 28.1, 28.3, 28.5, 28.6, 28.8, 29.0, 29.1, 29.3, 29.5, 29.6, 29.8, 30.0, 30.1, 30.3, 30.5, 30.6, 30.8, 31.0, 31.1, 31.3, 31.5, 31.6, 31.8, 32.0, 32.1, 32.3, 32.5, 32.6, 32.8, 33.0, 33.1, 33.3, 33.5, 33.6, 33.8, 34.0, 34.1, 34.3, 34.5, 34.6, 34.8, 35.0, 35.1, 35.3]
    responses:
      200:
        description: Previsão realizada com sucesso.
      401:
        description: Não autorizado - Usuário ou senha incorretos.
      400:
        description: Erro nos dados enviados (ex. quantidade diferente de 60).
    """
    if model is None or scaler is None:
        return jsonify({"error": "Modelo ou scaler não disponível. Treine o modelo primeiro."}), 500
    
    data_json = request.get_json()

    if not data_json or len(data_json) != 60:
        return jsonify({"error": "Entrada inválida. Envie um array JSON com 60 valores de fechamento."}), 400
    
    try:
        #logica de predição
        input_array = np.array(data_json).reshape(-1,1)
        input_scaled = scaler.transform(input_array)
        input_reshaped = input_scaled.reshape(1, 60, 1)

        prediction_scaled = model.predict(input_reshaped)
        prediction_real = scaler.inverse_transform(prediction_scaled)

        predicted_val = float(prediction_real[0][0])
        last_val = data_json[-1]
        change_pct = ((predicted_val - last_val) / last_val) *100

        return jsonify({
            "ticker": "PETR4.SA",
            "predicted_price": round(predicted_val, 2),
            "last_price": round(last_val, 2),
            "variation_percent": round(change_pct, 2),
            "currecy": "BRL",
            "model_info": "LSTM Deep Learning",
            "accuracy_metrics": {"MAPE_train": "2.3%"} 
        })
    except Exception as e:
        return jsonify({"error": f"Erro durante a predição: {str(e)}"}), 500