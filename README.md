# 📈 LSTM Stock Forecast API (PETR4)

Este projeto foi desenvolvido como parte do **Tech Challenge da Fase 4 - Pos Tech Machine Learning Engineering**. O objetivo é prever o preço de fechamento das ações da Petrobras (PETR4) utilizando Redes Neurais Recorrentes.

## 🚀 Tecnologias Utilizadas
- **Python 3.11+**
- **TensorFlow/Keras**: Para a arquitetura LSTM.
- **FastAPI**: Para a criação da API de previsão.
- **YFinance**: Coleta de dados históricos da B3.
- **Scikit-Learn**: Pré-processamento e métricas.

## 📂 Estrutura do Projeto
- `/src`: Scripts de treinamento (`train.py`) e API (`main.py`).
- `/models`: Arquivos exportados do modelo (.keras) e scaler (.pkl).
- `requirements.txt`: Dependências do projeto.

## 📊 Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Rode a API: `python src/main.py`
3. Acesse a documentação: `http://localhost:8000/docs`