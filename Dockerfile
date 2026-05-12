#Imagem base do python (leve e estavel)
FROM python:3.11-slim

#Define o diretorio de trabalho dentro do container
WORKDIR /app

#Instala dependencias de sintaema para TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#Copia os arquivos de requisitos e instala as bibliotecas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copia as pastas do progeto
COPY src/ ./src/
COPY models/ ./models/

#Expões a porta da API
EXPOSE 8000

#Para rodar a API(0.0.0.0 é obrigatorio)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]