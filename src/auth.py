import os
from dotenv import load_dotenv
from flask import Blueprint, jsonify, request
from flask_httpauth import HTTPBasicAuth


auth_blueprint = Blueprint('auth', __name__)
auth = HTTPBasicAuth()    

#busca os valores de user e pass do .env
user = {
    os.getenv('API_USER_DOUGLAS'): os.getenv('API_PASS_DOUGLAS'),
    os.getenv('API_USER_ADMIN'): os.getenv('API_PASS_ADMIN')
}

@auth.verify_password
def verify_password(username, password):
    #verifica se o user e pass existem no dicionario do .env
    if username in user and user[username] == password:
        return username
    return None

@auth_blueprint.route('/status', methods=['GET'])
def status():
    """
    Verifica o status operacional da API
    ---
    tags:
      - Status e Autenticação
    description: Retorna se o servidor da API está online e respondendo adequadamente.
    responses:
      200:
        description: API está online e operando com sucesso.
    """
    return jsonify({
        'status': 'online',
        'service': 'LSTM Stock prediction API',
        'message': 'API is running and ready to accept requests.'
    })

