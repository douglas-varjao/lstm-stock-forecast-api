import os 
from flask import Flask
from flasgger import Swagger
from dotenv import load_dotenv
from auth import auth_blueprint
from predictor import predictor_blueprint

load_dotenv()

def create_app():
    app=Flask(__name__)

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/apispec_1.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }

    template = {
        "swagger": "2.0",
        "info": {
            "title": "API de Previsão PETR4 - Tech Challenge [cite: 14]",
            "description": "API RESTful utilizando LSTM para previsão de preços de ações da Petrobras protegida por autenticação. [cite: 14, 33]",
            "version": "1.0.0"
        },
        "securityDefinitions": {
            "BasicAuth": {
                "type": "basic",
                "description": "Insira o usuário e senha configurados no seu arquivo .env"
            }
        }
    }

    # Inicializa o Swagger com o template de segurança
    swagger = Swagger(app, config=swagger_config, template=template)

    #Configuração do swagger
    swagger = Swagger(app)

    #registro de Blueprints
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(predictor_blueprint, url_prefix='/api')

    @app.route('/')
    def index():
        return {"message": "PETR4 Prediction API is running", "auth_docs":"/apidocs"}
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)