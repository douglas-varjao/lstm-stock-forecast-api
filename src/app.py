import os 
from flask import Flask
from flasgger import Swagger
from dotenv import load_dotenv
from src.predictor import predictor_blueprint
from src.auth import auth_bleprint, auth

load_dotenv()

def create_app():
    app=Flask(__name__)

    #Configuração do swagger
    swagger = Swagger(app)

    #registro de Blueprints
    app.register_blueprint(auth_bleprint, url_prefix='/auth')
    app.register_blueprint(predictor_blueprint, url_prefix='/api')

    @app.route('/')
    def index():
        return {"message": "PETR4 Prediction API is running", "auth_docs":"/apidocs"}
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)