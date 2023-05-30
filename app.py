from flask import Flask
from predict import predict_bp
from flask_swagger_ui import get_swaggerui_blueprint
from train import train_bp

app = Flask(__name__)
app.register_blueprint(predict_bp)

app.register_blueprint(train_bp)

SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "Wine Quality Prediction API"},
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == "__main__":
    app.run(host="192.168.0.111", port=5000, debug=True)
