from flask import Flask

from .config import Config
from .routes import bp


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(bp)
    return app
