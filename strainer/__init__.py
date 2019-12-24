from flask import Flask 

from .extensions import mongo

from .main import main

from flask_cors import CORS, cross_origin

def create_app(config_object='strainer.settings'):
    app = Flask(__name__)

    app.config.from_object(config_object)

    CORS(app)

    app.config['CORS_HEADERS'] = 'Content-Type'

    mongo.init_app(app)

    app.register_blueprint(main)

    return app