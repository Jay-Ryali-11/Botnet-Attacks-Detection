from flask import Flask
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import config
from app.utils.db import init_pool
from app.utils.logger import setup_logger

mail = Mail()
csrf = CSRFProtect()
limiter = Limiter(get_remote_address)
 

def create_app(config_name='default'):
    setup_logger()

    app = Flask(__name__, template_folder='templates', static_folder='../static')
    app.config.from_object(config[config_name])

    mail.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.ml import ml_bp
    from app.routes.chatbot import chatbot_bp
    from app.routes.general import general_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(general_bp)

    csrf.exempt(chatbot_bp)

    try:
        with app.app_context():
            init_pool(app)
    except Exception as e:
        app.logger.error(f"Error initializing database pool: {str(e)}")
        raise

    return app