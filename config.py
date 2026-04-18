import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fallback-dev-key-change-in-production'

    # Mail
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 465
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

    # Database
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    DB_NAME = os.environ.get('DB_NAME', 'botnetattack')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', 20))

    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CHATBOT_DATA_DIR = os.path.join(BASE_DIR, 'chatbot_data')
    UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    DB_NAME = os.environ.get('DB_NAME', 'botnetattack') + '_test'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}