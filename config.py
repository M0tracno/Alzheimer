import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))


class Config:
    # Set SECRET_KEY in .env so logins survive a restart; a random key is used otherwise.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB, matches the upload form
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    DATABASE = os.path.join(BASE_DIR, 'database.db')

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # ResNet-50 model from the NeuroThermo paper: 128x128 RGB input, 4 classes.
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_alzheimer.keras')
    IMAGE_SIZE = (128, 128)

    # Default admin, created on first start. Override all three in .env.
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@neuroscan.ai')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')


os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
