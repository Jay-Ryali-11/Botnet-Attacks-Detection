from werkzeug.security import generate_password_hash, check_password_hash
from app.utils.db import execute_query
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_user_by_email(email):
    return execute_query(
        'SELECT id, name, email, password, phonenumber, address FROM users WHERE email = %s',
        (email,),
        fetch='one'
    )


def get_user_safe(email):
    """Returns user WITHOUT password field — safe for templates."""
    return execute_query(
        'SELECT name, email, phonenumber, address FROM users WHERE email = %s',
        (email,),
        fetch='one'
    )


def user_exists(email, phonenumber):
    return execute_query(
        'SELECT id FROM users WHERE email = %s OR phonenumber = %s',
        (email, phonenumber),
        fetch='one'
    )


def create_user(name, email, plain_password, phonenumber, address):
    hashed = generate_password_hash(plain_password)
    execute_query(
        'INSERT INTO users (name, email, password, phonenumber, address) VALUES (%s, %s, %s, %s, %s)',
        (name, email, hashed, phonenumber, address),
        commit=True
    )
    logger.info(f"New user created: {email}")


def verify_password(email, plain_password):
    user = get_user_by_email(email)
    if not user:
        return None
    if check_password_hash(user[3], plain_password):
        return user
    return None


def update_user(email, name, plain_password, phonenumber, address):
    hashed = generate_password_hash(plain_password)
    execute_query(
        'UPDATE users SET name=%s, password=%s, phonenumber=%s, address=%s WHERE email=%s',
        (name, hashed, phonenumber, address, email),
        commit=True
    )
    logger.info(f"User profile updated: {email}")


def update_password(email, new_plain_password):
    hashed = generate_password_hash(new_plain_password)
    execute_query(
        'UPDATE users SET password=%s WHERE email=%s',
        (hashed, email),
        commit=True
    )
    logger.info(f"Password reset for: {email}")