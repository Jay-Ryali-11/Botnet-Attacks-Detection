from functools import wraps
from flask import session, redirect, url_for, current_app
from random import randint
from datetime import datetime, timedelta
from flask_mail import Message
from app.utils.logger import get_logger

logger = get_logger(__name__)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def send_otp_email(mail, subject, body_message, receiver, user):
    otp = randint(100000, 999999)
    otp_expiry = datetime.utcnow() + timedelta(minutes=5)
    msg = Message(
        subject,
        sender=current_app.config['MAIL_USERNAME'],
        recipients=[receiver]
    )
    msg.body = f"""Dear {user},

{body_message}

Your OTP code is: {otp}
It is valid for 5 minutes from the time of generation.

Do not share the OTP with anyone.

If you did not request this OTP, contact support immediately.

Thank you,
Botnet Attack Application Team

Note: This is a system-generated email. Please do not reply.
"""
    
    try:
        mail.send(msg)
        logger.info(f"OTP email sent to {receiver}")
        return otp, otp_expiry
    except Exception as e:
        logger.error(f"Failed to send OTP email to {receiver}: {e}")
        return None, None