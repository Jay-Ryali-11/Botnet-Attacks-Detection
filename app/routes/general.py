from flask import Blueprint, render_template, request, session, jsonify, redirect, url_for
from flask_mail import Message
from app.models.user import get_user_safe, get_user_by_email, update_user
from app.utils.auth_helpers import login_required
from app.utils.logger import get_logger

general_bp = Blueprint('general', __name__)
logger = get_logger(__name__)


@general_bp.route('/')
def index():
    return render_template('index.html')


@general_bp.route('/about')
def about():
    return redirect(url_for('general.index') + '#about')


@general_bp.route('/support')
def support():
    return render_template('support.html')


@general_bp.route('/submit-question', methods=['POST'])
def submit_question():
    from flask import current_app
    from app import mail

    name     = request.form.get('name', '').strip()
    email    = request.form.get('email', '').strip()
    question = request.form.get('question', '').strip()

    if not name or not email or not question:
        return jsonify({'error': 'All fields are required.'}), 400

    subject = f"Botnet Support Inquiry from {name}"
    body    = f"Name: {name}\nEmail: {email}\n\nQuestion:\n{question}"

    try:
        msg = Message(subject,
                      sender=current_app.config['MAIL_USERNAME'],
                      recipients=[current_app.config['MAIL_USERNAME']],
                      body=body)
        mail.send(msg)
        logger.info(f"Support question submitted by {email}")
        return jsonify({'success': 'Your message has been sent!'}), 200
    except Exception as e:
        logger.error(f"Support email failed: {e}")
        return jsonify({'error': 'Failed to send message. Please try again.'}), 500


@general_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_email = session['logged_user_mail']

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        password = request.form.get('password', '').strip()
        phone    = request.form.get('phone', '').strip()
        address  = request.form.get('address', '').strip()

        if not password:
            user_data =  get_user_by_email(user_email)
            return render_template('profile.html', user=user_data,
                                   msg="Password cannot be empty.")

        update_user(user_email, name, password, phone, address)
        logger.info(f"Profile updated for {user_email}")

    user_data =  get_user_by_email(user_email)
    return render_template('profile.html', user=user_data)