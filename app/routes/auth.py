import re
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from app.models.user import (
    get_user_by_email, get_user_safe, user_exists,
    create_user, verify_password, update_password
)
from app.utils.auth_helpers import send_otp_email
from app.utils.logger import get_logger

auth_bp = Blueprint('auth', __name__)
logger = get_logger(__name__)

PHONE_REGEX = re.compile(r'^[1-9][0-9]{9}$')


@auth_bp.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        name        = request.form['name'].strip()
        email       = request.form['email'].strip().lower()
        password    = request.form['password']
        confirm     = request.form['confirmpassword']
        phonenumber = request.form['phonenumber'].strip()
        address     = request.form['Address'].strip()

        if not PHONE_REGEX.match(phonenumber):
            return render_template('registration.html',
                msg="Invalid phone number. Must be 10 digits and cannot start with 0.")

        if password != confirm:
            return render_template('registration.html', msg="Passwords do not match.")

        if user_exists(email, phonenumber):
            return render_template('registration.html',
                msg="Email or phone number already registered.")

        from app import mail
        otp, expiry = send_otp_email(
            mail,
            subject='Email Verification — Botnet Attack Application',
            body_message='You have requested OTP verification for the Botnet Attack Application.',
            receiver=email,
            user=name
        )

        if otp is None:
            return render_template('registration.html',
                msg="Failed to send OTP. Please try again later.")

        session['current_otp'] = otp
        session['otp_expiry'] = expiry.isoformat()
        session['registration_data'] = {
            'name': name, 'email': email, 'password': password,
            'phonenumber': phonenumber, 'address': address
        }
        return render_template('verify.html')

    return render_template('registration.html')


@auth_bp.route('/verify', methods=['POST'])
def verify():
    user_otp    = request.form.get('otp', '').strip()
    stored_otp  = session.get('current_otp')
    expiry_str  = session.get('otp_expiry')

    if not stored_otp or not expiry_str:
        return render_template('verify.html',
            message="OTP session expired. Please register again.", error=True)

    expiry = datetime.fromisoformat(expiry_str)
    if datetime.utcnow() > expiry:
        session.pop('current_otp', None)
        session.pop('otp_expiry', None)
        return render_template('verify.html',
            message="OTP has expired. Please request a new one.", error=True)

    try:
        if int(stored_otp) != int(user_otp):
            return render_template('verify.html',
                message="Incorrect OTP. Please try again.", error=True)
    except ValueError:
        return render_template('verify.html',
            message="Invalid OTP format.", error=True)

    data = session.pop('registration_data', None)
    session.pop('current_otp', None)
    session.pop('otp_expiry', None)

    if not data:
        return render_template('verify.html',
            message="Session data lost. Please register again.", error=True)

    create_user(data['name'], data['email'], data['password'],
                data['phonenumber'], data['address'])

    return render_template('verify.html',
        message="Email verified! Your account has been created. You can now log in.",
        success=True)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email'].strip().lower()
        password = request.form['password']

        user = verify_password(email, password)
        if user:
            session['logged_in']         = True
            session['user_id']           = user[0]
            session['logged_user']       = user[1]
            session['logged_user_mail']  = user[2]
            logger.info(f"User logged in: {email}")
            return render_template('loading.html')
        else:
            user_record = get_user_by_email(email)
            if not user_record:
                return render_template('login.html', msg="No account found with that email.")
            return render_template('login.html', msg="Incorrect password.")

    return render_template('login.html')


@auth_bp.route('/check-email', methods=['POST'])
def check_email():
    email = request.form.get('email', '').strip().lower()
    user = get_user_by_email(email)

    if not user:
        return jsonify({'status': 'error', 'message': 'Email not found. Please register first.'})

    from app import mail
    otp, expiry = send_otp_email(
        mail,
        subject=f'Password Reset — Botnet Attack Application',
        body_message='You have requested an OTP to reset your password.',
        receiver=email,
        user=user[1]
    )

    if otp is None:
        return jsonify({'status': 'error', 'message': 'Failed to send OTP. Try again later.'})

    session['current_otp']  = otp
    session['otp_expiry']   = expiry.isoformat()
    session['reset_email']  = email
    return jsonify({'status': 'success', 'message': 'OTP sent to your email.'})


@auth_bp.route('/reset', methods=['GET', 'POST'])
def reset():
    if not session.get('reset_email') or not session.get('current_otp'):
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        expiry_str  = session.get('otp_expiry')
        stored_otp  = session.get('current_otp')
        email       = session.get('reset_email')
        expiry      = datetime.fromisoformat(expiry_str)

        if datetime.utcnow() > expiry:
            session.pop('current_otp', None)
            session.pop('otp_expiry', None)
            session.pop('reset_email', None)
            return render_template('pwdreset.html', msg="OTP expired. Please request a new one.")

        entered_otp = request.form.get('otp', '').strip()
        try:
            if int(entered_otp) != int(stored_otp):
                return render_template('pwdreset.html', msg="Incorrect OTP.")
        except ValueError:
            return render_template('pwdreset.html', msg="Invalid OTP format.")

        new_password = request.form.get('new_password', '')
        if len(new_password) < 6:
            return render_template('pwdreset.html', msg="Password must be at least 6 characters.")

        update_password(email, new_password)
        session.pop('current_otp', None)
        session.pop('otp_expiry', None)
        session.pop('reset_email', None)
        logger.info(f"Password reset successful for: {email}")

        return render_template('pwdreset.html',
            msg="Password changed successfully! Please log in.", success=True)

    return render_template('pwdreset.html')


@auth_bp.route('/logout')
def logout():
    email = session.get('logged_user_mail', 'unknown')
    session.clear()
    logger.info(f"User logged out: {email}")
    return redirect(url_for('auth.login'))