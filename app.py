from random import randint
from functools import wraps
from flask import Flask, render_template, url_for, redirect, request, jsonify, render_template_string, session
from flask_mail import Mail, Message
from sklearn.metrics import accuracy_score
from werkzeug.urls import unquote

import sys
# sys.path.append('../backend')
from chat import get_response

app = Flask(__name__)
import pandas as pd
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import os
import re
import secrets
from datetime import datetime, timedelta
import pytz

import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='codexslayer',
    passwd='code@123',
    database='botnetattack'
)

mycur = mydb.cursor()


@app.before_request
def before_request():
    ALLOWED_URLS = ['/viewdata', '/algo', '/prediction', '/get_accuracy', '/analysis', '/setup_model', '/profile']
    if request.path.startswith('/static/'):
        return
    if 'logged_in' in session:
        print("requested pathhhh1: ", request.path)
        if request.path not in ALLOWED_URLS:
            print("requested pathhhh: ", request.path)
            session.pop('logged_in', None)
            session.pop('user_id', None)
            session.modified = True
            return redirect(url_for('login'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return redirect(url_for('index') + '#about')

@app.route('/get_response', methods=['POST'])
def chat_response():
    user_message = request.json.get('message')
    print(f"Received message: {user_message}")
    if user_message:
        response = get_response(user_message)
        print(f"Bot response: {response}")
        return jsonify({"response": response})
    return jsonify({"response": "Sorry, I didn't understand that."})


# Flask-Mail Setup (for sending OTP)
app.secret_key = 'mail'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'thecodexcipher@gmail.com'
app.config['MAIL_PASSWORD'] = 'upibhbkdvhbclinc'
mail = Mail(app)

def sendEmailVerificationRequest(message,submsg,receiver,user):
    otp = randint(100000, 999999)
    otp_expiry = datetime.now() + timedelta(minutes=5)
    msg = Message(message, sender=app.config['MAIL_USERNAME'], recipients=[receiver])
    msg.body = f"""
    Dear {user},

    {submsg}

    Your OTP code is: {otp}
    It is valid for 5 minutes from the time of generation.

    Do not share the OTP with anyone to avoid misuse of your account.

    If you have not requested this OTP, please contact the "support team" immediately.

    Thank you,
    Botnet Attack Application Team

    Note: This is a system-generated email, please do not reply to this email. If you wish to unsubscribe from these emails, please follow the unsubscribe instructions on our platform.
    """
    try:
        mail.send(msg)
        return otp, otp_expiry
    except Exception as e:
        print(f"Failed to send email: {e}")
        return None, None

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber = request.form['phonenumber']
        address = request.form['Address']

        phone_regex = r'^[1-9][0-9]{9}$'
        if not re.match(phone_regex, phonenumber):
            msg = "Invalid phone number. It must be 10 digits, cannot start with a '+' symbol, and cannot have leading zeroes."
            return render_template('registration.html', msg=msg)

        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s or phonenumber = %s'
            val = (email, phonenumber,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                current_otp, otp_expiry = sendEmailVerificationRequest(message='Email Verification For Botnet Attack Application',submsg='You have successfully generated an OTP for the Botnet Attack Application Portal.',receiver=email, user=name)
                if current_otp is not None:
                    session['current_otp'] = current_otp
                    session['otp_expiry'] = otp_expiry
                    session['registration_data'] = {
                        'name': name,
                        'email': email,
                        'password': password,
                        'phonenumber': phonenumber,
                        'address': address
                    }
                    return render_template('verify.html')
                else:
                    msg = "There was an error sending the OTP. Please try again later."
                    return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/verify', methods=["POST"])
def verify():
    user_otp = request.form['otp']
    current_user_otp = session.get('current_otp')
    otp_expiry = session.get('otp_expiry')

    if not current_user_otp or not otp_expiry:
        return render_template('verify.html', message="Error: OTP not found. Please try again.", error=True)

    current_time = datetime.now(pytz.utc)
    if current_time > otp_expiry:
        session.pop('current_otp', None)
        session.pop('otp_expiry', None)
        session.modified = True
        return render_template('verify.html', message="Oops! The OTP has expired. Please request a new one.",
                               error=True)

    if int(current_user_otp) == int(user_otp):
        user_data = session['registration_data']
        sql = 'INSERT INTO users (name, email, password, phonenumber, Address) VALUES (%s, %s, %s, %s, %s)'
        val = (
            user_data['name'], user_data['email'], user_data['password'], user_data['phonenumber'],
            user_data['address'])
        mycur.execute(sql, val)
        mydb.commit()
        session.pop('current_otp', None)
        session.pop('otp_expiry', None)
        session.pop('registration_data', None)
        session.modified = True
        return render_template('verify.html',
                               message="Your email has been successfully verified and your account has been created!",
                               success=True)
    else:
        return render_template('verify.html', message="Oops! Email Verification Failure, OTP does not match.",
                               error=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
                session['logged_in'] = True
                session['user_id'] = data[0]
                session['logged_user'] = data[0]
                session['logged_user_mail'] = data[1]
                print(session['logged_user'])
                # setup_model_results()
                # return redirect('/viewdata')
                return render_template('loading.html')
            else:
                msg = 'Password does not match!'
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)  # Pass msg here
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)  # Pass msg here
    return render_template('login.html')

@app.route('/setup_model', methods=['GET'])
def setup_model():
    try:
        setup_model_results()
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error during setup: {str(e)}")  # Log the error
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/check-email', methods=['POST'])
def check_email():
    email = request.form['email']
    sql = 'SELECT * FROM users WHERE email=%s'
    val = (email,)
    mycur.execute(sql, val)
    data = mycur.fetchone()

    if data:
        user = data[0]
        current_otp, otp_expiry = sendEmailVerificationRequest(
            message=f'Password Reset Request For {user} for Botnet Attack Application',
            submsg='You have successfully generated an OTP to reset your password.',
            receiver=email, user=user
        )
        session['current_otp'] = current_otp
        session['email'] = email
        session['otp_expiry'] = otp_expiry

        return jsonify({"status": "success", "message": "Email found! A password reset link has been sent."})
    else:
        return jsonify({"status": "error", "message": "Email does not exist. Please register first."})

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if not session.get('email') or not session.get('current_otp') or not session.get('otp_expiry'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Extract session data
        email = session.get('email')
        current_otp = session.get('current_otp')
        otp_expiry = session.get('otp_expiry')

        # Check if OTP has expired
        current_time = datetime.now(pytz.utc)
        if current_time > otp_expiry:
            msg = "OTP has expired. Please request a new one."
            return render_template('pwdreset.html', msg=msg, success=False)

        # Validate the OTP entered by the user
        entered_otp = request.form.get('otp')
        if int(entered_otp) != int(current_otp):
            msg = "The OTP you entered is incorrect. Please check your email and try again."
            return render_template('pwdreset.html', msg=msg)

        # If OTP is correct, reset password
        new_password = request.form.get('new_password')
        sql = "UPDATE users SET password=%s WHERE email=%s"
        val = (new_password, email)
        mycur.execute(sql, val)
        mydb.commit()

        # Clear session variables after successful password reset
        print("Clearing session data...")
        session.pop('current_otp', None)
        session.pop('email', None)
        session.pop('otp_expiry', None)
        session.modified = True

        print("Session after clearing:", session)

        msg = "Password changed successfully! Please log in with your new password."
        return render_template('pwdreset.html', msg=msg, success=True)

    # Render reset form if not POST request
    return render_template('pwdreset.html')


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/submit-question', methods=['POST'])
def submit_question():
    name = request.form.get('name')
    email = request.form.get('email')
    question = request.form.get('question')
    if not name or not email or not question:
        return jsonify({"error": "All fields are required"}), 400
    subject = f"Botnet Security Support – User Inquiry from {name}"
    body = f"""
    Hi Vardhan,

    You have received a new support request regarding the Botnet Application.

    🔹 **User Name:** {name}  
    🔹 **Email:** {email}  
    🔹 **Question:**  

    {question}

    Please address this inquiry at the earliest convenience.

    Best Regards,  
    Botnet Support System
    """

    try:
        msg = Message(subject,  sender=app.config['MAIL_USERNAME'], recipients=[app.config['MAIL_USERNAME']], body=body)
        mail.send(msg)
        return jsonify({"success": "Your message is on its way!"}), 200
    except Exception as e:
        return jsonify({"error": "There was an issue submitting your question. Ensure your details are correct and try again."}), 500


if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/viewdata', methods=['GET', 'POST'])
@login_required
def viewdata():
    user = session.get('logged_user')
    default_dataset, uploaded_files, table_html, dataset_name, error_message = 'UNSW_NB15.csv', [], None, None, None
    for filename in os.listdir('uploads'):
        if filename.endswith('.csv'):
            uploaded_files.append(filename)
    if request.method == 'POST':
        if 'dataset' in request.files:
            dataset_file = request.files['dataset']
            if dataset_file:
                dataset_path = os.path.join('uploads', dataset_file.filename)
                dataset_file.save(dataset_path)
                uploaded_files.append(dataset_file.filename)
                df = pd.read_csv(dataset_path)
                df = df.head(1000)
                table_html = df.to_html(classes='table table-striped table-hover', index=False)
                dataset_name = dataset_file.filename
        elif 'default_dataset' in request.form:
            selected_default_dataset = request.form['default_dataset']
            if selected_default_dataset:
                dataset_path = os.path.join('uploads', selected_default_dataset)
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    df = df.head(1000)
                    table_html = df.to_html(classes='table table-striped table-hover', index=False)
                    dataset_name = selected_default_dataset
                else:
                    error_message = "Dataset not found in uploads folder"
    return render_template('viewdata.html', table=table_html, error=error_message,
                           uploaded_files=uploaded_files, default_dataset=default_dataset, dataset_name=dataset_name, user_name=user)


# Load models
ann_model = tf.keras.models.load_model('saved models/ann_model.h5')
# cnn_model = tf.keras.models.load_model('saved models/cnn_model.h5')
rnn_model = tf.keras.models.load_model('saved models/rnn_model.h5')
lstm_model = tf.keras.models.load_model('saved models/lstm_model.h5')

# Load the dataset
data = pd.read_csv('UNSW_NB15.csv')
data_cleaned = data.drop(columns=['id', 'label'])
label_encoder = LabelEncoder()
data_cleaned['attack_cat_encoded'] = label_encoder.fit_transform(data_cleaned['attack_cat'])
data_cleaned = data_cleaned.drop(columns=['attack_cat'])

# Encode categorical columns
categorical_columns = ['proto', 'service', 'state']
for column in categorical_columns:
    data_cleaned[column] = label_encoder.fit_transform(data_cleaned[column])

X = data_cleaned.drop(columns=['attack_cat_encoded'])
y = data_cleaned['attack_cat_encoded']

# Feature selection
k_best_selector = SelectKBest(score_func=chi2, k=10)
X_new = k_best_selector.fit_transform(X, y)
selected_features = X.columns[k_best_selector.get_support(indices=True)]
X_selected = data_cleaned[selected_features]
print(X_selected.columns)

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report

from sklearn.metrics import classification_report


def evaluate_model(model_name):
    if model_name == 'ANN':
        model = ann_model
    elif model_name == 'RNN':
        model = rnn_model
    elif model_name == 'LSTM':
        model = lstm_model
    else:
        return "Invalid model name"

    if model_name in ['ANN', 'CNN', 'RNN', 'LSTM']:
        y_pred = model.predict(X_test).argmax(axis=1)
    else:
        y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = round(report['accuracy'] * 100)
    macro_avg_precision = round(report['macro avg']['precision'],2)
    macro_avg_recall = round(report['macro avg']['recall'],2)
    macro_avg_f1 = round(report['macro avg']['f1-score'],2)
    weighted_avg_precision = round(report['weighted avg']['precision'],2)
    weighted_avg_recall = round(report['weighted avg']['recall'],2)
    weighted_avg_f1 = round(report['weighted avg']['f1-score'],2)
    metrics = {
        'accuracy': accuracy,
        'macro_avg_precision': macro_avg_precision,
        'macro_avg_recall': macro_avg_recall,
        'macro_avg_f1': macro_avg_f1,
        'weighted_avg_precision': weighted_avg_precision,
        'weighted_avg_recall': weighted_avg_recall,
        'weighted_avg_f1': weighted_avg_f1
    }
    return metrics

ann_res = None
rnn_res = None
lstm_res = None

def setup_model_results():
    global ann_res, rnn_res, lstm_res
    ann_res = evaluate_model('ANN')
    rnn_res = evaluate_model('RNN')
    lstm_res = evaluate_model('LSTM')


@app.route('/get_accuracy', methods=['GET'])
@login_required
def get_accuracy():
    model_name = request.args.get('model')
    print(model_name.upper())
    global ann_res, rnn_res, lstm_res
    accuracy = ''
    if model_name:
        if model_name.upper() == 'ANN':
            accuracy = ann_res['accuracy']
        elif model_name.upper() == 'RNN':
            accuracy = rnn_res['accuracy']
        elif model_name.upper() == 'LSTM':
            accuracy = lstm_res['accuracy']
    print("accuracy: ",accuracy)
    return jsonify({'accuracy': accuracy})

@app.route('/algo', methods=['GET', 'POST'])
@login_required
def algo():
    user = session.get('logged_user')
    model_name = request.form.get('model')
    return render_template('algo.html', user_name=user)


def generate_analysis_message(ann_res, rnn_res, lstm_res):
    ann_accuracy = ann_res['accuracy']
    rnn_accuracy = rnn_res['accuracy']
    lstm_accuracy = lstm_res['accuracy']

    best_model = ''
    if lstm_accuracy > ann_accuracy and lstm_accuracy > rnn_accuracy:
        best_model = 'LSTM'
    elif ann_accuracy > rnn_accuracy:
        best_model = 'ANN'
    else:
        best_model = 'RNN'

    message = f"Based on the analysis, {best_model} outperforms the other models across key metrics such as accuracy, precision, recall, and F1-score."

    if best_model == 'LSTM':
        message += f" With an accuracy of {lstm_accuracy}%, LSTM significantly surpasses both ANN ({ann_accuracy}%) and RNN ({rnn_accuracy}%)."
        message += " It also handles class imbalance well, has a lower risk of overfitting, and is highly suitable for time-series data."
        message += " Although it requires more training time and computational resources, its overall performance makes it the optimal choice for tasks that demand high accuracy and reliable predictions."
    elif best_model == 'ANN':
        message += f" With an accuracy of {ann_accuracy}%, ANN performs better than RNN ({rnn_accuracy}%) but lags behind LSTM ({lstm_accuracy}%)."
    else:
        message += f" With an accuracy of {rnn_accuracy}%, RNN performs better than ANN ({ann_accuracy}%) but lags behind LSTM ({lstm_accuracy}%)."
    return message



@app.route('/analysis')
@login_required
def analysis():
    user = session.get('logged_user')
    global ann_res, rnn_res, lstm_res
    analysis_message = generate_analysis_message(ann_res, rnn_res, lstm_res)
    return render_template('analysis.html', user_name=user, ann_res=ann_res, rnn_res=rnn_res, lstm_res=lstm_res, analysis_message=analysis_message)


# Dictionary mapping encoded values to attack categories
attack_cat_mapping = {
    0: 'Analysis',
    1: 'Backdoor',
    2: 'DoS',
    3: 'Exploits',
    4: 'Fuzzers',
    5: 'Generic',
    6: 'Normal',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms'
}


@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    user = session.get('logged_user')
    if request.method == 'POST':
        try:
            selected_algorithm = request.form.get('algorithm')
            print("model selected:", selected_algorithm)
            if not selected_algorithm:
                return render_template('prediction.html', msg='No algorithm selected. Please select a valid algorithm.', user_name=user)
            if selected_algorithm == 'ANN':
                model = ann_model
            elif selected_algorithm == 'RNN':
                model = rnn_model
            elif selected_algorithm == 'LSTM':
                model = lstm_model
            else:
                return render_template('prediction.html', msg='Invalid algorithm selected', user_name=user)

            input_data = [
                float(request.form['sbytes']),
                float(request.form['dbytes']),
                float(request.form['rate']),
                float(request.form['sload']),
                float(request.form['dload']),
                float(request.form['sinpkt']),
                float(request.form['sjit']),
                float(request.form['stcpb']),
                float(request.form['dtcpb']),
                float(request.form['response_body_len'])
            ]

            input_data = np.array([input_data])
            predicted_attack_idx = model.predict(input_data)
            predicted_attack_idx = np.argmax(predicted_attack_idx, axis=1)[0]
            predicted_attack_category = attack_cat_mapping[predicted_attack_idx]
            return render_template('prediction.html', msg=f'Predicted Attack Category: {predicted_attack_category}', user_name=user)
        except Exception as e:
            return render_template('prediction.html', msg=f'Error: {str(e)}', user_name=user)

    return render_template('prediction.html', user_name=user)


def get_user_data_by_email(email):
    sql = "SELECT * FROM users WHERE email=%s"
    val = (email,)
    mycur.execute(sql, val)
    user = mycur.fetchone()
    return user

def update_user_data(email, name, password, phonenumber, address):
    sql = '''
        UPDATE users
        SET name = %s, password = %s, phonenumber = %s, address = %s
        WHERE email = %s
    '''
    val = (name, password, phonenumber, address, email)
    mycur.execute(sql, val)
    mydb.commit()


@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    user_email = session['logged_user_mail']
    user_data = get_user_data_by_email(user_email)
    print(user_data)
    if request.method == 'POST':
        updated_name = request.form.get('name')
        updated_password = request.form.get('password')
        updated_phone = request.form.get('phone')
        updated_address = request.form.get('address')
        update_user_data(user_email, updated_name, updated_password, updated_phone, updated_address)
        user_data = get_user_data_by_email(user_email)
        return render_template("profile.html", user=user_data)

    return render_template("profile.html", user=user_data)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('logged_user', None)
    session.pop('logged_user_mail', None)
    session.modified = True
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)