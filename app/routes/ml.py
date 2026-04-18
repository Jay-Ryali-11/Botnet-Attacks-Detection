import os
from flask import Blueprint, render_template, request, session, jsonify, redirect, url_for
from app.utils.auth_helpers import login_required
from app.ml.validator import validate_prediction_input
from app.ml.predictor import (
    evaluate_all_models, predict, generate_analysis_message, load_models
)
from app.utils.logger import get_logger
import pandas as pd

ml_bp = Blueprint('ml', __name__)
logger = get_logger(__name__)


@ml_bp.route('/setup_model', methods=['GET'])
def setup_model():
    try:
        load_models()
        evaluate_all_models()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/get_accuracy', methods=['GET'])
@login_required
def get_accuracy():
    model_name = request.args.get('model', '').upper()
    results = evaluate_all_models()
    if model_name not in results:
        return jsonify({'accuracy': 'N/A'})
    return jsonify({'accuracy': results[model_name]['accuracy']})


@ml_bp.route('/viewdata', methods=['GET', 'POST'])
@login_required
def viewdata():
    from flask import current_app
    user = session.get('logged_user')
    uploads_dir = current_app.config['UPLOADS_DIR']
    os.makedirs(uploads_dir, exist_ok=True)

    uploaded_files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
    table_html = None
    dataset_name = None
    error_message = None

    if request.method == 'POST':
        if 'dataset' in request.files:
            dataset_file = request.files['dataset']
            if dataset_file and dataset_file.filename.endswith('.csv'):
                save_path = os.path.join(uploads_dir, dataset_file.filename)
                dataset_file.save(save_path)
                df = pd.read_csv(save_path).head(1000)
                table_html = df.to_html(classes='table table-striped table-hover', index=False)
                dataset_name = dataset_file.filename
                if dataset_file.filename not in uploaded_files:
                    uploaded_files.append(dataset_file.filename)
            else:
                error_message = "Please upload a valid CSV file."

        elif 'default_dataset' in request.form:
            selected = request.form['default_dataset']
            if selected:
                path = os.path.join(uploads_dir, selected)
                if os.path.exists(path):
                    df = pd.read_csv(path).head(1000)
                    table_html = df.to_html(classes='table table-striped table-hover', index=False)
                    dataset_name = selected
                else:
                    error_message = "Selected dataset not found."

    return render_template('viewdata.html',
        table=table_html, error=error_message,
        uploaded_files=uploaded_files, dataset_name=dataset_name,
        user_name=user)


@ml_bp.route('/algo', methods=['GET', 'POST'])
@login_required
def algo():
    user = session.get('logged_user')
    return render_template('algo.html', user_name=user)


@ml_bp.route('/analysis')
@login_required
def analysis():
    user = session.get('logged_user')
    results = evaluate_all_models()
    message = generate_analysis_message(results)
    return render_template('analysis.html',
        user_name=user,
        ann_res=results.get('ANN'),
        rnn_res=results.get('RNN'),
        lstm_res=results.get('LSTM'),
        analysis_message=message)


@ml_bp.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    user = session.get('logged_user')

    if request.method == 'POST':
        values, algorithm, errors = validate_prediction_input(request.form)

        if errors:
            return render_template('prediction.html',
                msg='Validation error: ' + ' | '.join(errors),
                user_name=user)

        try:
            result = predict(algorithm, values)
            logger.info(f"Prediction made by {user}: {algorithm} → {result}")
            return render_template('prediction.html',
                msg=f'Predicted Attack Category: {result}',
                user_name=user)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template('prediction.html',
                msg=f'Prediction failed: {str(e)}',
                user_name=user)

    return render_template('prediction.html', user_name=user)