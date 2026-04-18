import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from flask import current_app
from app.ml.pipeline import get_pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

ATTACK_CAT_MAPPING = {
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

_models = {}
_results_cache = {}


def load_models():
    global _models
    if _models:
        return
    models_dir = current_app.config['SAVED_MODELS_DIR']
    for name, filename in [('ANN', 'ann_model.h5'), ('RNN', 'rnn_model.h5'), ('LSTM', 'lstm_model.h5')]:
        path = os.path.join(models_dir, filename)
        logger.info(f"Loading {name} model from {path}")
        _models[name] = tf.keras.models.load_model(path)
    logger.info("All models loaded successfully.")


def get_model(name):
    if not _models:
        load_models()
    return _models.get(name.upper())


def evaluate_all_models():
    global _results_cache
    if _results_cache:
        return _results_cache

    if not _models:
        load_models()

    pipeline = get_pipeline()
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']

    for name, model in _models.items():
        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        _results_cache[name] = {
            'accuracy':             round(report['accuracy'] * 100),
            'macro_avg_precision':  round(report['macro avg']['precision'], 2),
            'macro_avg_recall':     round(report['macro avg']['recall'], 2),
            'macro_avg_f1':         round(report['macro avg']['f1-score'], 2),
            'weighted_avg_precision': round(report['weighted avg']['precision'], 2),
            'weighted_avg_recall':  round(report['weighted avg']['recall'], 2),
            'weighted_avg_f1':      round(report['weighted avg']['f1-score'], 2),
        }
        logger.info(f"{name} accuracy: {_results_cache[name]['accuracy']}%")

    return _results_cache


def predict(algorithm, input_values):
    model = get_model(algorithm)
    if model is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    input_array = np.array([input_values])
    raw_output = model.predict(input_array)
    class_idx = np.argmax(raw_output, axis=1)[0]
    return ATTACK_CAT_MAPPING.get(class_idx, 'Unknown')


def generate_analysis_message(results):
    accs = {k: v['accuracy'] for k, v in results.items()}
    best = max(accs, key=accs.get)
    msg = f"Based on the analysis, {best} outperforms the other models with an accuracy of {accs[best]}%."
    if best == 'LSTM':
        msg += (f" LSTM significantly surpasses ANN ({accs['ANN']}%) and RNN ({accs['RNN']}%)."
                " Its gating mechanism handles temporal traffic patterns that ANN treats independently.")
    elif best == 'ANN':
        msg += f" ANN performs better than RNN ({accs['RNN']}%) while being the lightest model at 66KB."
    else:
        msg += f" RNN outperforms ANN ({accs['ANN']}%) by capturing sequential packet patterns."
    return msg