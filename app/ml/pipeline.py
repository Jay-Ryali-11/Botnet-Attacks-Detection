import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from flask import current_app
from app.utils.logger import get_logger

logger = get_logger(__name__)

_pipeline_cache = None


def get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache
    _pipeline_cache = build_pipeline()
    return _pipeline_cache


def build_pipeline():
    data_path = os.path.join(current_app.config['DATA_DIR'], 'UNSW_NB15.csv')
    logger.info(f"Loading dataset from {data_path}")

    data = pd.read_csv(data_path)
    data_cleaned = data.drop(columns=['id', 'label'])

    label_encoder = LabelEncoder()
    data_cleaned['attack_cat_encoded'] = label_encoder.fit_transform(data_cleaned['attack_cat'])
    data_cleaned = data_cleaned.drop(columns=['attack_cat'])

    for col in ['proto', 'service', 'state']:
        data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])

    X = data_cleaned.drop(columns=['attack_cat_encoded'])
    y = data_cleaned['attack_cat_encoded']

    selector = SelectKBest(score_func=chi2, k=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    logger.info(f"Selected features: {list(selected_features)}")

    X_selected = data_cleaned[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': list(selected_features),
        'label_encoder': label_encoder,
    }