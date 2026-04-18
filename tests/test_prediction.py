import pytest
from app.ml.validator import validate_prediction_input


VALID_INPUT = {
    'algorithm': 'LSTM',
    'sbytes': '1000', 'dbytes': '500', 'rate': '50.5',
    'sload': '1200.0', 'dload': '800.0', 'sinpkt': '20',
    'sjit': '0.5', 'stcpb': '12345', 'dtcpb': '67890',
    'response_body_len': '2048'
}


def test_valid_input_passes():
    values, algorithm, errors = validate_prediction_input(VALID_INPUT)
    assert errors == []
    assert len(values) == 10
    assert algorithm == 'LSTM'


def test_negative_value_rejected():
    bad = dict(VALID_INPUT, sbytes='-100')
    values, algorithm, errors = validate_prediction_input(bad)
    assert values is None
    assert any('Source Bytes' in e for e in errors)


def test_non_numeric_rejected():
    bad = dict(VALID_INPUT, rate='abc')
    values, algorithm, errors = validate_prediction_input(bad)
    assert values is None
    assert any('Rate' in e for e in errors)


def test_invalid_algorithm_rejected():
    bad = dict(VALID_INPUT, algorithm='CNN')
    values, algorithm, errors = validate_prediction_input(bad)
    assert any('algorithm' in e.lower() for e in errors)


def test_missing_field_rejected():
    bad = {k: v for k, v in VALID_INPUT.items() if k != 'sjit'}
    values, algorithm, errors = validate_prediction_input(bad)
    assert values is None
    assert any('Jitter' in e for e in errors)