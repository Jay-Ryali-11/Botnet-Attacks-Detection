from app.ml.validator import FEATURE_CONFIG, VALID_ALGORITHMS


def test_feature_config_has_ten_fields():
    assert len(FEATURE_CONFIG) == 10


def test_all_feature_mins_are_zero():
    for cfg in FEATURE_CONFIG:
        assert cfg['min'] == 0, f"{cfg['field']} min should be 0"


def test_valid_algorithms_set():
    assert 'ANN' in VALID_ALGORITHMS
    assert 'RNN' in VALID_ALGORITHMS
    assert 'LSTM' in VALID_ALGORITHMS
    assert len(VALID_ALGORITHMS) == 3


def test_all_feature_fields_present():
    expected = {'sbytes', 'dbytes', 'rate', 'sload', 'dload',
                'sinpkt', 'sjit', 'stcpb', 'dtcpb', 'response_body_len'}
    actual = {cfg['field'] for cfg in FEATURE_CONFIG}
    assert actual == expected