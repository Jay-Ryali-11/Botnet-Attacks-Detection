FEATURE_CONFIG = [
    {'field': 'sbytes',            'label': 'Source Bytes',           'min': 0, 'max': 1e12},
    {'field': 'dbytes',            'label': 'Destination Bytes',      'min': 0, 'max': 1e12},
    {'field': 'rate',              'label': 'Rate',                   'min': 0, 'max': 1e8},
    {'field': 'sload',             'label': 'Source Load',            'min': 0, 'max': 1e12},
    {'field': 'dload',             'label': 'Destination Load',       'min': 0, 'max': 1e12},
    {'field': 'sinpkt',            'label': 'Source In-Packets',      'min': 0, 'max': 1e9},
    {'field': 'sjit',              'label': 'Source Jitter',          'min': 0, 'max': 1e7},
    {'field': 'stcpb',             'label': 'Source TCP Bytes',       'min': 0, 'max': 2**32},
    {'field': 'dtcpb',             'label': 'Destination TCP Bytes',  'min': 0, 'max': 2**32},
    {'field': 'response_body_len', 'label': 'Response Body Length',   'min': 0, 'max': 1e10},
]

VALID_ALGORITHMS = {'ANN', 'RNN', 'LSTM'}

def validate_prediction_input(form_data):
    errors = []
    values = []

    algorithm = form_data.get('algorithm', '').strip().upper()
    if algorithm not in VALID_ALGORITHMS:
        errors.append(f"Invalid algorithm '{algorithm}'. Choose ANN, RNN, or LSTM.")

    for cfg in FEATURE_CONFIG:
        field = cfg['field']
        label = cfg['label']
        raw = form_data.get(field, '').strip()

        if not raw:
            errors.append(f"{label} is required.")
            continue

        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number. Got: '{raw}'")
            continue

        if val < cfg['min'] or val > cfg['max']:
            errors.append(
                f"{label} must be between {cfg['min']:.0f} and {cfg['max']:.2e}."
            )
            continue

        values.append(val)

    if errors:
        return None, algorithm, errors
    return values, algorithm, []