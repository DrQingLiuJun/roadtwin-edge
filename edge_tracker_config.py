"""Validated ByteTrack configuration shared by the portal and edge services."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CONFIG = Path(__file__).with_name('edge_tracker.json')
FIELDS = {
    'CONF_THRESHOLD': (.25, .01, 1.0), 'HIGH_THRESH': (.30, .01, 1.0), 'LOW_THRESH': (.10, 0.0, 1.0),
    'MATCH_THRESH': (.85, .01, 1.0), 'GATING_THRESHOLD': (9.4877, .01, 100.0), 'LOW_MATCH_IOU': (.50, .01, 1.0),
    'BIRTH_SCORE': (.50, .01, 1.0), 'MAX_LOST_FRAMES': (30, 1, 300), 'DUPLICATE_IOU': (.85, .01, 1.0),
    'CLASS_CONFIRM_FRAMES': (5, 2, 30), 'CLASS_VOTE_DECAY': (.80, .1, .99), 'CLASS_VOTE_RATIO': (1.15, 1.01, 5.0),
    'CLASS_MATCH_MAX_GAP': (3, 1, 30), 'CLASS_MATCH_MIN_SCORE': (.45, .01, 1.0), 'CLASS_MATCH_MIN_IOU': (.65, .01, 1.0),
}

def validate(values):
    if not isinstance(values, dict) or values.keys() != FIELDS.keys():
        raise ValueError('ByteTrack parameter fields do not match')
    result = {}
    for name, (default, low, high) in FIELDS.items():
        value = values[name]
        if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
            raise ValueError(f'{name} must be within {low}..{high}')
        if type(default) is int and int(value) != value:
            raise ValueError(f'{name} must be an integer')
        result[name] = type(default)(value)
    if not result['LOW_THRESH'] < result['HIGH_THRESH'] <= result['BIRTH_SCORE'] or result['CONF_THRESHOLD'] > result['HIGH_THRESH']:
        raise ValueError('Require low < high <= birth and detector <= high')
    return result

def load():
    try:
        values = json.loads(CONFIG.read_text())
        legacy = tuple(FIELDS)[:9]
        if isinstance(values, dict) and values.keys() == set(legacy):
            values.update({key: FIELDS[key][0] for key in tuple(FIELDS)[9:]})
        return validate(values)
    except FileNotFoundError:
        return {key: field[0] for key, field in FIELDS.items()}

def revision(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]

def save(values):
    with tempfile.NamedTemporaryFile(mode='w', dir=CONFIG.parent, prefix='.edge-tracker-', delete=False) as file:
        path = Path(file.name)
        try:
            json.dump(values, file, indent=2)
            file.write('\n')
            file.close()
            os.replace(path, CONFIG)
        finally:
            path.unlink(missing_ok=True)

def configure_worker():
    values = load()
    os.environ.update({f'RSU_BT_{key}': str(value) for key, value in values.items()})
    return revision(values)
