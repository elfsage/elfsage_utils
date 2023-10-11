import json
from pathlib import Path


def load_json(file_path, create=False):
    if not Path(file_path).is_file() and create:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
