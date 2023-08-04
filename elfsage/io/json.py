import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
