import json
from pathlib import Path
from typing import Union


def load_json(
        file_path: Union[str, Path],
        create: bool = False,
        default: Union[dict, list] = None
) -> Union[dict, list]:
    assert not create or default is not None, 'If create is true, default value must be provided'

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.is_file():
        if create:
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(default, f, indent=2)

        if default is not None:
            return default

    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data: Union[dict, list], file_path: Union[str, Path]):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    with file_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
