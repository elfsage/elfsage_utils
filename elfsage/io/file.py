import pathlib
from os.path import isfile


def unique_filename(file_path, suffix_length=3):
    path = pathlib.Path(file_path)
    file_name_pattern = '{file_dir}/{file_name}_{file_name_suffix:0'+str(suffix_length)+'d}{file_ext}'

    while True:
        file_name_suffix = 1

        file_path = file_name_pattern.format(
            file_dir=path.parent.resolve(),
            file_name=path.stem,
            file_name_suffix=file_name_suffix,
            file_ext=path.suffix
        )

        if not isfile(file_path):
            break

    return file_path


# print(unique_filename('setup.py'))
