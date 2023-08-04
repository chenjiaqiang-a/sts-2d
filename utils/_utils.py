import os
import pickle
import zipfile

__all__ = ['save_result', 'load_pickle', 'zip_dir']


def save_result(data: dict, filename='result.pkl'):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def load_pickle(filename: str):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def zip_dir(dir_path, output_path=None):
    root = os.path.split(os.path.abspath(dir_path))[-1]
    if output_path is None:
        output_path = root + '.zip'
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path, dir_names, filenames in os.walk(dir_path):
            fpath = path.replace(dir_path, root)
            for filename in filenames:
                zipf.write(os.path.join(path, filename), os.path.join(fpath, filename))
