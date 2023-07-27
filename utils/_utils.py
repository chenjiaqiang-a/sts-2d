import pickle
import zipfile

__all__ = ['save_result', 'load_pickle', 'zip_files']


def save_result(data: dict, filename='result.pkl'):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def load_pickle(filename: str):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def zip_files(file_paths, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file)
