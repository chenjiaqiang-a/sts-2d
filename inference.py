import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image

from config import Config
from utils import zip_files
from utils.data import STS2DDataset


class Inference:
    def __init__(self, cfg: Config):
        self.result_path = cfg.INFER_SAVE_PATH
        self.logger = cfg.logger

        self.device = torch.device(cfg.DEVICE)

        self.model = smp.create_model(
            arch=cfg.SMP_ARCH,
            encoder_name=cfg.SMP_ENCODER_NAME,
            encoder_weights=cfg.SMP_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            **cfg.SMP_KWARGS,
        )
        self.model.to(self.device)
        self.load_model(cfg.INFER_MODEL)

        self.threshold = cfg.THRESHOLD

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))
        self.logger.info(f'Successfully load model {file_path}')

    @torch.no_grad()
    def infer(self, inputs, filename):
        self.logger.info(f'Infer image {filename}')
        self.model.eval()

        input_data = inputs['image'].unsqueeze(0).to(self.device)
        out: torch.Tensor = self.model(input_data)
        out = torch.where(out >= self.threshold, torch.tensor(255, dtype=torch.float).to(self.device), out)
        out = torch.where(out < self.threshold, torch.tensor(0, dtype=torch.float).to(self.device), out)
        out: np.ndarray = out.squeeze().cpu().numpy()

        img = Image.fromarray(out.astype(np.uint8))
        img = img.convert('1')
        img.save(os.path.join(self.result_path, filename))


def generate_inference(cfg: Config):
    if not os.path.exists(cfg.INFER_SAVE_PATH):
        os.makedirs(cfg.INFER_SAVE_PATH)

    infer = Inference(cfg)
    image_dir = os.path.join(cfg.INFERENCE_DATA_DIR, 'image')
    filenames = [filename for filename in os.listdir(image_dir)
                 if filename[-3:] == 'png']
    dataset = STS2DDataset(cfg.INFERENCE_DATA_DIR, filenames, tag='infer', transform=cfg.INFER_TRANSFORMS)

    for i, inputs in enumerate(dataset):
        infer.infer(inputs, dataset.filenames[i])

    file_paths = [os.path.join(cfg.INFER_SAVE_PATH, i)
                  for i in os.listdir(cfg.INFER_SAVE_PATH) if i[-3:] == 'png']
    zip_files(file_paths, os.path.join(cfg.RUNNING_DIR, 'infer.zip'))


if __name__ == '__main__':
    config = Config()
    generate_inference(config)
