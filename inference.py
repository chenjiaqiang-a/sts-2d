import os
import cv2
import torch
import numpy as np

import model as m
from config import Config
from utils import zip_dir


class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        """生成一系列滑动窗口，用于分割图片
        """
        if h < ch or w < cw:
            raise ValueError(f'`h` must greater than `ch` and `w` must greater than `cw`,'
                             f'but got `h` = {h} `ch` = {ch} and `w` = {w} `cw` = {cw}')
        self.h, self.w = h, w
        self.ch, self.cw = ch, cw
        self.si, self.sj = si, sj
        self._i, self._j = 0, 0

    def __next__(self):
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i + self.ch, self.h)
        right = min(self._j + self.cw, self.w)
        top = max(0, bottom - self.ch)
        left = max(0, right - self.cw)

        if self._j >= self.w - self.cw:
            if self._i >= self.h - self.ch:
                self._i = self.h + 1
            self._next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def _next_row(self):
        self._i += self.si
        self._j = 0

    def __iter__(self):
        return self


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Inference:
    def __init__(self, cfg: Config):
        self.result_path = cfg.INFER_SAVE_PATH
        self.logger = cfg.logger

        self.device = torch.device(cfg.DEVICE)

        self.model = m.create_model(
            arch=cfg.MODEL_ARCH,
            encoder_name=cfg.MODEL_ENCODER_NAME,
            encoder_weights=cfg.MODEL_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            **cfg.MODEL_KWARGS,
        )
        self.model.to(self.device)
        self.load_model(cfg.INFER_MODEL)

        self.transform = cfg.INFER_TRANSFORMS
        self.window_size = cfg.INFER_WINDOW_SIZE
        self.stride = cfg.INFER_WINDOW_STRIDE
        self.threshold = cfg.THRESHOLD

    def load_model(self, file_path: str):
        # from collections import OrderedDict
        # new_dict = OrderedDict()
        # state_dict = torch.load(file_path)
        # for param in state_dict:
        #     new_dict[param.replace('module.model.', '')] = state_dict[param]
        #
        # self.model.load_state_dict(new_dict)
        # self.logger.info(f'Successfully load model {file_path}')
        self.model.load_state_dict(torch.load(file_path))
        self.logger.info(f'Successfully load model {file_path}')

    def split_image(self, img):
        h, w = img.shape[0], img.shape[1]
        win_gen = WindowGenerator(
            h, w, self.window_size, self.window_size, self.stride, self.stride)
        patches = []
        windows = []
        for h_slice, w_slice in win_gen:
            patches.append(img[h_slice, w_slice])
            windows.append([h_slice, w_slice])
        return patches, windows

    @torch.no_grad()
    def infer(self, img, filename):
        self.logger.info(f'Infer image {filename}')
        self.model.eval()

        h, w = img.shape[0], img.shape[1]
        prob_map = np.zeros((h, w), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.float64)
        patches, windows = self.split_image(img)
        patches = torch.stack(
            [self.transform(image=patch)['image'] for patch in patches], dim=0)

        patches = patches.to(self.device)
        masks = self.model(patches)
        masks = masks.cpu().numpy()

        for mask, window in zip(masks, windows):
            prob_map[window[0], window[1]] += mask.squeeze()
            cnt[window[0], window[1]] += 1

        prob_map /= cnt
        prob_map = _sigmoid(prob_map)
        full_mask = ((prob_map > self.threshold) * 255).astype(np.uint8)
        full_mask = cv2.resize(full_mask, (w, h))

        cv2.imwrite(os.path.join(self.result_path, filename), full_mask)


def generate_inference(cfg: Config):
    if not os.path.exists(cfg.INFER_SAVE_PATH):
        os.makedirs(cfg.INFER_SAVE_PATH)

    infer = Inference(cfg)
    image_dir = os.path.join(cfg.INFERENCE_DATA_DIR, 'image')
    filenames = [filename for filename in os.listdir(image_dir)
                 if filename[-3:] == 'png']

    for filename in filenames:
        img = cv2.imread(os.path.join(image_dir, filename))

        infer.infer(img, filename)

    zip_dir(cfg.INFER_SAVE_PATH, os.path.join(cfg.RUNNING_DIR, 'infers.zip'))


if __name__ == '__main__':
    config = Config()
    generate_inference(config)
