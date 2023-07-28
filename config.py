import os
import torch
from utils import Logger
from utils.data import train_transforms, valid_transforms, infer_transforms
from metric import dice_coefficient, intersection_over_union, hausdorff_distance_2d
from losses import DiceCELoss


class Config:
    # 实验的基础配置
    LAB_ID = 'unet+efficientnet-b7'
    RUNNING_DIR = os.path.join('run', LAB_ID)
    LOG_PATH = os.path.join(RUNNING_DIR, 'log')
    MODEL_PATH = os.path.join(RUNNING_DIR, 'models')
    K_FOLD = 5

    # GPU硬件环境
    DEVICE = 'cuda:0'
    DEVICE_IDS = [0, 1]

    # SMP `create_model` 参数
    SMP_ARCH = 'unet'
    SMP_ENCODER_NAME = 'efficientnet-b7'
    SMP_ENCODER_WEIGHTS = 'imagenet'
    SMP_IN_CHANNELS = 3
    SMP_CLASSES = 1
    SMP_KWARGS = {}
    MODEL_CHECKPOINT = None

    # 训练超参数
    EPOCHS = 100
    START_EPOCH = 0
    LEARNING_RATE = 1e-3
    IS_SCHEDULER = True
    SAVE_INTERVAL = 10
    VALID_INTERVAL = 1

    # 数据集参数
    IMG_SIZE = 256
    DATA_DIR = './data/train/'
    BATCH_SIZE = 16
    NUM_WORKER = 0
    TRAIN_TRANSFORMS = train_transforms(IMG_SIZE)
    VALID_TRANSFORMS = valid_transforms()

    # 损失函数
    LOSS_FN = DiceCELoss()

    # 优化方法
    OPTIMIZER = torch.optim.Adam
    OPTIM_KWARGS = {
        'weight_decay': 1e-3
    }

    # 评估指标
    THRESHOLD = 0.7
    EVALUATION_METRIC = {
        'dice': dice_coefficient,
        'iou': intersection_over_union,
        # 'haus dist': hausdorff_distance_2d,
    }
    METRIC_KEYS = EVALUATION_METRIC.keys()

    # 推理
    INFER_MODEL = os.path.join(MODEL_PATH, f'{SMP_ARCH}+{SMP_ENCODER_NAME}+best.pth')
    INFERENCE_DATA_DIR = './data/test'
    INFER_SAVE_PATH = os.path.join(RUNNING_DIR, 'infer')
    INFER_TRANSFORMS = infer_transforms()

    def __init__(self):
        if not os.path.exists(self.RUNNING_DIR):
            os.makedirs(self.RUNNING_DIR)
        self.logger = Logger(self.LOG_PATH)
        self.logger.info(self)

    def change_running_dir(self, new_dir: str):
        self.RUNNING_DIR = new_dir
        self.MODEL_PATH = os.path.join(self.RUNNING_DIR, 'models')
        if not os.path.exists(self.RUNNING_DIR):
            os.makedirs(self.RUNNING_DIR)

    def __str__(self):
        # todo: 此处可修改为输出config.py文件
        with open(__file__, 'r', encoding='utf8') as fp:
            msg = fp.read()
        # msg = f'\nLAB {self.LAB_ID}\n' \
        #       f'MODEL: {self.SMP_ARCH} with encoder {self.SMP_ENCODER_NAME}\n' \
        #       f'{"USING CHECKPOINT {}".format(self.MODEL_CHECKPOINT) if self.MODEL_CHECKPOINT is not None else ""}\n' \
        #       f'# GPU硬件环境\n' \
        #       f'DEVICE = {self.DEVICE}\n' \
        #       f'DEVICE_IDS = {self.DEVICE_IDS}\n' \
        #       f'# 训练超参数\n' \
        #       f'EPOCHS = {self.EPOCHS}\n' \
        #       f'LEARNING_RATE = {self.LEARNING_RATE}\n' \
        #       f'IS_SCHEDULER = {self.IS_SCHEDULER}\n' \
        #       f'SAVE_INTERVAL = {self.SAVE_INTERVAL}\n' \
        #       f'VALID_INTERVAL = {self.VALID_INTERVAL}\n' \
        #       f'# 数据集参数\n' \
        #       f'IMG_SIZE = {self.IMG_SIZE}\n' \
        #       f'BATCH_SIZE = {self.BATCH_SIZE}\n' \
        #       f'NUM_WORKER = {self.NUM_WORKER}\n'
        return '\n --- config --- \n\n' + msg
