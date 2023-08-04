import os
import torch
from utils import Logger
from utils.data import train_transforms, valid_transforms, infer_transforms
import metric


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

    # `create_model` 参数
    MODEL_ARCH = 'unet++'
    MODEL_ENCODER_NAME = 'efficientnet-b7'
    MODEL_ENCODER_WEIGHTS = 'imagenet'
    MODEL_IN_CHANNELS = 3
    MODEL_CLASSES = 1
    MODEL_KWARGS = {}
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
    LOSS_FN = 'dice_ce_loss'
    LOSS_FN_KWARGS = {}

    # 优化方法
    OPTIMIZER = torch.optim.Adam
    OPTIM_KWARGS = {
        'weight_decay': 1e-3
    }

    # 评估指标
    THRESHOLD = 0.5
    EVALUATION_METRIC = {
        'dice': metric.dice_coefficient,
        'iou': metric.intersection_over_union,
        # 'haus dist': hausdorff_distance_2d,
    }
    METRIC_KEYS = EVALUATION_METRIC.keys()

    # 推理
    INFER_MODEL = os.path.join(MODEL_PATH, f'{MODEL_ARCH}+{MODEL_ENCODER_NAME}+best.pth')
    INFERENCE_DATA_DIR = './data/test'
    INFER_SAVE_PATH = os.path.join(RUNNING_DIR, 'infers')
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
        with open(__file__, 'r', encoding='utf8') as fp:
            msg = fp.read()

        return '\n --- config --- \n\n' + msg
