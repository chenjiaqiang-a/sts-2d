import logging
import time
import os


class Logger:
    def __init__(self, log_path="./", verbose=True):
        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # setting level
        formatter = logging.Formatter("[%(asctime)s] %(message)s")

        # create file handler
        start_time = time.strftime('%y-%m-%d-%H%M', time.localtime(time.time()))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, start_time + '.log')
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if verbose:
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)
