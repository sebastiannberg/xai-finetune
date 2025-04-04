import os
import argparse
from datetime import datetime
from pathlib import Path
import logging
import torch

# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader, Subset
# from timm.models.layers import to_2tuple, trunc_normal_
# from sklearn.metrics import accuracy_score, f1_score
# from tqdm import tqdm
# import logging
# import random
# from collections import defaultdict
# import matplotlib.pyplot as plt


class ExperimentManager:

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.absolute()

        self.pretrained_path = os.path.join(self.project_root, "ckpt", "pretrained.pth")
        self.dataset_path = os.path.join(self.project_root, "data", "UrbanSound8K")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.results_dir = os.path.join(self.project_root, "results", self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

        self.log_dir = os.path.join(self.results_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(self.results_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.img_dir = os.path.join(self.results_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s'",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "experiment_manager.log"), mode='a')
            ]
        )
        self.logger = logging.getLogger()
        self.logger.info("Logging is configured")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Pretrained path: {self.pretrained_path}")
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")
        self.logger.info(f"Image directory: {self.img_dir}")
        self.logger.info("All directories created successfully")

        self.logger.info("Experiment manager initialized")

    def get_args(self):
        parser = argparse.ArgumentParser(description='Finetune on UrbanSound8K dataset')
        parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
        parser.add_argument('--target_length', type=int, default=512, help='Number of time frames for fbank')
        parser.add_argument('--checkpoint', type=str, default='pretrained.pth', help='Filename for model checkpoint to load before fine-tuning')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
        parser.add_argument('--num_workers', type=int, default=10, help='Number of worker threads for data loading')
        parser.add_argument('--seed', type=int, default=0, help='To control the random seed for reproducibility')
        return parser.parse_args()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

