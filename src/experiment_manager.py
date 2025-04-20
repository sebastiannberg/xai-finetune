import os
from datetime import datetime
from pathlib import Path
import logging
import time
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from timm.models.layers import trunc_normal_
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from tqdm import tqdm

import models_vit as models_vit
from models_vit import PatchEmbed_new
from urban_dataset import UrbanDataset
from plots import Plots
from methods import baseline_one_epoch, ifi_one_epoch


class ExperimentManager:

    def __init__(self, args):
        self.args = args

        self.project_root = Path(__file__).parent.parent.absolute()

        self.pretrained_path = os.path.join(self.project_root, "ckpt", "pretrained.pth")
        self.dataset_path = os.path.join(self.project_root, "data", "UrbanSound8K")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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
            format="%(asctime)s [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "main.log"), mode="a")
            ]
        )
        self.logger = logging.getLogger()
        self.logger.info("Logging is configured")
        self.logger.info(f"Arguments: {self.args}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Pretrained path: {self.pretrained_path}")
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")
        self.logger.info(f"Image directory: {self.img_dir}")
        self.logger.info("All directories created successfully")

        self.plotter = Plots(self.img_dir)
        self.plot_epochs = [1, 5, 10, 20, 30, 40, 50, 60]
        self.logger.info("Plots class initialized")

        self.watched_filenames = {"197318-6-7-0.wav", "138015-3-0-1.wav", "26270-9-0-30.wav", "7389-1-0-1.wav", "16692-5-0-3.wav", "20688-2-0-0.wav", "14110-4-0-2.wav", "22883-7-7-0.wav", "24347-8-0-0.wav", "47160-0-0-4.wav"}
        self.logger.info(f"Watching files: {self.watched_filenames}, this will be the basis for plots")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Running on device: {self.device}")

        self.epoch_metrics = []
        self.snr_values = {filename: [] for filename in self.watched_filenames}
        self.class_attention_grads = None

        self.logger.info("Experiment manager initialized")

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fix_seeds(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def set_cuda_settings(self, deterministic=False, benchmark=True):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

    def setup_dataset(self):
        dataset_train = UrbanDataset(
            root=self.dataset_path,
            fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            mixup_prob=0.5,
            roll_mag_aug=True,
            target_length=self.args.target_length,
            freqm=48,
            timem=192,
            num_classes=self.args.num_classes
        )
        dataset_val = UrbanDataset(
            root=self.dataset_path,
            fold=[10],
            mixup_prob=0.0,
            roll_mag_aug=False,
            target_length=self.args.target_length,
            freqm=0,
            timem=0,
            num_classes=self.args.num_classes
        )
        self.data_loader_train = DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.data_loader_val = DataLoader(
            dataset_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        if self.args.mode == "ifi":
            dataset_interpret = UrbanDataset(
                root=self.dataset_path,
                fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                mixup_prob=0.0,
                roll_mag_aug=False,
                target_length=self.args.target_length,
                freqm=0,
                timem=0,
                num_classes=self.args.num_classes
            )
            # Partition data to samples for each label
            class_to_indices = defaultdict(list)
            for idx in range(len(dataset_interpret)):
                _, data_y, _ = dataset_interpret[idx]
                class_idx = data_y.argmax().item()
                class_to_indices[class_idx].append(idx)
            class_loaders = []
            for class_idx in range(self.args.num_classes):
                indices = class_to_indices[class_idx]
                subset = Subset(dataset_interpret, indices)
                loader = DataLoader(
                    subset,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    drop_last=False
                )
                class_loaders.append(loader)
            self.class_loaders = class_loaders

    def setup_model(self):
        model = models_vit.__dict__["vit_base_patch16"](
            num_classes=self.args.num_classes,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=True,
            use_custom_patch=False
        )
        model.patch_embed = PatchEmbed_new(
            img_size=(self.args.target_length, 128),
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=768,
            stride=16
        )
        model.pos_embed = nn.Parameter(
            torch.zeros(1, model.patch_embed.num_patches + 1, 768),
            requires_grad=False
        )

        checkpoint = torch.load(self.pretrained_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()

        # Check if positional embeddings need to be interpolated
        if "pos_embed" in checkpoint_model:
            if checkpoint_model["pos_embed"].shape != model.pos_embed.shape:
                self.logger.info(f"Interpolating positional embeddings from {checkpoint_model['pos_embed'].shape} to {torch.Size([1, 257, 768])}")
                pos_embed_checkpoint = checkpoint_model["pos_embed"]  # [1, old_num_tokens, embed_dim]
                cls_token = pos_embed_checkpoint[:, :1, :]            # [1, 1, embed_dim]
                pos_tokens = pos_embed_checkpoint[:, 1:, :]           # [1, old_num_tokens-1, embed_dim]

                # Determine the original grid shape
                num_tokens_pretrained = pos_tokens.shape[1]
                if num_tokens_pretrained == 512:
                    grid_shape_pretrained = (8, 64)  # Known grid shape from pretraining
                else:
                    grid_size = int(np.sqrt(num_tokens_pretrained))
                    grid_shape_pretrained = (grid_size, grid_size)

                # Reshape from (1, num_tokens, embed_dim) -> (1, grid_height, grid_width, embed_dim)
                pos_tokens = pos_tokens.reshape(1, grid_shape_pretrained[0], grid_shape_pretrained[1], -1)
                # Permute to (1, embed_dim, grid_height, grid_width) for interpolation
                pos_tokens = pos_tokens.permute(0, 3, 1, 2)

                # New grid size from your custom patch embedding (e.g., (32, 8) in your case)
                new_grid_size = model.patch_embed.patch_hw

                # Interpolate using bilinear interpolation
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=new_grid_size,
                    mode="bilinear",
                    align_corners=False
                )
                # Permute back and reshape to (1, new_num_tokens, embed_dim)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_embed_checkpoint.shape[-1])

                # Concatenate the class token back
                new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed

        # Remove the classification head weights if they don"t match the current model"s output shape
        # This prevents shape mismatch issues when fine-tuning on a different number of classes
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                self.logger.info(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
                del checkpoint_model[k]

        # Load the remaining pre-trained weights into the model
        # strict=False allows partial loading (ignores missing keys like the removed head)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        self.logger.info(msg)
        # Reinitialize the classification head with small random values
        trunc_normal_(model.head.weight, std=2e-5)
        self.model = model.to(self.device)

    def setup_optimizer(self):
        self. optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.args.lr,
            weight_decay = self.args.weight_decay
        )

    def setup_criterion(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def setup_lr_scheduler(self):
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-5
        )

    def setup_scaler(self):
        self.scaler = GradScaler()

    def train_one_epoch(self, epoch):
        if self.args.mode == "baseline":
            train_loss = baseline_one_epoch(self, epoch)
        elif self.args.mode == "ifi":
            train_loss = ifi_one_epoch(self, epoch)
        return train_loss

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for fbank, label, _ in tqdm(self.data_loader_val, desc="Validation", leave=False, position=1):
                fbank = fbank.to(self.device)
                label = label.to(self.device)

                logits = self.model(fbank)
                loss = self.criterion(logits, label)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                true_classes = torch.argmax(label, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(true_classes.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        val_loss = total_loss / len(self.data_loader_val)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        return val_loss, val_accuracy, val_f1

    def epoch_summary(self, epoch, train_loss, val_loss, val_accuracy, val_f1):
        self.logger.info("-"*40)
        self.logger.info(f"Epoch [{epoch+1}/{self.args.epochs}]")
        self.logger.info(f"  Train Loss:    {train_loss:.4f}")
        self.logger.info(f"  Val Loss:      {val_loss:.4f}")
        self.logger.info(f"  Val Accuracy:  {val_accuracy:.4f}")
        self.logger.info(f"  Val F1:        {val_f1:.4f}")

    def step_lr_scheduler(self):
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()
        self.logger.info(f"Current learning rate after step: {current_lr[0]:.6f}")

    def save_model_ckpt(self, epoch, val_loss, val_accuracy, val_f1):
        if (epoch + 1) % self.args.save_frequency == 0:
            model_path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'args': vars(self.args)
            }, model_path)
            self.logger.info(f'Model saved at epoch {epoch+1}: {model_path}')

    def train_epochs(self):
        start_time = time.time()
        for epoch in tqdm(range(self.args.epochs), desc="Training Progress", leave=True, position=0):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_accuracy, val_f1 = self.validate()
            self.epoch_metrics.append((train_loss, val_loss, val_accuracy, val_f1))
            self.epoch_summary(epoch, train_loss, val_loss, val_accuracy, val_f1)
            self.step_lr_scheduler()
            self.save_model_ckpt(epoch, val_loss, val_accuracy, val_f1)
        self.final_train_loss = train_loss
        self.final_val_loss = val_loss
        self.final_val_accuracy = val_accuracy
        self.final_val_f1 = val_f1
        self.best_val_accuracy = max(entry[2] for entry in self.epoch_metrics)
        self.best_val_f1 = max(entry[3] for entry in self.epoch_metrics)
        self.total_training_time = time.time() - start_time
        self.logger.info(f'Total training time: {self.total_training_time / 60:.2f} minutes')

    def save_experiment_summary(self):
        summary = {
            "args": vars(self.args),
            "finished_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_training_time_minutes": self.total_training_time / 60,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "final_val_accuracy": self.final_val_accuracy,
            "final_val_f1": self.final_val_f1,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_f1": self.best_val_f1,
        }
        summary_path = os.path.join(self.results_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Experiment summary saved to {summary_path}")

    def post_experiment_plots(self):
        train_loss_list = [entry[0] for entry in self.epoch_metrics]
        val_loss_list = [entry[1] for entry in self.epoch_metrics]
        acc_list = [entry[2] for entry in self.epoch_metrics]
        f1_list = [entry[3] for entry in self.epoch_metrics]
        self.plotter.plot_loss_curve(train_loss_list, val_loss_list)
        self.plotter.plot_accuracy_f1_curve(acc_list, f1_list)

        print(self.snr_values)
        for filename, snr_values in self.snr_values.items():
            print(len(snr_values))
            self.plotter.plot_snr(snr_values, filename)

    def run_experiment(self):
        self.logger.info("### Running experiment ###")

        self.fix_seeds()
        self.logger.info("Fixing the seeds for reproducability")

        self.set_cuda_settings()
        self.logger.info("Setting cuda settings for cudnn.deterministic and cudnn.benchmark")

        self.setup_dataset()

        self.setup_model()

        self.setup_optimizer()

        self.setup_criterion()

        self.setup_lr_scheduler()

        self.setup_scaler()

        self.train_epochs()

        self.post_experiment_plots()

        self.save_experiment_summary()

        self.logger.info(f"Experiment finished")
