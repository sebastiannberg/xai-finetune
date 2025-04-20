import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import rcParams
import numpy as np


class Plots:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        # rcParams.update({
        #     'font.size': 30,        # Base font size
        #     'axes.labelsize': 26,   # Axis label size
        #     'xtick.labelsize': 20,  # X-axis tick label size
        #     'ytick.labelsize': 20,  # Y-axis tick label size
        # })

    def plot_loss_curve(self, train_loss_list, val_loss_list):
        epochs = range(len(train_loss_list))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss_list, label="Training Loss", color="blue", linewidth=2)
        plt.plot(epochs, val_loss_list, label="Validation Loss", color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.img_dir, "loss_curve.png"))
        plt.close()

    def plot_accuracy_f1_curve(self, acc_list, f1_list):
        epochs = range(len(acc_list))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, acc_list, label="Accuracy", color="blue", linewidth=2)
        plt.plot(epochs, f1_list, label="F1", color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.img_dir, "accuracy_f1_curve.png"))
        plt.close()

    def plot_spectrogram(self, spectrogram, filename):
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(
            spectrogram,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            aspect="auto"
        )

        ax.set_title(f"{filename}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax)

        filename_wo_ext = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(self.img_dir, f"{filename_wo_ext}_spectrogram.png"), bbox_inches="tight")
        plt.close(fig)

    def plot_attention_heatmap(self, attention, filename, epoch, mode="attention"):
        # attention shape: (block, head, seq, seq)
        filename_wo_ext = os.path.splitext(filename)[0]
        epoch_dir = os.path.join(self.img_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for i, block_attention in enumerate(attention):
            avg_attention = block_attention.mean(axis=0)
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(
                avg_attention,
                cmap="hot",
                aspect="equal"
            )
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            # ax.tick_params(bottom=False)
            ax.set_xlabel("Key Index")
            ax.set_ylabel("Query Index")
            ax.set_title(f"{filename} - {mode.capitalize()} - Block {i}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(im, cax=cax)

            fig.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_{i}_{mode}_heatmap.png"), bbox_inches="tight")
            plt.close(fig)

        avg_blocks_attention = attention.mean(axis=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(
            avg_blocks_attention,
            cmap="hot",
            aspect="equal"
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        # ax.tick_params(bottom=False)
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")
        ax.set_title(f"{filename} - {mode.capitalize()} - All Blocks")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax)

        fig.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_all_{mode}_heatmap.png"), bbox_inches="tight")
        plt.close(fig)

    def plot_avg_received_attention(self, attention, filename, epoch, mode="attention"):
        filename_wo_ext = os.path.splitext(filename)[0]
        epoch_dir = os.path.join(self.img_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        # attention shape: (block, head, seq, seq)
        avg_received_attention = attention.mean(axis=(0, 1, 2))
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(avg_received_attention.shape[0]), avg_received_attention)
        plt.xlabel("Token Index")
        plt.ylabel("Value")
        plt.title(f"{filename} - Average Received {mode.capitalize()} - All Blocks")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_average_received_{mode}.png"), bbox_inches="tight")
        plt.close()

    def plot_attention_gradient(self, attention_gradient, filename, epoch, block_idx):
        filename_wo_ext = os.path.splitext(filename)[0]
        epoch_dir = os.path.join(self.img_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        # attention_gradient shape: (head, seq, seq)
        avg_over_queries = attention_gradient.mean(axis=(0, 1))
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(avg_over_queries.shape[0]), avg_over_queries)
        plt.xlabel("Token Index")
        plt.ylabel("Value")
        plt.title(f"{filename} - Attention Gradient (avg over queries) - Block {block_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_{block_idx}_attention_gradient_queries.png"), bbox_inches="tight")
        plt.close()
        avg_over_keys = attention_gradient.mean(axis=(0, 2))
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(avg_over_keys.shape[0]), avg_over_keys)
        plt.xlabel("Token Index")
        plt.ylabel("Value")
        plt.title(f"{filename} - Attention Gradient (avg over keys) - Block {block_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_{block_idx}_attention_gradient_keys.png"), bbox_inches="tight")
        plt.close()

    def plot_snr(self, snr_values, filename, mode="standard"):
        filename_wo_ext = os.path.splitext(filename)[0]
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(snr_values)), snr_values)
        plt.xlabel("Epoch")
        plt.ylabel("SNR")
        plt.grid(True)
        if mode == "standard":
            plt.title(f"{filename} - Attention Gradient SNR")
            plt.savefig(os.path.join(self.img_dir, f"{filename_wo_ext}_attention_gradient_snr_curve.png"))
        elif mode == "pre":
            plt.title(f"{filename} - Attention Gradient (Pre Softmax) SNR")
            plt.savefig(os.path.join(self.img_dir, f"{filename_wo_ext}_attention_gradient_pre_softmax_snr_curve.png"))
        plt.close()
