import numpy as np
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def plot_class_attention_grads(class_attention_grads: torch.Tensor):
    avg_class = class_attention_grads.mean(dim=0)
    avg_block = avg_class.mean(dim=0)
    avg_head = avg_block.mean(dim=0)
    avg_query = avg_head.mean(dim=0)

    plt.figure(figsize=(10, 6))
    x_vals = np.arange(avg_query.size(0))
    plt.plot(x_vals, avg_query.cpu().numpy())
    plt.xlabel("Token Index")
    plt.ylabel("Average Gradient Value")
    plt.title("Average Attention Gradient (averaged over classes/blocks/heads/query)")
    plt.tight_layout()

    file_name = "class_attention_grads.png"
    plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
    plt.close()

def plot_attention(attention: torch.Tensor):
    avg_batch = attention.mean(dim=0)
    avg_block = avg_batch.mean(dim=0)
    avg_head = avg_block.mean(dim=0)
    avg_query = avg_head.mean(dim=0)

    plt.figure(figsize=(10, 6))
    x_vals = np.arange(avg_query.size(0))
    plt.plot(x_vals, avg_query.cpu().detach().numpy())
    plt.xlabel("Token Index")
    plt.ylabel("Average Attention Value")
    plt.title("Average Attention (averaged over batch/blocks/heads/query)")
    plt.tight_layout()

    file_name = "attention.png"
    plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
    plt.close()

def plot_attention_heatmap(attention: torch.Tensor):
    avg = attention.mean(dim=(0, 1, 2))

    plt.figure(figsize=(15, 15))
    plt.imshow(avg.cpu().detach().numpy(), cmap='viridis')
    plt.title("Attention Heatmap (averaged over batch/blocks/heads)")
    plt.tight_layout()

    file_name = "attention_heatmap.png"
    plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
    plt.close()
