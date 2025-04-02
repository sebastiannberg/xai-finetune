import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    # plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
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
    # plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
    plt.close()

def plot_attention_heatmap(attention: torch.Tensor, title: str):
    avg = attention.mean(dim=(0, 1, 2)).cpu().detach().numpy()

    # vmin = np.percentile(avg, 1)
    # vmax = np.percentile(avg, 99)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(avg, cmap='hot')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Attention Weight')

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Key Index")
    ax.set_ylabel("Query Index")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    return fig

def cls_argmax_percentage(tensor: torch.Tensor):
    batch_size, num_blocks, num_heads, num_queries, _ = tensor.shape
    total_queries = 0
    cls_attention_count = 0

    for sample in range(batch_size):
        for block in range(num_blocks):
            for head in range(num_heads):
                # Shape: [num_queries, num_keys]
                att = tensor[sample, block, head]

                max_attended_keys = att.argmax(dim=1)

                # Count how many queries in this matrix attend most to CLS (index 0)
                cls_attention_count += (max_attended_keys == 0).sum().item()
                total_queries += num_queries

    percentage = cls_attention_count / total_queries
    return percentage

def avg_received_attention_cls(tensor: torch.Tensor):
    avg = tensor.mean(dim=(0, 1, 2, 3))
    avg_received_attn_cls = avg[0]
    return avg_received_attn_cls
