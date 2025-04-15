import os
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import rcParams


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
        epochs = range(1, len(train_loss_list) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss_list, label="Training Loss", color="blue", linewidth=2)
        plt.plot(epochs, val_loss_list, label="Validation Loss", color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.img_dir, "loss_curve.png"))
        plt.close()

    def plot_accuracy_f1_curve(self, acc_list, f1_list):
        epochs = range(1, len(acc_list) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, acc_list, label="Accuracy", color="blue", linewidth=2)
        plt.plot(epochs, f1_list, label="F1", color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.img_dir, "accuracy_f1_curve.png"))
        plt.close()

    def plot_spectrogram(self, spectrogram, filename):
        plt.figure(figsize=(10, 3))
        plt.imshow(
            spectrogram,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            aspect="auto"
        )
        plt.title(f"{filename}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()

        filename_wo_ext = os.path.splitext(filename)[0]
        plt.savefig(os.path.join(self.img_dir, f"{filename_wo_ext}_spectrogram.png"), bbox_inches="tight")
        plt.close()

    def plot_attention_heatmap(self, attention, filename, epoch, mode="attention"):
        # attention shape: (block, head, seq, seq)
        filename_wo_ext = os.path.splitext(filename)[0]
        epoch_dir = os.path.join(self.img_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for i, block_attention in enumerate(attention):
            avg_attention = block_attention.mean(axis=0)
            plt.figure(figsize=(10, 10))
            plt.imshow(
                avg_attention,
                cmap="hot",
                aspect="equal"
            )
            ax = plt.gca()
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            plt.title(f"{filename} - {mode.capitalize()} - Block {i}")
            plt.xlabel("Key Index")
            plt.ylabel("Query Index")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_{i}_{mode}_heatmap.png"), bbox_inches="tight")
            plt.close()

        avg_blocks_attention = attention.mean(axis=(0, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(
            avg_blocks_attention,
            cmap="hot",
            aspect="equal"
        )
        ax = plt.gca()
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        plt.title(f"{filename} - {mode.capitalize()} - Avg Over Blocks")
        plt.xlabel("Key Index")
        plt.ylabel("Query Index")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"{filename_wo_ext}_block_all_{mode}_heatmap.png"), bbox_inches="tight")
        plt.close()

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.05)
        # cbar = fig.colorbar(im, cax=cax)
        # cbar.set_label('Attention Weight')

#     def stack_visualizations(self, case: Dict, visualization_methods: List[str]) -> Figure:
#         fig, axs = plt.subplots(len(visualization_methods), 1, figsize=(20, 7 * len(visualization_methods)))
#
#         title_str = f"{case['method'].capitalize()}"
#         title_str += f"\nClass: {case['explain_index_name']}    Probability: {round(float(case['probability']), 3):.3f}"
#         if case["target_indices"] and self.config["targets_in_title"]:
#             title_str += f"\nTargets: {case['target_names']}"
#         fig.suptitle(title_str)
#
#         if len(visualization_methods) == 1:
#             axs = [axs]
#
#         for idx, method_name in enumerate(visualization_methods):
#             visualization_method = getattr(self, method_name)
#             visualization_method(case, ax=axs[idx])
#             axs[idx].set_xlabel("Time")
#             axs[idx].set_ylabel("Frequency")
#
#         plt.subplots_adjust(hspace=0)
#         plt.tight_layout()
#         return fig
#
#     def _validate_shape(self, case: Dict) -> Tuple[NDArray, NDArray]:
#         input_tensor = case["input_tensor"]
#         input_tensor_np = np.squeeze(input_tensor.numpy()).T
#         attributions = case["attributions"]
#         attributions_np = np.squeeze(attributions.cpu().numpy()).T
#         assert input_tensor_np.shape == attributions_np.shape
#         return input_tensor_np, attributions_np
#
#     def original_image(self, case: Dict, ax: Axes) -> None:
#         input_tensor_np, _ = self._validate_shape(case)
#         im = ax.imshow(
#             input_tensor_np,
#             cmap=self.config["original_image_cmap"],
#             origin='lower',
#             interpolation='nearest',
#             aspect='auto'
#         )
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="1%", pad=0.1)
#         plt.colorbar(im, cax=cax)
#
#     def heatmap(self, case: Dict, ax: Axes) -> None:
#         input_tensor_np, attributions_np = self._validate_shape(case)
#
#         if self.config["normalize"]:
#             max_abs_attr = np.max(np.abs(attributions_np))
#             max_abs_attr = max_abs_attr if max_abs_attr != 0 else 1e-10
#             normalized_attributions = attributions_np / max_abs_attr
#         else:
#             normalized_attributions = attributions_np
#
#         alpha = self.config["alpha"]
#
#         ax.imshow(
#             input_tensor_np,
#             cmap='gray',
#             origin='lower',
#             interpolation='nearest',
#             aspect='auto'
#         )
#
#         im = ax.imshow(
#             normalized_attributions,
#             cmap=self.config["attributions_cmap"],
#             origin='lower',
#             interpolation='nearest',
#             aspect='auto',
#             alpha=alpha,
#             vmin=-1,
#             vmax=1
#         )
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="1%", pad=0.1)
#         plt.colorbar(im, cax=cax)

# def plot_class_attention_grads(class_attention_grads: torch.Tensor):
#     avg_class = class_attention_grads.mean(dim=0)
#     avg_block = avg_class.mean(dim=0)
#     avg_head = avg_block.mean(dim=0)
#     avg_query = avg_head.mean(dim=0)
#
#     plt.figure(figsize=(10, 6))
#     x_vals = np.arange(avg_query.size(0))
#     plt.plot(x_vals, avg_query.cpu().numpy())
#     plt.xlabel("Token Index")
#     plt.ylabel("Average Gradient Value")
#     plt.title("Average Attention Gradient (averaged over classes/blocks/heads/query)")
#     plt.tight_layout()
#
#     file_name = "class_attention_grads.png"
#     # plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
#     plt.close()
#
# def plot_attention(attention: torch.Tensor):
#     avg_batch = attention.mean(dim=0)
#     avg_block = avg_batch.mean(dim=0)
#     avg_head = avg_block.mean(dim=0)
#     avg_query = avg_head.mean(dim=0)
#
#     plt.figure(figsize=(10, 6))
#     x_vals = np.arange(avg_query.size(0))
#     plt.plot(x_vals, avg_query.cpu().detach().numpy())
#     plt.xlabel("Token Index")
#     plt.ylabel("Average Attention Value")
#     plt.title("Average Attention (averaged over batch/blocks/heads/query)")
#     plt.tight_layout()
#
#     file_name = "attention.png"
#     # plt.savefig(os.path.join(PROJECT_ROOT, 'img', file_name))
#     plt.close()
#

# def cls_argmax_percentage(tensor: torch.Tensor):
#     batch_size, num_blocks, num_heads, num_queries, _ = tensor.shape
#     total_queries = 0
#     cls_attention_count = 0
#
#     for sample in range(batch_size):
#         for block in range(num_blocks):
#             for head in range(num_heads):
#                 # Shape: [num_queries, num_keys]
#                 att = tensor[sample, block, head]
#
#                 max_attended_keys = att.argmax(dim=1)
#
#                 # Count how many queries in this matrix attend most to CLS (index 0)
#                 cls_attention_count += (max_attended_keys == 0).sum().item()
#                 total_queries += num_queries
#
#     percentage = cls_attention_count / total_queries
#     return percentage
#
# def avg_received_attention_cls(tensor: torch.Tensor):
#     avg = tensor.mean(dim=(0, 1, 2, 3))
#     avg_received_attn_cls = avg[0]
#     return avg_received_attn_cls
