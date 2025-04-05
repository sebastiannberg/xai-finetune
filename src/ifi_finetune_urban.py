import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from timm.models.layers import trunc_normal_
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import datetime
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from dataset_urban import UrbanDataset
import models_vit as models_vit
from grad import attribute
from utils import cls_argmax_percentage, avg_received_attention_cls, plot_attention_heatmap

def perform_analysis(model, device, criterion, class_attention_grads, args, epoch, data_loader_stats, data_loader_plot):
    for fbank, label in data_loader_stats:
        fbank = fbank.to(device)
        label = label.to(device)

        logits, attention = model(fbank, return_attention=True)

        classification_loss = criterion(logits, label)
        logger.info(f'classification loss: {args.alpha * classification_loss.item()}')

        label_indices = torch.argmax(label, dim=1)
        selected_attention_grads = class_attention_grads[label_indices, ...]
        selected_attention_grads = args.grad_scale * selected_attention_grads

        # attention_wo_cls = attention[:, :, :, 1:, 1:]
        # attention_wo_cls_softmaxed = attention_wo_cls.softmax(dim=-1)
        # grads_wo_cls = selected_attention_grads[:, :, :, 1:, 1:]

        pre_attention_interpret = attention * selected_attention_grads
        post_attention_interpret = pre_attention_interpret.softmax(dim=-1)
        interpret_loss = -(post_attention_interpret.detach() * (attention + 1e-12).log()).sum(dim=-1).mean()
        logger.info(f'interpret loss: {(1 - args.alpha) * interpret_loss.item()}')

        # logger.info(
        #     f"Attention stats:\n"
        #     f"  shape: {attention.shape}\n"
        #     f"  max: {attention.max().item()}\n"
        #     f"  min: {attention.min().item()}\n"
        #     f"  mean: {attention.mean().item()}\n"
        #     f"  std: {attention.std().item()}"
        # )
        # logger.info(
        #     f"Attention wo CLS stats:\n"
        #     f"  shape: {attention_wo_cls.shape}\n"
        #     f"  max: {attention_wo_cls.max().item()}\n"
        #     f"  min: {attention_wo_cls.min().item()}\n"
        #     f"  mean: {attention_wo_cls.mean().item()}\n"
        #     f"  std: {attention_wo_cls.std().item()}"
        # )
        # logger.info(
        #     f"Attention wo CLS softmaxed stats:\n"
        #     f"  shape: {attention_wo_cls_softmaxed.shape}\n"
        #     f"  max: {attention_wo_cls_softmaxed.max().item()}\n"
        #     f"  min: {attention_wo_cls_softmaxed.min().item()}\n"
        #     f"  mean: {attention_wo_cls_softmaxed.mean().item()}\n"
        #     f"  std: {attention_wo_cls_softmaxed.std().item()}"
        # )
        # logger.info(
        #     f"Selected Attention Grads stats:\n"
        #     f"  shape: {selected_attention_grads.shape}\n"
        #     f"  max: {selected_attention_grads.max().item()}\n"
        #     f"  min: {selected_attention_grads.min().item()}\n"
        #     f"  mean: {selected_attention_grads.mean().item()}\n"
        #     f"  std: {selected_attention_grads.std().item()}"
        # )
        # logger.info(
        #     f"Selected Attention Grads wo CLS stats:\n"
        #     f"  shape: {grads_wo_cls.shape}\n"
        #     f"  max: {grads_wo_cls.max().item()}\n"
        #     f"  min: {grads_wo_cls.min().item()}\n"
        #     f"  mean: {grads_wo_cls.mean().item()}\n"
        #     f"  std: {grads_wo_cls.std().item()}"
        # )
        # logger.info(
        #     f"Pre-Softmax Attention Interpret stats:\n"
        #     f"  shape: {pre_attention_interpret.shape}\n"
        #     f"  max: {pre_attention_interpret.max().item()}\n"
        #     f"  min: {pre_attention_interpret.min().item()}\n"
        #     f"  mean: {pre_attention_interpret.mean().item()}\n"
        #     f"  std: {pre_attention_interpret.std().item()}"
        # )
        # logger.info(
        #     f"Post-Softmax Attention Interpret stats:\n"
        #     f"  shape: {post_attention_interpret.shape}\n"
        #     f"  max: {post_attention_interpret.max().item()}\n"
        #     f"  min: {post_attention_interpret.min().item()}\n"
        #     f"  mean: {post_attention_interpret.mean().item()}\n"
        #     f"  std: {post_attention_interpret.std().item()}"
        # )

        epoch_path = os.path.join(IMG_PATH, f"epoch_{epoch}")
        os.makedirs(epoch_path, exist_ok=True)

        heatmap_fig = plot_attention_heatmap(attention, title="Attention Heatmap")
        heatmap_fig.savefig(os.path.join(epoch_path, 'attention_heatmap.png'))
        plt.close(heatmap_fig)

        grads_heatmap_fig = plot_attention_heatmap(class_attention_grads, title="Attentiong Gradient Heatmap")
        grads_heatmap_fig.savefig(os.path.join(epoch_path, 'class_attention_grads_heatmap.png'))
        plt.close(grads_heatmap_fig)

        attention_interpret_heatmap_fig = plot_attention_heatmap(post_attention_interpret, title="Attention Interpret Heatmap")
        attention_interpret_heatmap_fig.savefig(os.path.join(epoch_path, 'post_attention_interpret_heatmap.png'))
        plt.close(attention_interpret_heatmap_fig)

        attention_cls_argmax_percentage = cls_argmax_percentage(attention)
        logger.info(f"Attention Argmax Percentage for [CLS] token: {attention_cls_argmax_percentage}")

        avg_received_attn_cls = avg_received_attention_cls(attention)
        logger.info(f"Average Received Attention for [CLS] token: {avg_received_attn_cls}")

def main():
    dataset_train = UrbanDataset(
        root=URBAN_PATH,
        fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        mixup_prob=0.5,
        roll_mag_aug=True,
        target_length=args.target_length,
        freqm=48,
        timem=192,
        num_classes=args.num_classes
    )
    dataset_val = UrbanDataset(
        root=URBAN_PATH,
        fold=[10],
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Create stats and plot set used for debugging and statistics
    stats_indices = random.sample(range(len(dataset_train)), 32)
    stats_subset = Subset(dataset_train, stats_indices)
    data_loader_stats = DataLoader(stats_subset, batch_size=32, shuffle=False)
    plot_indices = random.sample(range(len(dataset_train)), 5)
    plot_subset = Subset(dataset_train, plot_indices)
    data_loader_plot = DataLoader(plot_subset, batch_size=5, shuffle=False)

    # Partition data to samples for each label
    dataset_interpret = UrbanDataset(
        root=URBAN_PATH,
        fold=[1], # TODO: Try all training folds
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset_interpret)):
        _, data_y = dataset_interpret[idx]
        class_idx = data_y.argmax().item()
        class_to_indices[class_idx].append(idx)
    class_loaders = []
    for class_idx in range(args.num_classes):
        indices = class_to_indices[class_idx]
        subset = Subset(dataset_interpret, indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        class_loaders.append(loader)

    model = models_vit.__dict__['vit_base_patch16'](
        num_classes=args.num_classes,
        drop_path_rate=0.1,
        global_pool=True,
        mask_2d=True,
        use_custom_patch=False
    )
    model.patch_embed = PatchEmbed_new(
        img_size=(args.target_length, 128),
        patch_size=(16, 16),
        in_chans=1,
        embed_dim=768,
        stride=16
    )
    model.pos_embed = nn.Parameter(
        torch.zeros(1, model.patch_embed.num_patches + 1, 768),
        requires_grad=False
    )
    checkpoint = torch.load(os.path.join(CKPT_PATH, args.checkpoint), map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Check if positional embeddings need to be interpolated
    if 'pos_embed' in checkpoint_model:
        if checkpoint_model['pos_embed'].shape != model.pos_embed.shape:
            logger.info(f"Interpolating positional embeddings from {checkpoint_model['pos_embed'].shape} to {torch.Size([1, 257, 768])}")
            pos_embed_checkpoint = checkpoint_model['pos_embed']  # [1, old_num_tokens, embed_dim]
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
                mode='bilinear',
                align_corners=False
            )
            # Permute back and reshape to (1, new_num_tokens, embed_dim)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_embed_checkpoint.shape[-1])

            # Concatenate the class token back
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    # Remove the classification head weights if they don't match the current model's output shape
    # This prevents shape mismatch issues when fine-tuning on a different number of classes
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
            del checkpoint_model[k]
    # Load the remaining pre-trained weights into the model
    # strict=False allows partial loading (ignores missing keys like the removed head)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    # Reinitialize the classification head with small random values
    # This ensures the new classification layer starts learning from scratch
    trunc_normal_(model.head.weight, std=2e-5)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler()

    logger.info(f'Device: {device}')

    start_time = time.time()
    for epoch in tqdm(range(args.epochs), desc="Training Progress", leave=True, position=0):
        if epoch == 0:
            # Training without interpretability loss
            model.train()

            total_train_loss = 0.0

            for fbank, label in tqdm(data_loader_train, desc=f"Training [Epoch {epoch+1}/{args.epochs}]", leave=False, position=1):
                fbank = fbank.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with autocast():
                    logits, attention = model(fbank, return_attention=True)
                    loss = criterion(logits, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            epoch_loss = total_train_loss / len(data_loader_train)

            # Calculate attention gradients
            class_attention_grads = attribute(model, class_loaders)

            # Validation
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for fbank_val, label_val in tqdm(data_loader_val, desc="Validation", leave=False, position=1):
                    fbank_val = fbank_val.to(device)
                    label_val = label_val.to(device)

                    logits_val = model(fbank_val)
                    loss_val = criterion(logits_val, label_val)
                    total_val_loss += loss_val.item()

                    preds = torch.argmax(logits_val, dim=1)
                    true_classes = torch.argmax(label_val, dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(true_classes.cpu())

            val_loss = total_val_loss / len(data_loader_val)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            logger.info("-"*40)
            logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
            logger.info(f"  Train Loss:    {epoch_loss:.4f}")
            logger.info(f"  Val Loss:      {val_loss:.4f}")
            logger.info(f"  Val Accuracy:  {val_accuracy:.4f}")
            logger.info(f"  Val F1:        {val_f1:.4f}")

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            logger.info(f"Current learning rate after step: {current_lr[0]:.6f}")

            # Analysis
            perform_analysis(model, device, criterion, class_attention_grads, args, epoch, data_loader_stats, data_loader_plot)
        else:
            # Training with interpretability loss
            model.train()
            # torch.autograd.set_detect_anomaly(True)

            total_train_loss = 0.0

            for fbank, label in tqdm(data_loader_train, desc=f"Training [Epoch {epoch+1}/{args.epochs}]", leave=False, position=1):
                fbank = fbank.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with autocast():
                    logits, attention = model(fbank, return_attention=True)
                    classification_loss = criterion(logits, label)

                    label_indices = torch.argmax(label, dim=1)
                    selected_attention_grads = class_attention_grads[label_indices, ...]
                    selected_attention_grads = args.grad_scale * selected_attention_grads

                    # attention_wo_cls = attention[:, :, :, 1:, 1:]
                    # attention_wo_cls_softmaxed = attention_wo_cls.softmax(dim=-1)
                    # grads_wo_cls = selected_attention_grads[:, :, :, 1:, 1:]

                    pre_attention_interpret = attention * selected_attention_grads
                    post_attention_interpret = pre_attention_interpret.softmax(dim=-1)

                    interpret_loss = -(post_attention_interpret.detach() * (attention + 1e-12).log()).sum(dim=-1).mean()

                    loss = (args.alpha * classification_loss) + ((1 - args.alpha) * interpret_loss)
                    # loss = classification_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            epoch_loss = total_train_loss / len(data_loader_train)

            # Calculate attention gradients
            class_attention_grads = attribute(model, class_loaders)

            # Validation
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for fbank_val, label_val in tqdm(data_loader_val, desc="Validation", leave=False, position=1):
                    fbank_val = fbank_val.to(device)
                    label_val = label_val.to(device)

                    logits_val = model(fbank_val)
                    loss_val = criterion(logits_val, label_val)
                    total_val_loss += loss_val.item()

                    preds = torch.argmax(logits_val, dim=1)
                    true_classes = torch.argmax(label_val, dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(true_classes.cpu())

            val_loss = total_val_loss / len(data_loader_val)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            logger.info("-"*40)
            logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
            logger.info(f"  Train Loss:    {epoch_loss:.4f}")
            logger.info(f"  Val Loss:      {val_loss:.4f}")
            logger.info(f"  Val Accuracy:  {val_accuracy:.4f}")
            logger.info(f"  Val F1:        {val_f1:.4f}")

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            logger.info(f"Current learning rate after step: {current_lr[0]:.6f}")

            # Analysis
            perform_analysis(model, device, criterion, class_attention_grads, args, epoch, data_loader_stats, data_loader_plot)

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(CKPT_PATH, f'epoch_{epoch+1}_{timestamp}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1,
                    'args': vars(args)
                }, model_path)
                logger.info(f'Model saved at epoch {epoch+1}: {model_path}')

    total_time = time.time() - start_time
    logger.info(f'Total training time: {total_time / 60:.2f} minutes')

if __name__ == '__main__':
    main()
