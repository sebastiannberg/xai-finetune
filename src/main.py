import argparse

from experiment_manager import ExperimentManager


def get_args():
    parser = argparse.ArgumentParser(description="Finetune on UrbanSound8K dataset")

    # Mode
    parser.add_argument("--mode", type=str, choices=["baseline", "ifi"], required=True, help="Choose experiment mode")

    # Model args
    parser.add_argument("--checkpoint", type=str, default="pretrained.pth", help="Filename for model checkpoint to load before fine-tuning")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of target classes in output")
    parser.add_argument("--target_length", type=int, default=512, help="Number of time frames for fbank spectrograms")

    # Training args
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in data loaders")
    parser.add_argument("--seed", type=int, default=0, help="To control the random seed for reproducibility")
    parser.add_argument("--save_frequency", type=int, default=100, help="Save a model ckpt every x epochs")

    # Data loader args
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for data loading")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")

    # IFI
    parser.add_argument("--alpha", type=float, default=0.95, help="The strength of classification loss vs interpret loss")
    parser.add_argument("--start_epoch", type=int, default=2, help="What epoch to start the interpretability loss")
    parser.add_argument("--which_blocks", type=str, default="all", choices=["all", "first", "last"], help="What blocks to apply interpretability loss to")
    parser.add_argument("--cls_deactivated", action="store_true", help="Should CLS token be included in interpretability loss")

    parser.add_argument(
        "--grad_processing_mode",
        choices=["off", "temp_scaling", "relu", "standardize", "percentile_clamping", "l1_amplification"],
        default="temp_scaling",
        help="Choose how to process attention gradiets before softmax to avoid uniform distributions"
    )
    parser.add_argument("--temperature", type=float, default=1e-5, help="Scaling up gradients to avoid uniform distribution for attention_interpret")
    parser.add_argument("--percentile", type=float, default=0.2, help="Sparsification, zero out the percentile smallest values in target")
    parser.add_argument("--l1_lambda", type=float, default=0.5, help="Lambda value for L1 amplification to increase dynamic range")

    # Augmentation
    parser.add_argument("--mixup_prob", type=float, default=0.0, help="Probability of applying mixup data augmentation for a sample")
    parser.add_argument("--freqm", type=int, default=0, help="Frequency masking")
    parser.add_argument("--timem", type=int, default=0, help="Time masking")
    parser.add_argument("--roll_mag_aug", action="store_true", help="Applying time shift augmentation or not")

    # Slurm
    parser.add_argument('--sbatch_script', type=str, default="unknown", help="Name of the sbatch script used to launch the job")

    return parser.parse_args()

def main():
    args = get_args()
    experiment_manager = ExperimentManager(args)
    experiment_manager.run_experiment()

if __name__ == "__main__":
    main()
