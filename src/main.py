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

    # Data loader args
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for data loading")
    parser.add_argument("--seed", type=int, default=0, help="To control the random seed for reproducibility")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")

    # IFI
    parser.add_argument("--alpha", type=float, default=0.95, help="The strength of classification loss vs interpret loss")
    parser.add_argument("--grad_scale", type=float, default=1e5, help="Scaling up gradients to avoid uniform distribution for attention_interpret")

    return parser.parse_args()

def main():
    args = get_args()
    experiment_manager = ExperimentManager(args)
    experiment_manager.run_experiment()

if __name__ == "__main__":
    main()
