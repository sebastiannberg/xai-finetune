import argparse

from experiment_manager import ExperimentManager


def get_args():
    parser = argparse.ArgumentParser(description="Finetune on UrbanSound8K dataset")
    parser.add_argument("--checkpoint", type=str, default='pretrained.pth', help='Filename for model checkpoint to load before fine-tuning')
    parser.add_argument("--num_classes", type=int, default=10, help='Number of target classes in output')
    parser.add_argument("--target_length", type=int, default=512, help='Time frames for fbank')
    parser.add_argument("--epochs", type=int, default=60, help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size in data loaders')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument("--num_workers", type=int, default=10, help='Number of worker threads for data loading')
    parser.add_argument("--seed", type=int, default=0, help='To control the random seed for reproducibility')
    return parser.parse_args()

def main():
    pass

if __name__ == "__main__":
    main()
