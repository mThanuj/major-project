import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
import argparse

from utils.CNN import (
    ImprovedSpeechCNNClassifier,
    train_classifier,
    evaluate_classifier,
)
from utils.MFCCAutoencoder import ImprovedMFCCAutoencoder, train_autoencoder
from utils.dataset import DATASET_PATH, SpeechCommandsDataset, download_dataset


def parse_args():
    parser = argparse.ArgumentParser("AE + CNN Training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--clf_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=12)
    return parser.parse_args()


def load_model_compat(model, path, device, strict=False):
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=strict)
        print(f"Loaded checkpoint: {path}")
    return model


def main():
    args = parse_args()
    if os.path.exists("best"):
        shutil.rmtree("best")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Download & prepare
    print("Downloading dataset…")
    download_dataset()
    ds = SpeechCommandsDataset(DATASET_PATH, augment=True)
    N = len(ds)
    train_N = int(0.8 * N)
    val_N = N - train_N
    train_ds, val_ds = random_split(
        ds, [train_N, val_N], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Autoencoder
    ae = ImprovedMFCCAutoencoder().to(device)
    ae = load_model_compat(ae, "best/best_autoencoder.pth", device, strict=False)
    ae = train_autoencoder(ae, train_loader, device, epochs=args.ae_epochs)
    os.makedirs("best", exist_ok=True)
    torch.save(ae.state_dict(), "best/best_autoencoder.pth")

    # Classifier
    encoder = ae.get_encoder()
    classifier = ImprovedSpeechCNNClassifier(encoder, num_classes=len(ds.classes)).to(
        device
    )
    classifier = load_model_compat(
        classifier, "best/best_classifier.pth", device, strict=False
    )
    classifier = train_classifier(
        ds, classifier, train_loader, val_loader, device, epochs=args.clf_epochs
    )
    torch.save(classifier.state_dict(), "best/best_classifier.pth")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    evaluate_classifier(classifier, val_loader, ds, device)
    print("All done!")


if __name__ == "__main__":
    main()
