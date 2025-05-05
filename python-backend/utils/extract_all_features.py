import os
import torch
import torchaudio
import torch.nn.functional as F

from model.utils.MFCCAutoencoder import ImprovedMFCCAutoencoder
from model.utils.CNN import ImprovedSpeechCNNClassifier
from model.utils.dataset import SpeechCommandsDataset


def extract_all_features(audio_file: str) -> dict:
    """
    Given a path to a .wav file, this returns:
      - 'latent_features': the encoder's feature map (numpy array)
      - 'logits': raw classifier logits (numpy array)
      - 'probabilities': softmaxed class probabilities (numpy array)
      - 'predicted_index': the integer class index
      - 'predicted_label': the human-readable class name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths relative to python-backend/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ae_path = os.path.join(base_dir, "model", "best", "best_autoencoder.pth")
    clf_path = os.path.join(base_dir, "model", "best", "best_classifier.pth")
    data_dir = os.path.join(base_dir, "model", "data")

    # --- Prepare dataset (for transform + label map) ---
    ds = SpeechCommandsDataset(
        dataset_path=data_dir,
        augment=False,
        test_mode=True,
    )
    idx2label = ds.index_to_label
    num_classes = len(ds.classes)

    # --- Load Autoencoder ---
    autoencoder = ImprovedMFCCAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(ae_path, map_location=device, weights_only=True)
    )
    autoencoder.eval()

    # --- Load Classifier (using the AE encoder) ---
    # Pass the AE's encoder module into the classifier
    classifier = ImprovedSpeechCNNClassifier(
        encoder=autoencoder.get_encoder(),
        num_classes=num_classes,
    ).to(device)
    classifier.load_state_dict(
        torch.load(clf_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    # --- Preprocess the audio file ---
    waveform, _ = torchaudio.load(audio_file)  # [1, T]
    mfcc = ds.mfcc_transform(waveform).squeeze(0)  # [n_mfcc, T]

    # Pad or crop to target_length
    T = mfcc.shape[1]
    if T < ds.target_length:
        mfcc = F.pad(mfcc, (0, ds.target_length - T))
    elif T > ds.target_length:
        start = (T - ds.target_length) // 2
        mfcc = mfcc[:, start : start + ds.target_length]

    # Per-sample normalization
    m, s = mfcc.mean(), mfcc.std() + 1e-8
    mfcc = (mfcc - m) / s

    # Add batch & channel dims: [1, 1, n_mfcc, target_length]
    x = mfcc.unsqueeze(0).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        # Latent feature map from encoder
        latent = autoencoder.get_latent_features(x)  # [1, 128, 5, 12] (for example)

        # Classification
        logits = classifier(x)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1)  # [1, num_classes]
        pred_idx = int(torch.argmax(probs, dim=1).item())  # scalar
        pred_label = idx2label[pred_idx]

    return {
        "latent_features": latent.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "probabilities": probs.cpu().numpy(),
        "predicted_index": pred_idx,
        "predicted_label": pred_label,
    }
