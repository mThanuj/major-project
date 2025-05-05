import os
import torch
import torchaudio
import torch.nn.functional as F

from model.utils.MFCCAutoencoder import ImprovedMFCCAutoencoder
from model.utils.CNN import ImprovedSpeechCNNClassifier
from model.utils.dataset import SpeechCommandsDataset


def extract_all_features(
    audio_file: str,
    window_size: float = 1.0,
    hop_size: float = 0.5,
) -> list[dict]:
    """
    Given a path to a .wav file (possibly longer than 1s), this returns a list of dicts for each
    sliding window segment containing:
      - 'start_time', 'end_time'        : segment boundaries in seconds
      - 'latent_features'               : encoder's feature map (numpy array)
      - 'logits'                        : raw classifier logits (numpy array)
      - 'probabilities'                 : softmaxed class probabilities (numpy array)
      - 'predicted_index'               : integer class index
      - 'predicted_label'               : human-readable class name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ae_path = os.path.join(base_dir, "model", "best", "best_autoencoder.pth")
    clf_path = os.path.join(base_dir, "model", "best", "best_classifier.pth")
    data_dir = os.path.join(base_dir, "model", "data")

    # Prepare dataset for transforms and labels
    ds = SpeechCommandsDataset(dataset_path=data_dir, augment=False, test_mode=True)
    idx2label = ds.index_to_label
    num_classes = len(ds.classes)

    # Load Autoencoder
    autoencoder = ImprovedMFCCAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(ae_path, map_location=device, weights_only=True)
    )
    autoencoder.eval()

    # Load Classifier with AE encoder
    classifier = ImprovedSpeechCNNClassifier(
        encoder=autoencoder.get_encoder(), num_classes=num_classes
    ).to(device)
    classifier.load_state_dict(
        torch.load(clf_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    # Load waveform
    waveform, sr = torchaudio.load(audio_file)  # [1, total_samples]
    total_samples = waveform.shape[1]
    win_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    results: list[dict] = []
    # Slide window
    for start in range(0, max(total_samples - win_samples + 1, 1), hop_samples):
        end = start + win_samples
        segment = waveform[:, start:end]
        # If last segment shorter, pad to full length
        if segment.shape[1] < win_samples:
            pad_amt = win_samples - segment.shape[1]
            segment = F.pad(segment, (0, pad_amt))

        # Compute MFCC and adjust to fixed target_length
        mfcc = ds.mfcc_transform(segment).squeeze(0)  # [n_mfcc, T]
        T = mfcc.shape[1]
        if T < ds.target_length:
            mfcc = F.pad(mfcc, (0, ds.target_length - T))
        elif T > ds.target_length:
            start_idx = (T - ds.target_length) // 2
            mfcc = mfcc[:, start_idx : start_idx + ds.target_length]

        # Normalize
        m, s = mfcc.mean(), mfcc.std() + 1e-8
        mfcc = (mfcc - m) / s
        x = mfcc.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,n_mfcc,target_length]

        with torch.no_grad():
            latent = autoencoder.get_latent_features(x)  # [1,128,h,w]
            logits = classifier(x)  # [1,num_classes]
            probs = torch.softmax(logits, dim=1)  # [1,num_classes]
            pred_idx = int(probs.argmax(dim=1).item())  # scalar
            pred_label = idx2label[pred_idx]

        results.append(
            {
                "start_time": start / sr,
                "end_time": min(end, total_samples) / sr,
                "latent_features": latent.cpu().numpy().tolist(),
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probs.cpu().numpy().tolist(),
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
            }
        )

    return results
