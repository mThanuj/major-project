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
    hop_size: float = 1.0,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    target_length: int = 100,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ——— Load models ——————————————————————————————————————————————
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ae_path = os.path.join(base_dir, "model", "best", "best_autoencoder.pth")
    clf_path = os.path.join(base_dir, "model", "best", "best_classifier.pth")
    data_dir = os.path.join(base_dir, "model", "data")

    # Dataset only for labels/index mapping
    ds = SpeechCommandsDataset(dataset_path=data_dir, augment=False, test_mode=True)
    idx2label = ds.index_to_label
    num_classes = len(ds.classes)

    # Autoencoder
    autoencoder = ImprovedMFCCAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(ae_path, map_location=device, weights_only=True)
    )
    autoencoder.eval()

    # Classifier
    classifier = ImprovedSpeechCNNClassifier(
        encoder=autoencoder.get_encoder(), num_classes=num_classes
    ).to(device)
    classifier.load_state_dict(
        torch.load(clf_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    # ——— Prepare MFCC transform ————————————————————————————————
    melkwargs = {
        "n_fft": int(0.025 * sample_rate),  # 400
        "hop_length": int(0.010 * sample_rate),  # 160
        "n_mels": n_mfcc,
        "f_min": 50,
        "f_max": 8000,
    }
    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        log_mels=True,
        melkwargs=melkwargs,
    )

    # ——— Load & downmix audio ————————————————————————————————
    waveform, sr = torchaudio.load(audio_file)  # [C, N]
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, N]

    total_samples = waveform.shape[1]
    win_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    results: list[dict] = []
    for start in range(0, max(total_samples - win_samples + 1, 1), hop_samples):
        end = start + win_samples
        segment = waveform[:, start:end]

        # pad last segment if shorter
        if segment.shape[1] < win_samples:
            pad_amt = win_samples - segment.shape[1]
            segment = F.pad(segment, (0, pad_amt))

        # compute MFCC → [1, n_mfcc, T]
        mfcc = mfcc_tf(segment)
        mfcc = mfcc.squeeze(0)  # [n_mfcc, T]
        T = mfcc.shape[1]

        # crop or pad time axis to target_length
        if T < target_length:
            mfcc = F.pad(mfcc, (0, target_length - T))
        elif T > target_length:
            start_idx = (T - target_length) // 2
            mfcc = mfcc[:, start_idx : start_idx + target_length]

        # normalize per chunk
        m, s = mfcc.mean(), mfcc.std() + 1e-8
        mfcc = (mfcc - m) / s

        # prep for model: [1,1,n_mfcc, target_length]
        x = mfcc.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = autoencoder.get_latent_features(x)
            logits = classifier(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_label = idx2label[pred_idx]

        results.append(
            {
                "start_time": round(start / sample_rate, 3),
                "end_time": round(min(end, total_samples) / sample_rate, 3),
                "latent_features": latent.cpu().numpy().tolist(),
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probs.cpu().numpy().tolist(),
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
            }
        )

    return results
