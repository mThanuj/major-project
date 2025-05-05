import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ImprovedSpeechCNNClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 36):
        super().__init__()
        self.encoder = encoder
        # Always fine‑tune the encoder
        for p in self.encoder.parameters():
            p.requires_grad = True

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Adaptive pooling to (1,1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # reduced from 0.4
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        weights = self.attention(features)
        features = features * weights
        pooled = self.pool(features)
        out = self.classifier(pooled)
        return out


def train_classifier(
    dataset,
    classifier: ImprovedSpeechCNNClassifier,
    train_loader,
    val_loader,
    device,
    epochs: int = 30,
    patience: int = 10,
):
    """Train the classifier, reporting per‑class metrics each epoch."""
    classifier.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate LRs: encoder vs. attention+head
    optimizer = optim.AdamW(
        [
            {"params": classifier.encoder.parameters(), "lr": 1e-4},
            {"params": classifier.attention.parameters(), "lr": 1e-3},
            {"params": classifier.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=4
    )

    best_val_acc = 0.0
    wait = 0
    idx2label = dataset.index_to_label

    for epoch in range(1, epochs + 1):
        # --- Training ---
        classifier.train()
        running_loss = correct = total = 0
        class_correct = {lbl: 0 for lbl in idx2label.values()}
        class_total = {lbl: 0 for lbl in idx2label.values()}

        with tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}") as bar:
            for inputs, labels in bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # per‑class update
                for t, p in zip(labels, preds):
                    lbl = idx2label[t.item()]
                    class_total[lbl] += 1
                    if p == t:
                        class_correct[lbl] += 1

                bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"  → Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # Print worst & best 3 classes
        train_accs = {
            lbl: (
                (100 * class_correct[lbl] / class_total[lbl])
                if class_total[lbl] > 0
                else 0.0
            )
            for lbl in class_total
        }
        worst3 = sorted(train_accs.items(), key=lambda x: x[1])[:3]
        best3 = sorted(train_accs.items(), key=lambda x: x[1], reverse=True)[:3]
        print("    • Worst 3 classes:")
        for lbl, acc in worst3:
            print(
                f"       {lbl:15s}: {acc:5.1f}% ({class_correct[lbl]}/{class_total[lbl]})"
            )
        print("    • Best 3 classes:")
        for lbl, acc in best3:
            print(
                f"       {lbl:15s}: {acc:5.1f}% ({class_correct[lbl]}/{class_total[lbl]})"
            )

        # --- Validation ---
        classifier.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad(), tqdm(val_loader, desc="[ Val ]") as bar:
            for inputs, labels in bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                bar.set_postfix(
                    loss=val_loss / (bar.n + 1), acc=100 * val_correct / val_total
                )

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        print(f"  → Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")

        # Scheduler & Early Stopping
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            os.makedirs("best", exist_ok=True)
            torch.save(classifier.state_dict(), "best/best_classifier.pth")
            print(f"  ✓ Saved best model (acc={best_val_acc:.2f}%)")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ✗ Early stopping at epoch {epoch}")
                break

    # Load best
    classifier.load_state_dict(
        torch.load("best/best_classifier.pth", map_location=device, weights_only=True),
        strict=True,
    )
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return classifier


def evaluate_classifier(classifier, val_loader, dataset, device):
    """Comprehensive evaluation of the classifier"""
    classifier.to(device).eval()

    total_correct = 0

    total_samples = 0

    index_to_label = dataset.index_to_label

    label_correct = {label: 0 for label in index_to_label.values()}

    label_total = {label: 0 for label in index_to_label.values()}

    with torch.no_grad(), tqdm(
        val_loader, desc="Evaluating", unit="batch"
    ) as valloader:

        for inputs, labels in valloader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = classifier(inputs)

            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()

            total_samples += labels.size(0)

            for true, pred in zip(labels, predicted):

                true_label = index_to_label[true.item()]

                label_total[true_label] += 1

                if true == pred:

                    label_correct[true_label] += 1

            valloader.set_postfix(
                accuracy=f"{100 * total_correct / total_samples:.2f}%"
            )

    overall_accuracy = 100 * total_correct / total_samples

    print(f"\nFinal Validation Accuracy: {overall_accuracy:.2f}%")

    # Print per-class accuracy (sorted by performance)

    print("\nPer-Class Accuracy:")

    class_accuracies = [
        (
            label,
            100 * label_correct[label] / max(1, label_total[label]),
            label_total[label],
        )
        for label in sorted(label_total.keys())
    ]

    # Sort by accuracy

    class_accuracies.sort(key=lambda x: x[1])

    # Print results

    for label, accuracy, count in class_accuracies:

        print(f"{label:15s}: {accuracy:.2f}% ({label_correct[label]}/{count})")

    # Print summary statistics

    print("\nSummary Statistics:")

    accuracies = [acc for _, acc, _ in class_accuracies]

    print(f"Mean class accuracy: {sum(accuracies) / len(accuracies):.2f}%")

    print(f"Median class accuracy: {accuracies[len(accuracies) // 2]:.2f}%")

    print(f"Min class accuracy: {min(accuracies):.2f}%")

    print(f"Max class accuracy: {max(accuracies):.2f}%")
