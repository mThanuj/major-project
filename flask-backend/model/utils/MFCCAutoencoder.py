import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class ImprovedMFCCAutoencoder(nn.Module):
    def __init__(self, dropout_p: float = 0.2):
        super().__init__()

        def enc_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.01, inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_p),
            )

        def dec_block(in_c, skip_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c + skip_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = enc_block(1, 32)
        self.enc2 = enc_block(32, 64)
        self.enc3 = enc_block(64, 128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, output_padding=(0, 1)
        )
        self.dec3 = dec_block(64, 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = dec_block(32, 32, 32)

        self.up1 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.up3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        out = self.final_conv(d1)
        return F.interpolate(out, size=(40, 100), mode="bilinear", align_corners=False)

    def get_encoder(self) -> nn.Sequential:
        return nn.Sequential(self.enc1, self.enc2, self.enc3)

    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        # deterministic (no dropout)
        x = self.enc1[0:6](x)
        x = self.enc1[6](x)
        x = self.enc2[0:6](x)
        x = self.enc2[6](x)
        x = self.enc3[0:6](x)
        x = self.enc3[6](x)
        return x


def train_autoencoder(
    model: ImprovedMFCCAutoencoder,
    dataloader,
    device,
    epochs: int = 30,
    patience: int = 7,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "best",
):
    model.to(device).train()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )

    best_loss = float("inf")
    wait = 0
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_autoencoder.pth")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        with tqdm(dataloader, desc=f"AE Epoch {epoch}/{epochs}") as bar:
            for inputs, _ in bar:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                bar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(dataloader)
        print(f"AE Epoch {epoch}: Loss={epoch_loss:.6f}")
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best AE (loss={best_loss:.6f}) → {ckpt_path}")
        else:
            wait += 1
            if wait >= patience:
                print("  ✗ Early stopping AE")
                break

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True), strict=False
    )
    print("AE training complete. Best loss:", best_loss)
    return model
