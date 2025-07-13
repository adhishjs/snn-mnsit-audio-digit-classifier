
import torch
import torch.nn as nn

#define the SNN model
from SNN_network import SimpleSNN
from loading import load_checkpoint
from saving import save_checkpoint
from encoder import AudioDigitDataset
from torch.utils.data import DataLoader


def train_snn(model, dataloader, epochs=10, resume=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0

    if resume:
        try:
            start_epoch = load_checkpoint(model, optimizer)
            print(f"✅ Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("🚀 Starting training from scratch...")

    for epoch in range(start_epoch, epochs):
        total_loss, correct = 0, 0
        model.train()
        total_seen = 0
        for batch_spikes, batch_labels in dataloader:
            # batch_spikes: [batch_size, 8000, 20]
            # Reshape to [8000, batch_size, 20]
            batch_spikes = batch_spikes.permute(1, 0,2, 3).squeeze(2)  # [T, B, F]
            batch_labels = batch_labels.long()

            model.reset_state(batch_spikes.shape[1])
            out = model(batch_spikes)  # [B, 10]
            loss = loss_fn(out, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_correct = (out.argmax(dim=1) == batch_labels).sum().item()
            correct += batch_correct
            total_seen += batch_labels.shape[0]
            print(f"correct: {correct}, total: {total_seen}, acc: {correct / total_seen * 100:.6f}")

        acc = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")
        save_checkpoint(model, optimizer, epoch + 1)


dataset = AudioDigitDataset(root_dir="DATASET/TRAIN", n_mfcc=20, sample_rate=16000, time_steps=100)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# === Main ===
if __name__ == "__main__":

    model = SimpleSNN()
    train_snn(model, dataloader, epochs=2, resume=True)
