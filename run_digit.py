import torch
from torch.utils.data import DataLoader, Subset
from SNN_network import SimpleSNN
from loading import load_model_from_checkpoint
from encoder import AudioDigitDataset
import random
import os
# === MAIN TEST ===
if __name__ == "__main__":
    # Load test dataset
    test_dataset = AudioDigitDataset(root_dir="DATASET_TEST", n_mfcc=20, sample_rate=16000, time_steps=100)
    batch_size=10
# Randomly select 10 indices
    random_indices = random.sample(range(len(test_dataset)), batch_size)

    # Create a subset
    sampled_dataset = Subset(test_dataset, random_indices)
    test_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)
    print("Selected Files:")
    for i in random_indices:
        print(os.path.basename(test_dataset.filepaths[i]))
    # print(test_loader.dataset[0])
    # Load trained model
    model = SimpleSNN()
    load_model_from_checkpoint(model, "parameters.pt")
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_spikes, batch_labels in test_loader:  # Limit to first 10 samples for testing
            # batch_spikes: [B, T, 1, F] → [T, B, F]
            batch_spikes = batch_spikes.permute(1, 0, 2, 3).squeeze(2)
            batch_labels = batch_labels.long()

            model.reset_state(batch_spikes.shape[1])
            output = model(batch_spikes)  # [B, 10]
            predicted = torch.argmax(output, dim=1)

            print("True labels     :", batch_labels.numpy())
            print("Predicted labels:", predicted.numpy())
            print("Spike counts    :\n", output.int())
            print(f"Accuracy: {((predicted == batch_labels).sum().item() / batch_labels.shape[0]) * 100:.2f}%\n")

