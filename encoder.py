import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# === MFCC Extraction ===
def extract_mfcc(filepath, n_mfcc=20, sample_rate=16000):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc,
        sample_rate=sample_rate,
        
        melkwargs={"n_mels": 64}
    )(waveform)
    # print(mfcc.shape)  # Debugging line to check shape
    # print("hi")
    # print(mfcc.squeeze(0).transpose(0, 1).shape)  # Debugging line to check shape after squeeze and transpose
    # print("hi2")
    return mfcc.squeeze(0).transpose(0, 1)  # shape: [T, n_mfcc]

# === Normalization ===
def normalize_per_feature(mfcc_tensor):
    min_vals = mfcc_tensor.min(dim=0, keepdim=True).values
    max_vals = mfcc_tensor.max(dim=0, keepdim=True).values
    norm = (mfcc_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return norm

# === Rate Encoding ===
def rate_encode_100_steps(norm_mfcc):
    T, F = norm_mfcc.shape
    spike_trains = [(torch.rand(10, F) < norm_mfcc[t]).float() for t in range(T)]
    spikes_cat = torch.cat(spike_trains, dim=0)  # shape: [100*T, F]
    return spikes_cat.squeeze(0).unsqueeze(1)  # shape: [100*T, 1, F]



class AudioDigitDataset(Dataset):
    def __init__(self, root_dir, n_mfcc=20, sample_rate=16000, time_steps=100):
        self.root_dir = root_dir
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.time_steps = time_steps
        self.filepaths = []
        self.labels = []

        # Collect all file paths and labels
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith(".wav"):
                        self.filepaths.append(os.path.join(folder_path, fname))
                        label = int(fname.split("_")[0])  # Extract digit label
                        self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        mfcc = extract_mfcc(filepath, self.n_mfcc, self.sample_rate)
        norm_mfcc = normalize_per_feature(mfcc)

        # Pad or truncate to fixed length T
        T_target = 80
        T_current = norm_mfcc.shape[0]
        
        if T_current < T_target:
            pad_amount = T_target - T_current
            pad_tensor = torch.zeros(pad_amount, norm_mfcc.shape[1])
            norm_mfcc = torch.cat([norm_mfcc, pad_tensor], dim=0)
        elif T_current > T_target:
            norm_mfcc = norm_mfcc[:T_target]

        # Now norm_mfcc is [100, F]
        spikes = rate_encode_100_steps(norm_mfcc)  # shape: [10000, F]

        return spikes, label





# mel=extract_mfcc("DATASET/03/4_03_43.wav")
# norm_mel=normalize_per_feature(mel)
# spikes=rate_encode_100_steps(norm_mel)

# print(spikes.shape)  # Should print: torch.Size([100*T, 1, n_mfcc])

# dataset = AudioDigitDataset(root_dir="DATASET", n_mfcc=20, sample_rate=16000, time_steps=100)
# dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


# l=0
# # Example usage
# for batch_spikes, batch_labels in dataloader:
#     batch_spikes = batch_spikes.permute(1, 0,2, 3).squeeze(2)
#     print(batch_labels)
#     print(batch_spikes.shape)  # [100*T, batch_size, n_mfcc]
#     print(batch_labels.shape)  # [batch_size]
#     # print(batch_spikes[:,0,:])
#     l+=1
#     if l==5:

#         break
