import torch
import torch.nn as nn


# === LIF Inference Layer ===
class LIFNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.threshold = nn.Parameter(torch.rand(output_size) * 1.0)
        self.decay = nn.Parameter(torch.rand(output_size) * 0.5 + 0.4)
        self.v = None

    def reset_state(self, batch_size):
        self.v = torch.zeros(batch_size, self.fc.out_features)

    def forward(self, x):
        if self.v is None or self.v.shape[0] != x.shape[0]:
            self.reset_state(x.shape[0])
        self.v = self.v * self.decay + self.fc(x)
        spikes = torch.sigmoid(5 * (self.v - self.threshold))  # surrogate gradient
        self.v = self.v * (1.0 - spikes)
        return spikes