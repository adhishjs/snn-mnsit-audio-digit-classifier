import torch
import torch.nn as nn

from LIFneuron import LIFNeuronLayer
# print(__name__)
# === Full Inference Model ===
class SimpleSNN(nn.Module):
    print(__name__)
    def __init__(self):
        super().__init__()
        self.l1 = LIFNeuronLayer(20, 32)
        self.l2 = LIFNeuronLayer(32, 32)
        self.l3 = LIFNeuronLayer(32, 32)  # 🆕 New hidden layer
        self.l4 = LIFNeuronLayer(32, 10)  # Output layer

    def reset_state(self, batch_size):
        self.l1.reset_state(batch_size)
        self.l2.reset_state(batch_size)
        self.l3.reset_state(batch_size)
        self.l4.reset_state(batch_size)

    def forward(self, x_seq):
        # print(f"Input shape: {x_seq.shape}")
        spike_counts = torch.zeros(x_seq.shape[1], 10)
        # print(f"Spike counts shape: {spike_counts.shape}")
        for t in range(x_seq.shape[0]):
            # print(x_seq[t])
            h1 = self.l1(x_seq[t])
            h2 = self.l2(h1)
            h3 = self.l3(h2)
            o = self.l4(h3)
            spike_counts += o
        return spike_counts
    