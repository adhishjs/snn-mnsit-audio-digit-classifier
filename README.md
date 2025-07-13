# 🧠🔊 Spiking Neural Network for Audio MNIST Digit Classification

This project implements a Spiking Neural Network (SNN) to classify spoken digits (0–9) using audio recordings. The model is trained and evaluated using MFCC features extracted from audio samples — a biologically inspired approach leveraging temporal dynamics!

## 📁 Project Structure

<pre>

AUDIO/
├── DATASET/              # (Optional) Original dataset folder
├── DATASET_TEST/         # Test samples (preprocessed)
├── SNN_network.py        # Simple SNN architecture
├── run_digit.py          # Script to evaluate test audio digits
├── train_digit.py        # Script to train the model
├── ds_loader.py          # Data loading and preprocessing logic
├── encoder.py            # Audio to spike encoding logic
├── loading.py            # Utility to load model checkpoints
├── saving.py             # Utility to save model checkpoints
├── audio_rec.py          # Optional audio recording
├── parameters.pt         # Trained model parameters

</pre>

🚀 How to Run

1. 📦 Requirements
   Install the required Python packages:

```bash
pip install torch torchaudio numpy
```

2. 🎧 Dataset
   Ensure you have the preprocessed MFCC dataset in the DATASET_TEST/ folder. Each file should be an audio sample corresponding to a digit from 0 to 9.

For Dataset,

https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data



4. 🏁 Run Inference
   To test the model on random 10 audio samples:

```bash
python run_digit.py
```

4. 🧠 Train the Model
   If you want to train your own model from scratch:

```bash
python train_digit.py
```

The trained model will be saved as parameters.pt.


🔍 Model Overview
<pre>
The model is a SimpleSNN:

Input: MFCC features → Spike Trains

Architecture: Feedforward spiking layers

Activation: Leaky Integrate-and-Fire (LIF) dynamics

Output: 10-class softmax layer (digit prediction)
</pre>


The model uses a biologically inspired temporal encoding mechanism.

encoder.py handles converting audio features into spike trains.

You can record new audio with audio_rec.py and test it live.

🧠 Inspiration
   This project blends neuroscience with modern machine learning — showing how SNNs can interpret temporal signals like speech.



Made by Adhish J S

For Contact,
adhishjs05@gmail.com 
