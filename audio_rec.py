import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(filename="my_audio.wav", duration=1, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"Saved recording to '{filename}'")



if __name__ == "__main__":
    audio_file = "DATASET/TEST/01/3_01_03.wav"
    record_audio(filename=audio_file, duration=1, sample_rate=16000)