#!/usr/bin/env python3

import torch
import numpy as np
from scipy.io.wavfile import write


def talk(text: str = "beep beep boop boop"):
    waveglow = torch.hub.load("nvidia/DeepLearningExamples:torchhub", "nvidia_waveglow")
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to("cpu")
    waveglow.eval()
    tacotron2 = torch.hub.load(
        "nvidia/DeepLearningExamples:torchhub", "nvidia_tacotron2"
    )
    tacotron2 = tacotron2.to("cpu")
    tacotron2.eval()
    # preprocessing
    # preprocessing
    sequence = np.array(tacotron2.text_to_sequence(text, ["english_cleaners"]))[None, :]
    sequence = torch.from_numpy(sequence).to(device="cpu", dtype=torch.int64)

    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    write("audio.wav", rate, audio_numpy)
    return audio


if __name__ == "__main__":
    talk("Hello how are you today?")
