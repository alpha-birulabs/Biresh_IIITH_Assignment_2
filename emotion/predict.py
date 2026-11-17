"""Predict an emotion from an audio file, text, or both."""

import argparse
import json
from pathlib import Path

import torch

from emotion.data_prep import EMOTIONS, HERE, encode_text, log_mel
from emotion.experiment import load_model


@torch.inference_mode()
def predict(mode, checkpoint_path, audio=None, text=None, device="cpu"):
    device = torch.device(device)
    model, checkpoint = load_model(mode, checkpoint_path, device)
    speech = speech_lengths = tokens = text_lengths = None
    if mode in ("speech", "fusion"):
        if audio is None:
            raise ValueError(f"{mode} prediction requires audio")
        frames = torch.from_numpy(log_mel(audio)).to(device)
        mean, std = checkpoint["stats"]
        speech = ((frames - mean) / std).unsqueeze(0)
        speech_lengths = torch.tensor([len(frames)])
    if mode in ("text", "fusion"):
        if text is None:
            raise ValueError(f"{mode} prediction requires text")
        tokens = torch.tensor([encode_text(text)], device=device)
        text_lengths = torch.tensor([tokens.shape[1]])
    logits, _ = model(speech, speech_lengths, tokens, text_lengths)
    probabilities = logits.softmax(dim=1)[0].cpu().tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {"mode": mode, "emotion": EMOTIONS[index], "confidence": probabilities[index],
            "probabilities": dict(zip(EMOTIONS, probabilities))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("speech", "text", "fusion"), required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode in ("speech", "fusion") and args.audio is None:
        parser.error("--audio is required for speech and fusion")
    if args.mode in ("text", "fusion") and (args.text is None or not args.text.strip()):
        parser.error("--text must contain non-whitespace characters for text and fusion")
    torch.set_num_threads(4)
    checkpoint = args.checkpoint or HERE / "checkpoints" / f"{args.mode}.pt"
    result = predict(args.mode, checkpoint, args.audio, args.text, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
