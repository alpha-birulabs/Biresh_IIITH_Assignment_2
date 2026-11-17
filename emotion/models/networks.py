"""Shared encoders for the three modality comparisons."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from emotion.data_prep import ALPHABET, EMOTIONS


class SequenceEncoder(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.recurrent = nn.GRU(inputs, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden, 1)

    def forward(self, sequence, lengths):
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        states, _ = self.recurrent(packed)
        states, _ = pad_packed_sequence(states, batch_first=True)
        valid = torch.arange(states.size(1), device=states.device) < lengths.to(states.device)[:, None]
        scores = self.attention(states).squeeze(-1).masked_fill(~valid, -torch.inf)
        return (states * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)


class EmotionClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("speech", "text", "fusion"):
            raise ValueError(f"Unknown modality: {mode}")
        self.mode = mode
        if mode in ("speech", "fusion"):
            self.acoustic = nn.Sequential(nn.Linear(64, 48), nn.LayerNorm(48), nn.GELU())
            self.temporal = SequenceEncoder(48, 32)
        if mode in ("text", "fusion"):
            self.embedding = nn.Embedding(len(ALPHABET) + 2, 24, padding_idx=0)
            self.contextual = SequenceEncoder(24, 16)
        if mode == "fusion":
            self.fusion = nn.Sequential(nn.Linear(96, 64), nn.LayerNorm(64), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(32 if mode == "text" else 64, len(EMOTIONS)))

    def forward(self, speech, speech_lengths, text, text_lengths):
        if self.mode in ("speech", "fusion"):
            # Retain the 40 ms frame-level trajectory while reducing recurrent cost.
            acoustic = self.temporal(self.acoustic(speech[:, ::4]), (speech_lengths + 3) // 4)
        if self.mode in ("text", "fusion"):
            context = self.contextual(self.embedding(text), text_lengths)
        if self.mode == "fusion":
            representation = self.fusion(torch.cat((acoustic, context), dim=1))
        else:
            representation = acoustic if self.mode == "speech" else context
        return self.classifier(representation), representation
