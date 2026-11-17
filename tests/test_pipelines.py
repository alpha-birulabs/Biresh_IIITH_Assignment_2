"""Regression checks for modality isolation, input validation and saved state."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from emotion.data_prep import encode_text, log_mel
from emotion.experiment import save_checkpoint, split_fingerprints
from emotion.models.networks import EmotionClassifier
from alignment.run import by_stem, extend_dictionary, normalize


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_predictions_ignore_padding_and_unused_modalities(self):
        torch.manual_seed(42)
        speech = torch.randn(1, 9, 64)
        text = torch.tensor([encode_text("say the word boat")])
        speech_lengths = torch.tensor([9])
        text_lengths = torch.tensor([text.size(1)])
        for mode in ("speech", "text", "fusion"):
            with self.subTest(mode=mode), torch.inference_mode():
                model = EmotionClassifier(mode).eval()
                audio = None if mode == "text" else speech
                audio_lengths = None if audio is None else speech_lengths
                tokens = None if mode == "speech" else text
                token_lengths = None if tokens is None else text_lengths
                baseline = model(audio, audio_lengths, tokens, token_lengths)[0]
                padded_audio = None if audio is None else torch.cat([audio, torch.randn(1, 7, 64)], dim=1)
                padded_text = None if tokens is None else torch.cat([tokens, torch.ones(1, 7, dtype=torch.long)], dim=1)
                padded = model(padded_audio, audio_lengths, padded_text, token_lengths)[0]
                torch.testing.assert_close(baseline, padded)

    def test_audio_and_text_validation(self):
        self.assertEqual(encode_text(" SAY  THE word boat\n"), encode_text("say the word boat"))
        with self.assertRaises(ValueError):
            encode_text("  ")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            for samples in (np.array([]), np.zeros(1000), np.full(1000, np.nan)):
                sf.write(path, samples, 16000, subtype="FLOAT")
                with self.assertRaises(ValueError):
                    log_mel(path)
            # Short stereo input at a different sampling rate exercises resampling and padding.
            samples = np.sin(np.arange(100) * 0.2)
            sf.write(path, np.column_stack([samples, samples]), 22050, subtype="FLOAT")
            features = log_mel(path)
            self.assertEqual(features.shape[1], 64)
            self.assertTrue(np.isfinite(features).all())

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint({"epoch": 1}, path)
            original = path.read_bytes()
            with patch("emotion.experiment.torch.save", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    save_checkpoint({"epoch": 2}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_split_overlap_is_rejected_and_content_changes_are_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, word in (("train", "boat"), ("validation", "cat"), ("test", "dog")):
                (root / f"{split}.json").write_text(json.dumps([{"id": split, "word": word}]))
                np.savez(root / f"{split}.npz", frames=np.zeros((1, 64)))
            original = split_fingerprints(root)
            np.savez(root / "test.npz", frames=np.ones((1, 64)))
            self.assertNotEqual(original, split_fingerprints(root))
            (root / "test.json").write_text(json.dumps([{"id": "test", "word": "boat"}]))
            with self.assertRaisesRegex(ValueError, "leakage"):
                split_fingerprints(root)

    def test_alignment_transcript_pairing_and_normalization(self):
        self.assertEqual(normalize("S.J.C.'s WBUR's seventy- six"), "s j c's w b u r's seventy six")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.TXT").write_text("words")
            (root / "ignored.csv").write_text("other")
            self.assertEqual(list(by_stem(root, ".txt")), ["sample"])

    def test_dictionary_additions_keep_existing_words_and_the_phone_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Pretrained dictionaries may carry probability columns between the word and its phones.
            (root / "base.dict").write_text("said\t0.99\t0.2\t1.0\t1.0\tS EH1 D\nbait\tB EY1 T\n", encoding="utf-8")
            (root / "new.tsv").write_text("said\tS EY1 D\nbaited\tB EY1 T EH1 D\n", encoding="utf-8")
            extended = extend_dictionary(root / "base.dict", root / "new.tsv", root / "out" / "extended.dict")
            self.assertEqual(extended.read_text(encoding="utf-8").splitlines()[2:], ["baited\tB EY1 T EH1 D"])
            (root / "new.tsv").write_text("zoo\tZ UW1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported phones"):
                extend_dictionary(root / "base.dict", root / "new.tsv", root / "out" / "extended.dict")

    def test_added_noise_is_reproducible_and_leaves_clean_features_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            sf.write(path, 0.5 * np.sin(np.arange(8000) * 0.1), 16000, subtype="FLOAT")
            clean = log_mel(path)
            np.testing.assert_array_equal(clean, log_mel(path))
            noisy = log_mel(path, 10, np.random.default_rng(0))
            np.testing.assert_array_equal(noisy, log_mel(path, 10, np.random.default_rng(0)))
            self.assertGreater(noisy.mean(), clean.mean())


if __name__ == "__main__":
    unittest.main()
