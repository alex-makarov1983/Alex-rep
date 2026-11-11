"""Проверка поиска щелчков: синтетический звук с ударами в известные моменты.
Запуск: python3 -m unittest discover coach/test"""
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cut_strokes  # noqa: E402


class AudioHits(unittest.TestCase):
    def test_finds_clicks_over_noise(self):
        sr, dur = 16000, 12.0
        rng = np.random.default_rng(1)
        t = np.arange(int(sr * dur)) / sr
        # Фон: ветер (низкий гул) и тихий шум, как на корте.
        x = 0.05 * np.sin(2 * np.pi * 90 * t) + 0.005 * rng.standard_normal(len(t))
        clicks = [1.5, 3.1, 4.4, 7.25, 9.8]
        for c in clicks:
            i = int(c * sr)
            n = int(0.004 * sr)
            x[i:i + n] += 0.6 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.001 * sr))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "court.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            found = [h for h, _ in cut_strokes.audio_hits(p, 6.0)]
        self.assertEqual(len(found), len(clicks), found)
        for a, b in zip(found, clicks):
            self.assertLess(abs(a - b), 0.02)


if __name__ == "__main__":
    unittest.main()
