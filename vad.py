"""
vad.py — RMS Energy-based Voice Activity Detection

Silero VAD doesn't work with Google Meet audio from Recall.ai
(WebRTC processing alters spectral characteristics → max conf 0.066).

RMS energy detection works perfectly:
  Speech RMS: 0.01 - 0.35
  Silence RMS: 0.00001
  Separation: 1000x+

No model download, no ONNX, no PyTorch. Just numpy math.
Processes 32ms audio chunks, same interface as Silero version.
"""

import time
import numpy as np


class RmsVAD:
    """RMS energy-based Voice Activity Detection.
    
    Drop-in replacement for SileroVAD with identical interface.
    Calibrated on actual Recall.ai Google Meet audio.
    """

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512          # 32ms at 16kHz (same as Silero)
    RMS_THRESHOLD = 0.01         # RMS above this = speech (ambient noise sits at 0.003-0.008)
    SILENCE_FLOOR = 0.0003       # Below this = absolute silence
    DIRECT_SPEECH_RMS = 0.06     # Peak RMS must exceed this for flush (headset mic: 0.08+, background: <0.05)

    def __init__(self):
        self._ready = False
        self._audio_buffer = np.array([], dtype=np.float32)

        # State tracking (same interface as SileroVAD)
        self.is_speaking = False
        self.speech_start = 0.0
        self.silence_start = 0.0
        self.last_confidence = 0.0     # stores RMS value (0.0-1.0)
        self.heard_speech = False      # sticky flag — True until end_turn()
        self.last_speech_time = 0.0    # when RMS last exceeded threshold
        self.peak_rms = 0.0            # highest RMS since last end_turn()

    async def setup(self):
        """No model to download — just mark ready."""
        self._ready = True
        print("[VAD] ✅ RMS energy VAD ready (no model needed, ~0ms/chunk)")

    def process_chunk(self, pcm_bytes: bytes) -> list[float]:
        """Feed raw PCM bytes (16kHz S16LE mono). Returns list of RMS values."""
        if not self._ready:
            return []

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer = np.concatenate([self._audio_buffer, samples])

        rms_values = []
        while len(self._audio_buffer) >= self.CHUNK_SAMPLES:
            chunk = self._audio_buffer[:self.CHUNK_SAMPLES]
            self._audio_buffer = self._audio_buffer[self.CHUNK_SAMPLES:]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_values.append(rms)

        return rms_values

    def update_state(self, rms: float, threshold: float = None):
        """Update speaking/silence tracking from RMS value.
        
        Args:
            rms: RMS energy level (0.0-1.0)
            threshold: Ignored — uses calibrated RMS_THRESHOLD
        """
        thresh = self.RMS_THRESHOLD
        self.last_confidence = rms
        now = time.time()

        if rms >= thresh:
            # Speech energy detected
            self.heard_speech = True
            self.last_speech_time = now
            self.silence_start = 0.0
            # Track loudest sound this turn — used to filter background voices
            if rms > self.peak_rms:
                self.peak_rms = rms
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start = now
                print(f"[VAD] 🎙️ Speech detected (rms={rms:.4f})")
        else:
            # Below threshold — track silence
            if self.heard_speech and self.silence_start == 0.0:
                self.silence_start = now
            # Reset is_speaking after 3s of silence (but keep heard_speech!)
            if self.is_speaking and self.silence_start > 0.0:
                if (now - self.silence_start) > 3.0:
                    self.is_speaking = False

    def silence_duration_ms(self) -> float:
        """Milliseconds of continuous silence. 0 if no silence started."""
        if not self.heard_speech or self.silence_start == 0.0:
            return 0.0
        return (time.time() - self.silence_start) * 1000

    def silence_since_last_speech_ms(self) -> float:
        """Milliseconds since last speech detected."""
        if not self.heard_speech or self.last_speech_time == 0.0:
            return 0.0
        if self.silence_start == 0.0:
            return 0.0
        return (time.time() - self.last_speech_time) * 1000

    def end_turn(self):
        """Mark turn as finished. Call after flushing buffer."""
        self.is_speaking = False
        self.silence_start = 0.0
        self.speech_start = 0.0
        self.heard_speech = False
        self.last_speech_time = 0.0
        self.peak_rms = 0.0

    @property
    def is_direct_speech(self) -> bool:
        """True if peak RMS this turn exceeded direct-speech threshold.
        Background voices: peak 0.01-0.02, direct mic: peak 0.03+"""
        return self.peak_rms >= self.DIRECT_SPEECH_RMS

    def reset(self):
        """Full reset — call between meetings."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self.end_turn()
        self.last_confidence = 0.0

    @property
    def ready(self) -> bool:
        return self._ready