"""Software echo canceller (SOFT_AEC=1, raawr US-060, path B)."""
import numpy as np
import pytest

from app.mjuk_aec import FRAME, RATE, MjukAec

pytest.importorskip("aec_audio_processing")


def _tal(sek, seed=1):
    """Speech-like: band-limited noise with syllable-rate amplitude changes."""
    r = np.random.default_rng(seed)
    n = int(sek * RATE)
    x = np.convolve(r.normal(0, 1, n), np.ones(4) / 4, "same")
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * np.arange(n) / RATE) ** 2
    return (x * env * 6000).astype(np.int16)


def _eko(ref, tap_ms=(150, 190, 240), amp=(0.5, 0.25, 0.12)):
    ut = np.zeros(len(ref) + RATE, dtype=np.float32)
    for ms, g in zip(tap_ms, amp):
        d = int(ms * RATE / 1000)
        ut[d:d + len(ref)] += g * ref
    return ut[:len(ref)].astype(np.int16)


def _kor(aec_pa, sek=12.0, forst=0.02, vid=0.0):
    ref = _tal(sek)
    mic = _eko(ref)
    klock = [0.0]
    aec = MjukAec(klocka=lambda: klock[0]) if aec_pa else None
    ut, steg = [], int(forst * RATE)
    for i in range(0, len(ref) - steg, steg):
        t = i / RATE
        klock[0] = t
        if aec:
            aec.ref(ref[i:i + steg].tobytes(), RATE, t)                 # sent now, played later
            klock[0] = t + forst
            ut.append(aec.process(mic[i:i + steg].tobytes(), t + forst))
        else:
            ut.append(mic[i:i + steg].tobytes())
    return mic, np.frombuffer(b"".join(ut), dtype=np.int16)


def _db(x):
    return 20 * np.log10(max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-3) / 32768)


def test_the_echo_is_taken_down_when_the_reference_leads():
    mic, ut = _kor(True)
    a, b = 6 * RATE, 11 * RATE
    erle = _db(mic[a:b]) - _db(ut[a - FRAME * 3:b - FRAME * 3])
    assert erle >= 12, f"only {erle:.1f} dB"


def test_it_is_only_off_by_flag(monkeypatch):
    monkeypatch.delenv("SOFT_AEC", raising=False)
    assert MjukAec.skapa() is None
    monkeypatch.setenv("SOFT_AEC", "1")
    assert MjukAec.skapa() is not None


def test_no_reference_and_agc_off_means_the_mic_passes_untouched():
    aec = MjukAec(klocka=lambda: 100.0)
    aec.agc_max_db = 0
    pcm = _tal(0.1).tobytes()
    assert aec.process(pcm, 100.0) == pcm


def test_a_failure_switches_it_off_and_the_mic_goes_on():
    aec = MjukAec(klocka=lambda: 0.0)
    aec.ref(_tal(0.1).tobytes(), RATE, 0.0)
    aec.ap = None   # anything that throws
    pcm = _tal(0.05).tobytes()
    assert aec.process(pcm, 0.05) == pcm and aec.trasig


def _ram_db(x):
    return _db(x)


def test_the_agc_lifts_distant_speech_but_not_the_noise_floor():
    r = np.random.default_rng(5)
    brus = (r.normal(0, 1, 3 * RATE) * 10 ** (-50 / 20) * 32768).astype(np.int16)     # floor -50 dBFS
    tal = np.concatenate([_tal(2.0, seed=9) * 0.2, np.zeros(RATE // 2, dtype=np.int16)])   # far away: ~-36 dBFS
    tal = np.pad(tal, (0, max(0, len(brus) - len(tal))))[:len(brus)]
    aec = MjukAec(klocka=lambda: 0.0)
    ut_brus = np.frombuffer(aec.process(brus.tobytes(), 3.0), dtype=np.int16)
    assert _db(ut_brus) < -46, "noise alone must not be lifted"
    mix = (brus.astype(np.int32) + tal).clip(-32768, 32767).astype(np.int16)
    aec2 = MjukAec(klocka=lambda: 0.0)
    ut = np.frombuffer(aec2.process(mix.tobytes(), 3.0), dtype=np.int16)
    assert _db(ut[RATE:2 * RATE]) > _db(mix[RATE:2 * RATE]) + 6, "distant speech should be lifted"
    aec3 = MjukAec(klocka=lambda: 0.0)
    aec3.agc_max_db = 0
    assert np.array_equal(np.frombuffer(aec3.process(mix.tobytes(), 3.0), dtype=np.int16)[:RATE], mix[:RATE])
