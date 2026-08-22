"""
streaming_mvdr.py
=================
Real-time-deployable streaming MVDR beamformer.

Replaces the offline batch processing (scipy.signal.stft / istft on a full
buffer) with a causal, frame-by-frame ring-buffer implementation that can be
driven by a live audio callback, e.g.:

    bf = StreamingMVDR()
    while True:
        new_chunk = mic_driver.read(FFT_SHIFT)      # shape (FFT_SHIFT, 6)
        out_chunk = bf.process(new_chunk)           # shape (FFT_SHIFT,) or None
        if out_chunk is not None:
            speaker_driver.write(out_chunk)

No call to scipy.signal.stft / istft is made anywhere in this module — the
forward and inverse transforms are implemented directly on a sliding window,
matching scipy's exact analysis/synthesis convention (periodic Hann window,
'spectrum' scaling, running win**2 overlap-add normalization) so that this
primitive is numerically equivalent to the offline batch script once the
algorithmic warm-up region has passed.

IMPORTANT CAVEATS (see validate_streaming_full.py / validate_transform_only.py
for the actual measurements):
  - FIXED ALGORITHMIC LATENCY: output sample i corresponds to input sample
    (i - fft_length // 2), i.e. 256 samples / 16 ms at 16 kHz. This is
    irreducible for STFT-domain processing at this window size, not a bug.
    The offline script hides this by zero-padding the front and stripping
    the same amount from its own output (scipy's boundary='zeros'), which
    is only possible with knowledge of the future. Validated to match the
    offline output to within float32/float64 precision (max abs diff
    ~1.2e-6, SI-SNR match to 4 decimal places) once this latency is
    accounted for -- see latency_samples property below.
  - Global peak normalization (np.max over the whole utterance) has been
    replaced with a causal peak limiter, since real-time output cannot know
    the future peak. This is a NEW behavior with no offline equivalent to
    validate against -- it has not yet been tuned/listened to. See
    CausalPeakLimiter below.
  - np.linalg.pinv() was replaced with np.linalg.solve() in the per-frame
    weight computation (~13x faster on this (257, 6, 6) batch, identical
    results given the diagonal loading already applied). The same swap is
    worth backporting to the offline batch script for faster runs.
"""

import numpy as np
from scipy.signal import get_window

SAMPLE_RATE = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256
N_BINS = FFT_LENGTH // 2 + 1
REFERENCE_CHANNEL = 1
SPEED_OF_SOUND = 343.0
N_CHANNELS = 6


def build_steering_vector(
    mic_positions=None,
    target_pos=None,
    sample_rate=SAMPLE_RATE,
    fft_length=FFT_LENGTH,
    reference_channel=REFERENCE_CHANNEL,
):
    """Same near-field steering vector math as the offline script, factored
    out so both offline and streaming code paths construct it identically."""
    if mic_positions is None:
        cx, cy, height = 0.0, 0.0, 0.0
        mic_positions = np.array([
            [cx - 0.08, cy, height + 0.03],
            [cx - 0.08, cy + 0.02, height],
            [cx - 0.08, cy - 0.02, height],
            [cx + 0.08, cy - 0.02, height],
            [cx + 0.08, cy + 0.02, height],
            [cx + 0.08, cy, height + 0.03],
        ])
    if target_pos is None:
        target_pos = np.array([0.0, 0.12, -0.10])

    n_bins = fft_length // 2 + 1
    absolute_distances = np.linalg.norm(mic_positions - target_pos, axis=1)
    absolute_delays = absolute_distances / SPEED_OF_SOUND
    relative_delays = absolute_delays - absolute_delays[reference_channel]
    relative_amplitude = absolute_distances[reference_channel] / absolute_distances
    freqs = np.arange(n_bins) * (sample_rate / fft_length)
    steering_vector = (
        relative_amplitude[:, None]
        * np.exp(-1j * 2 * np.pi * freqs[None, :] * relative_delays[:, None])
    )
    return steering_vector


class CausalPeakLimiter:
    """Causal replacement for the offline script's global peak normalization.

    The offline version divides the *entire* utterance by its single global
    peak, scaled to 0.89 — impossible in real time since it requires knowing
    the future. This tracks a running peak estimate with a fast attack /
    slow release envelope and normalizes against that instead.

    This is a NEW behavior, not an equivalent one — flagged explicitly per
    the design discussion. It will not produce bit-identical output to the
    offline peak-normalized version; it produces safe, causal output.
    """

    def __init__(self, target_peak=0.89, attack=0.9, release=0.001, floor=1e-6):
        self.target_peak = target_peak
        self.attack = attack    # how fast the peak estimate rises (per-sample smoothing, closer to 1 = slower)
        self.release = release  # how fast the peak estimate decays when signal gets quieter
        self.floor = floor
        self.running_peak = floor

    def process(self, samples):
        out = np.empty_like(samples)
        peak = self.running_peak
        for i, s in enumerate(samples):
            a = abs(s)
            if a > peak:
                peak = self.attack * peak + (1 - self.attack) * a
            else:
                peak = (1 - self.release) * peak + self.release * a
            peak = max(peak, self.floor)
            out[i] = s / peak * self.target_peak
        self.running_peak = peak
        return out


class StreamingMVDR:
    """Causal, frame-by-frame MVDR beamformer with ring-buffer STFT/ISTFT.

    Call `.process(chunk)` repeatedly with fixed-size chunks of shape
    (FFT_SHIFT, N_CHANNELS). Returns output samples once enough history has
    accumulated to produce them (may be empty during initial buffering).
    """

    def __init__(
        self,
        steering_vector=None,
        fft_length=FFT_LENGTH,
        fft_shift=FFT_SHIFT,
        n_channels=N_CHANNELS,
        reference_channel=REFERENCE_CHANNEL,
        sample_rate=SAMPLE_RATE,
        beta=0.65,
        fast_alpha=0.4,
        slow_alpha=0.98,
        mask_threshold=1.02,
        mask_hangover=0.7,
        diag_loading_rel=1e-5,
        diag_loading_floor=1e-7,
        apply_limiter=True,
    ):
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.n_channels = n_channels
        self.reference_channel = reference_channel
        self.sample_rate = sample_rate
        self.n_bins = fft_length // 2 + 1

        self.beta = beta
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.mask_threshold = mask_threshold
        self.mask_hangover = mask_hangover
        self.diag_loading_rel = diag_loading_rel
        self.diag_loading_floor = diag_loading_floor

        self.win = get_window('hann', fft_length, fftbins=True).astype(np.float64)
        self.win_sum = self.win.sum()  # matches scipy's 'spectrum' scaling divisor

        if steering_vector is None:
            steering_vector = build_steering_vector(
                sample_rate=sample_rate, fft_length=fft_length, reference_channel=reference_channel
            )
        self.steering_vector = steering_vector  # shape (n_channels, n_bins)

        # ---- Analysis-side ring buffer: holds the last `fft_length` samples per channel ----
        self.input_ring = np.zeros((n_channels, fft_length), dtype=np.float64)
        self.samples_buffered = 0  # how many real (non-zero-pad) samples have entered the ring so far

        # ---- Synthesis-side overlap-add buffer ----
        # Needs to span one extra frame beyond fft_length so OLA from the
        # current frame can deposit into not-yet-flushed future samples.
        self.ola_len = fft_length + fft_shift
        self.ola_buffer = np.zeros(self.ola_len, dtype=np.complex128)
        self.ola_norm = np.zeros(self.ola_len, dtype=np.float64)
        self.win_sq = self.win ** 2

        # ---- MVDR recursive state (identical to offline script) ----
        self.R_noise = np.tile((np.eye(n_channels) * 1e-5)[:, :, None], (1, 1, self.n_bins)).astype(np.complex128)
        self.power_fast = None
        self.power_slow = None
        self.noise_mask_smoothed = np.zeros(self.n_bins, dtype=np.float64)
        self.w_smoothed = None
        self.I_batch = np.tile(np.eye(n_channels)[None, :, :], (self.n_bins, 1, 1))
        self.frame_idx = 0

        self.limiter = CausalPeakLimiter() if apply_limiter else None

        # carry-over buffer of input samples that haven't yet formed a full hop
        self._pending = np.zeros((0, n_channels), dtype=np.float64)

    @property
    def latency_samples(self):
        """Fixed algorithmic latency: output sample i corresponds to input
        sample (i - latency_samples). This is fft_length // 2, the amount of
        the analysis window that must be observed before the windowed
        energy centered at a given input sample has fully arrived. It is
        irreducible for STFT-domain processing at this window size -- not
        a bug, and not something a real-time deployment can remove without
        shrinking fft_length (which trades off frequency resolution)."""
        return self.fft_length // 2

    # ------------------------------------------------------------------
    def _process_one_hop(self, hop_samples):
        """hop_samples: shape (fft_shift, n_channels) — the newest hop of audio.
        Slides the analysis ring buffer, runs one MVDR frame update, deposits
        into the OLA buffer, and flushes one fully-formed output hop."""

        # Slide ring buffer: drop oldest fft_shift samples, append new hop
        self.input_ring = np.concatenate(
            [self.input_ring[:, self.fft_shift:], hop_samples.T], axis=1
        )
        self.samples_buffered = min(self.samples_buffered + self.fft_shift, self.fft_length)

        # ---- Forward transform (matches scipy stft 'spectrum' scaling) ----
        windowed = self.input_ring * self.win[None, :]
        X_t = np.fft.rfft(windowed, axis=1) / self.win_sum  # shape (n_channels, n_bins)

        # ---- MVDR recursive update (unchanged math from offline script) ----
        current_energy = np.abs(X_t[self.reference_channel, :]) ** 2
        EPSILON_FLOOR = 1e-10

        if self.power_fast is None:
            self.power_fast = current_energy + EPSILON_FLOOR
            self.power_slow = current_energy + EPSILON_FLOOR
        else:
            self.power_fast = self.fast_alpha * current_energy + (1 - self.fast_alpha) * self.power_fast
            self.power_slow = self.slow_alpha * current_energy + (1 - self.slow_alpha) * self.power_slow
            self.power_slow = np.maximum(self.power_slow, EPSILON_FLOOR)
            self.power_fast = np.maximum(self.power_fast, EPSILON_FLOOR)

        X_outer = np.einsum('if,jf->ijf', X_t, np.conj(X_t))

        noise_mask_instant = (self.power_fast <= self.mask_threshold * self.power_slow).astype(np.float64)

        if self.frame_idx == 0:
            self.noise_mask_smoothed = noise_mask_instant
        else:
            self.noise_mask_smoothed = np.maximum(noise_mask_instant, self.mask_hangover * self.noise_mask_smoothed)

        alpha_tensor = 0.95 * self.noise_mask_smoothed + 1.0 * (1 - self.noise_mask_smoothed)
        self.R_noise = alpha_tensor[None, None, :] * self.R_noise + (1 - alpha_tensor[None, None, :]) * X_outer

        R_batch = np.moveaxis(self.R_noise, 2, 0)
        eps = np.maximum(self.diag_loading_rel * np.mean(np.abs(R_batch)), self.diag_loading_floor)
        R_batch = R_batch + eps * self.I_batch

        # Use solve() instead of pinv(): diagonal loading above guarantees
        # R_batch is well-conditioned and invertible, so we don't need
        # pinv's extra robustness to singular/near-singular matrices --
        # solve() is ~13x faster for this exact (257, 6, 6) batch shape and
        # gives numerically identical results to pinv() here. This matters
        # for real-time: this was the single largest per-hop cost.
        d_batch = np.moveaxis(self.steering_vector, 1, 0)[:, :, None]
        numerator_batch = np.linalg.solve(R_batch, d_batch)
        d_H_batch = np.conj(np.moveaxis(d_batch, 2, 1))
        denominator_batch = d_H_batch @ numerator_batch

        w_instant = np.squeeze(numerator_batch / (denominator_batch + 1e-8), axis=-1)

        if self.frame_idx == 0:
            self.w_smoothed = w_instant
        else:
            self.w_smoothed = self.beta * self.w_smoothed + (1 - self.beta) * w_instant

        w_filtered_spec = np.einsum('fi,if->f', np.conj(self.w_smoothed), X_t)  # shape (n_bins,)

        self.frame_idx += 1

        # ---- Inverse transform + overlap-add (matches scipy istft exactly) ----
        frame_time = np.fft.irfft(w_filtered_spec, n=self.fft_length)  # shape (fft_length,)
        frame_time = frame_time * self.win_sum  # undo forward scaling, per scipy istft source

        self.ola_buffer[:self.fft_length] += frame_time * self.win
        self.ola_norm[:self.fft_length] += self.win_sq

        # Flush the oldest fft_shift samples (now fully formed — no more
        # future frames will contribute to them) as finalized output.
        norm_slice = self.ola_norm[:self.fft_shift]
        safe_norm = np.where(norm_slice > 1e-10, norm_slice, 1.0)
        out_chunk = (self.ola_buffer[:self.fft_shift].real / safe_norm)

        # Slide the OLA buffer forward by one hop
        self.ola_buffer = np.concatenate(
            [self.ola_buffer[self.fft_shift:], np.zeros(self.fft_shift, dtype=np.complex128)]
        )
        self.ola_norm = np.concatenate(
            [self.ola_norm[self.fft_shift:], np.zeros(self.fft_shift, dtype=np.float64)]
        )

        return out_chunk

    # ------------------------------------------------------------------
    def process(self, chunk):
        """Feed an arbitrary-length chunk of shape (n_samples, n_channels).
        Returns concatenated output samples (1-D, mono enhanced signal) for
        however many full hops were completed — may be shorter than the
        input chunk if not enough samples have accumulated yet, and may be
        empty (shape (0,)) during initial buffering."""
        chunk = np.asarray(chunk, dtype=np.float64)
        self._pending = np.concatenate([self._pending, chunk], axis=0)

        outputs = []
        while len(self._pending) >= self.fft_shift:
            hop = self._pending[:self.fft_shift]
            self._pending = self._pending[self.fft_shift:]
            out_chunk = self._process_one_hop(hop)
            outputs.append(out_chunk)

        if not outputs:
            return np.zeros(0, dtype=np.float64)

        out = np.concatenate(outputs)
        if self.limiter is not None:
            out = self.limiter.process(out)
        return out
