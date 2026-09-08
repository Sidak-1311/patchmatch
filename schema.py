"""Single source of truth for the v1 parameter set.

Ranges are sampling bounds, not clamps. Values outside them are legal
in Vital — they just produce sounds nobody would ask a matcher to find
(silence, envelopes that never open). The label stores the raw
normalized value, so predictions go straight back into the plugin.
"""

import random

# Always set, never varied.
FIXED = {
    384: 1.0,   # Oscillator 1 Switch — on
    114: 1.0,   # Filter 1 Switch — on
    106: 1.0,   # Filter 1 FILTER Input — osc 1 routes into filter 1
}

# (index, name, lo, hi) — sampled uniformly in [lo, hi]. Regression outputs.
CONTINUOUS = [
    (382, "Oscillator 1 Level",            0.55, 1.00),  # dB curve: below ~0.5 is inaudible
    (391, "Oscillator 1 Transpose",        0.00, 1.00),
    (393, "Oscillator 1 Tune",             0.00, 1.00),
    (397, "Oscillator 1 Wave Frame",       0.00, 1.00),
    (386, "Oscillator 1 Phase",            0.00, 1.00),
    (385, "Oscillator 1 Pan",              0.00, 1.00),
    (395, "Oscillator 1 Unison Detune",    0.00, 1.00),
    (377, "Oscillator 1 Distortion Amount",0.00, 1.00),
    (104, "Filter 1 Cutoff",               0.15, 1.00),  # below ~0.15 kills a 261 Hz note
    (115, "Filter 1 Resonance",            0.00, 1.00),
    (105, "Filter 1 Drive",                0.00, 1.00),
    (102, "Filter 1 Blend",                0.00, 1.00),
    (112, "Filter 1 Mix",                  0.00, 1.00),
    (111, "Filter 1 Key Track",            0.00, 1.00),
    (48,  "Envelope 1 Attack",             0.00, 0.70),  # near 1.0 never opens in 3 s
    (50,  "Envelope 1 Decay",              0.00, 1.00),
    (54,  "Envelope 1 Sustain",            0.00, 1.00),
    (52,  "Envelope 1 Release",            0.00, 1.00),
    (49,  "Envelope 1 Attack Power",       0.00, 1.00),
    (51,  "Envelope 1 Decay Power",        0.00, 1.00),
    (53,  "Envelope 1 Release Power",      0.00, 1.00),
    (536, "Envelope 1 Delay",              0.00, 0.30),  # high delay + attack = silence
    (542, "Envelope 1 Hold",               0.00, 0.50),
]

# (index, name, [(raw_value, label), ...])
# Raw values measured from the plugin by sweeping and taking each
# option band's midpoint. Not derived from a formula — the spacing
# is uneven (Distortion Type is 1/12 then 1/8).
# List position is the class index the model predicts.
CATEGORICAL = [
    (113, "Filter 1 Model", [
        (0.03550, "Analog"),
        (0.14275, "Dirty"),
        (0.28575, "Ladder"),
        (0.42850, "Digital"),
        (0.57125, "Diode"),
        (0.71425, "Formant"),
        (0.85725, "Comb"),
        (0.96450, "Phaser"),
    ]),
    (116, "Filter 1 Style", [
        (0.03100, "12dB"),
        (0.12475, "24dB"),
        (0.24975, "Notch Blend"),
        (0.37475, "Notch Spread"),
        (0.49975, "B/P/N"),
    ]),
    (380, "Oscillator 1 Distortion Type", [
        (0.02075, "None"),
        (0.08325, "Sync"),
        (0.16650, "Formant"),
        (0.25000, "Quantize"),
        (0.33325, "Bend"),
        (0.41650, "Squeeze"),
        (0.50000, "Pulse"),
        (0.62500, "FM <- Osc"),
        (0.87500, "RM <- Osc"),
    ]),
]

N_CONTINUOUS = len(CONTINUOUS)
CATEGORICAL_SIZES = [len(opts) for _, _, opts in CATEGORICAL]
CONTINUOUS_NAMES = [n for _, n, _, _ in CONTINUOUS]


def sample_continuous():
    """Raw normalized values, one per continuous parameter."""
    return [lo + random.random() * (hi - lo) for _, _, lo, hi in CONTINUOUS]


def sample_classes():
    """Class indices, one per categorical parameter."""
    return [random.randrange(len(opts)) for _, _, opts in CATEGORICAL]


def apply(synth, continuous, classes):
    """Set every v1 parameter on a plugin processor."""
    for idx, val in FIXED.items():
        synth.set_parameter(idx, val)
    for (idx, _, _, _), val in zip(CONTINUOUS, continuous):
        synth.set_parameter(idx, val)
    for (idx, _, opts), c in zip(CATEGORICAL, classes):
        synth.set_parameter(idx, opts[c][0])