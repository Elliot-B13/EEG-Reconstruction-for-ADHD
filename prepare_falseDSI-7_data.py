"""Prepare aligned DSI-24 and simulated DSI-7 EEG windows.

This script recursively finds DSI EDF recordings, applies the same conservative
preprocessing to all scalp channels, cuts them into aligned windows, and writes
two matching directory trees:

* a full-montage dataset containing every recognized DSI-24 scalp channel;
* a simulated DSI-7 dataset containing only the selected seven channels.

The output is intentionally *not* normalized. Normalization must be fitted only
on training participants, so it belongs in the model-training script.

Example
-------
python prepare_dsi7_data.py \
    --input-root /Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-24.Sourcedata \
    --dsi24-output /Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-24.Clean \
    --dsi7-output /Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-7.Artificial
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np


# Current working assumption for a fixed DSI-7-style montage. Keep this easy to
# override because DSI-7 systems can be customized and the exact montage/reference
# must be confirmed from the intended physical headset.
DEFAULT_DSI7_CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "Pz")

# Nineteen standard scalp positions available on a typical DSI-24. A1 and A2 are
# ear electrodes and are not reconstruction targets.
DSI24_SCALP_CHANNELS = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
)

# Older 10-20 temporal names sometimes appear in EDF files.
CHANNEL_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dsi24-output", type=Path, required=True)
    parser.add_argument("--dsi7-output", type=Path, required=True)
    parser.add_argument(
        "--file-pattern",
        default="*_acq-DSI_eeg.edf",
        help="Recursive filename pattern used to find DSI recordings.",
    )
    parser.add_argument(
        "--dsi7-channels",
        nargs=7,
        default=DEFAULT_DSI7_CHANNELS,
        metavar=("CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"),
        help="Exactly seven canonical 10-20 channel names to retain.",
    )
    parser.add_argument(
        "--reference-channels",
        nargs="*",
        default=None,
        help=(
            "Optional recorded reference channels, for example A1 A2. "
            "Omit this option to preserve the EDF's acquisition reference."
        ),
    )
    parser.add_argument("--highpass", type=float, default=0.5)
    parser.add_argument("--lowpass", type=float, default=45.0)
    parser.add_argument(
        "--notch",
        type=float,
        default=60.0,
        help="Line-noise fundamental in Hz; use 0 to disable.",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=300.0,
        help="Output sampling rate in Hz. DSI-24 data are expected at 300 Hz.",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=2.0,
        help="Distance between consecutive window starts.",
    )
    parser.add_argument(
        "--reject-uv",
        type=float,
        default=0.0,
        help=(
            "Reject a window if any scalp channel exceeds this peak-to-peak "
            "amplitude in microvolts. Zero disables amplitude rejection; "
            "non-finite windows are always rejected."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files for a quick test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing paired output files.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect channel mappings without filtering or writing output files.",
    )
    return parser.parse_args()


def canonical_channel_name(name: str) -> str:
    """Convert common EDF channel-label variants to canonical 10-20 names."""
    cleaned = name.strip()
    cleaned = re.sub(r"^EEG[ .:_-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([ .:_-]*(REF|LE|RE))$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .:_-")

    known_names = list(DSI24_SCALP_CHANNELS) + ["A1", "A2", "Fpz"]
    case_lookup = {known.casefold(): known for known in known_names}
    canonical = case_lookup.get(cleaned.casefold(), cleaned)
    return CHANNEL_ALIASES.get(canonical.upper(), canonical)
