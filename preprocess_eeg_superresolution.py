"""Create paired 7-channel/high-density EEG windows for spatial reconstruction.

Example
-------
python preprocess_eeg_superresolution.py \
    --data-root /path/to/openneuro_dataset \
    --output-dir /path/to/processed_eeg

The output contains one compressed .npz file per recording and a manifest.csv.
Inputs have shape (windows, 7, samples); targets have shape
(windows, high_density_channels, samples).
"""

from __future__ import annotations

# argparse lets the program receive paths and settings from the Terminal. This
# means you do not need to edit the Python file every time you change datasets.
import argparse

# csv is part of Python's standard library. We use it to create a summary table
# showing which recordings were processed and how the DSI electrodes were mapped.
import csv

# json turns a Python dictionary into text that can be stored in one CSV cell.
import json

# re provides regular expressions. These help us recognize channel names such as
# A1, A2, ..., A256 while excluding unrelated channels.
import re

# pathlib provides the Path object, which is safer and more readable than manually
# joining folder names with forward slashes.
from pathlib import Path

# MNE is the main Python package used here for reading and filtering EEG data.
import mne

# NumPy stores EEG signals as multidimensional numerical arrays.
import numpy as np


# Seven scalp positions used as the default simulated DSI-7 montage. Confirm these
# against the channel labels exported by your lab's particular DSI-7 configuration.
# This is our initial assumption about the seven DSI scalp locations. Confirm it
# using an exported file from your particular headset. Some DSI-7 configurations
# may use Pz instead of Fz because one position can be used for the reference/CMF.
DEFAULT_DSI_CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "Fz")

# The script will search the dataset folder for files with these extensions.
# BDF is common for BioSemi systems, while EDF, FIF, and SET are also common EEG
# formats. File-extension matching is made lowercase later in the script.
SUPPORTED_EXTENSIONS = {".bdf", ".edf", ".fif", ".set"}


def read_raw(path: Path) -> mne.io.BaseRaw:
    """Read one EEG recording based on its extension."""
    # This dictionary connects each file extension to the correct MNE reader.
    # Storing the functions themselves lets us choose one without a long if/elif.
    readers = {
        ".bdf": mne.io.read_raw_bdf,
        ".edf": mne.io.read_raw_edf,
        ".fif": mne.io.read_raw_fif,
        ".set": mne.io.read_raw_eeglab,
    }
    # preload=True loads the signal into memory because filtering and resampling
    # modify the data. verbose="ERROR" suppresses most routine MNE messages.
    return readers[path.suffix.lower()](path, preload=True, verbose="ERROR")


def select_probable_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Keep EEG channels while excluding common BioSemi auxiliary channels."""
    # MNE stores the ordered channel labels in raw.ch_names.
    names = raw.ch_names

    # This expression recognizes A1 through A256. The full expression is strict so
    # that names such as EXG1, Resp, Status, or accelerometer axes do not match.
    biosemi_a = [name for name in names if re.fullmatch(r"A(?:[1-9]|[1-9]\d|1\d\d|2[0-4]\d|25[0-6])", name)]
    if len(biosemi_a) >= 32:
        # Open BioSemi files sometimes label respiration, plethysmography, and
        # external sensors as EEG. A1...A256 are the actual cap electrodes.
        # raw.pick changes the Raw object so it contains only these channels.
        raw.pick(biosemi_a)
    else:
        # For a non-BioSemi dataset, trust the EEG channel types supplied by the
        # dataset and exclude channels already marked as bad by its authors.
        raw.pick(picks="eeg", exclude="bads")

    # Seven inputs plus at least one missing target are necessary for this project.
    if raw.info["nchan"] < 8:
        raise ValueError(f"Only {raw.info['nchan']} usable EEG channels were found")
    return raw


def add_standard_montage(raw: mne.io.BaseRaw) -> None:
    """Attach known electrode coordinates when the file does not contain them."""
    # Names such as A1 do not communicate locations like F3 does. If every channel
    # has an A-number name, assume the known BioSemi cap layout when its density is
    # consistent with a 256-channel cap.
    if all(re.fullmatch(r"A\d+", name) for name in raw.ch_names):
        montage_name = "biosemi256" if len(raw.ch_names) >= 200 else None
        if montage_name is None:
            raise ValueError(
                "A-numbered channels were found, but their cap layout is unknown. "
                "Provide a dataset with digitized coordinates or add its MNE montage."
            )
        # A montage is a mapping from channel names to 3-D head coordinates.
        # on_missing="raise" stops instead of silently assigning wrong locations.
        raw.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="raise")


def channel_positions(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    """Return a dictionary mapping every available channel to its x/y/z location."""
    # get_montage returns None when the file and script supplied no coordinates.
    montage = raw.get_montage()
    if montage is None:
        raise ValueError("This recording has no electrode coordinates/montage")
    # MNE's ch_pos dictionary contains one three-dimensional coordinate per sensor.
    positions = montage.get_positions()["ch_pos"]

    # Keep the original raw.ch_names order and ignore montage entries not recorded
    # in this particular file.
    return {name: np.asarray(positions[name]) for name in raw.ch_names if name in positions}


def find_dsi_channels(raw: mne.io.BaseRaw, desired: tuple[str, ...]) -> list[str]:
    """Use exact 10-20 names, or the spatially nearest high-density electrodes."""
    # casefold allows Fz and FZ to match while preserving the dataset's spelling.
    case_lookup = {name.casefold(): name for name in raw.ch_names}

    # Some datasets already contain ordinary 10-20 labels. In that case there is
    # no reason to estimate which electrode is closest.
    if all(name.casefold() in case_lookup for name in desired):
        return [case_lookup[name.casefold()] for name in desired]

    # If exact labels are absent (for example A1...A256), compare their 3-D
    # coordinates with the desired positions in MNE's standard 10-20 template.
    observed = channel_positions(raw)
    template = mne.channels.make_standard_montage("standard_1020")
    target_positions = template.get_positions()["ch_pos"]
    missing = [name for name in desired if name not in target_positions]
    if missing:
        raise ValueError(f"Unknown requested 10-20 positions: {missing}")

    # Work with a copy so a selected electrode can be removed from consideration.
    available = dict(observed)
    selected = []
    for target in desired:
        target_xyz = np.asarray(target_positions[target])

        # Euclidean distance in three-dimensional head coordinates selects the cap
        # electrode physically closest to the requested DSI location.
        nearest = min(available, key=lambda name: np.linalg.norm(available[name] - target_xyz))
        selected.append(nearest)
        del available[nearest]  # Never assign one cap electrode to two DSI positions.
    return selected


def preprocess(raw: mne.io.BaseRaw, highpass: float, lowpass: float, sfreq: float) -> None:
    """Apply deliberately conservative preprocessing for a first reconstruction study."""
    # Use the line frequency stored in the file. If it is absent, default to 60 Hz,
    # which is the electrical-grid frequency in the United States.
    line_frequency = raw.info.get("line_freq") or 60.0

    # Frequencies at or above the Nyquist frequency cannot be represented, so only
    # construct notch frequencies below half the original sampling rate.
    nyquist = raw.info["sfreq"] / 2.0
    notches = np.arange(line_frequency, nyquist, line_frequency)
    if len(notches):
        # A notch filter attenuates line noise at 60 Hz and its harmonics.
        raw.notch_filter(notches, verbose="ERROR")

    # The band-pass removes very slow drift below 0.5 Hz and high-frequency noise
    # above 45 Hz. These defaults still preserve delta through gamma-range activity.
    raw.filter(highpass, lowpass, verbose="ERROR")

    # All recordings need the same number of samples per second before they can be
    # placed into one machine-learning dataset.
    if not np.isclose(raw.info["sfreq"], sfreq):
        raw.resample(sfreq, verbose="ERROR")


def make_windows(
    raw: mne.io.BaseRaw,
    dsi_names: list[str],
    window_seconds: float,
    reject_uv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window data and reject segments with very large peak-to-peak amplitudes."""
    # MNE returns shape (channels, time samples) in volts. float32 uses half the
    # memory of float64 and is normally sufficient for neural-network training.
    data = raw.get_data().astype(np.float32)

    # Convert the seven retained channel names into their integer row positions.
    dsi_indices = np.array([raw.ch_names.index(name) for name in dsi_names])

    # Reference every output to the mean of only the retained electrodes. Using
    # the full-cap average here would leak target-channel information into input.
    dsi_reference = data[dsi_indices].mean(axis=0, keepdims=True)
    data -= dsi_reference

    # For the defaults, 2 seconds * 256 samples/second = 512 samples per window.
    samples = int(round(window_seconds * raw.info["sfreq"]))

    # Integer division discards a final partial window because it would be shorter
    # than every other machine-learning example.
    n_windows = data.shape[1] // samples
    data = data[:, : n_windows * samples]

    # First reshape to (channels, windows, samples), then transpose to the standard
    # machine-learning order (windows, channels, samples).
    windows = data.reshape(data.shape[0], n_windows, samples).transpose(1, 0, 2)

    # Peak-to-peak amplitude is max minus min for each channel in each window.
    peak_to_peak = np.ptp(windows, axis=2)

    # Keep a window only when every value is finite and every channel stays below
    # the rejection threshold. The threshold is converted from microvolts to volts.
    good = np.isfinite(windows).all(axis=(1, 2)) & (peak_to_peak < reject_uv * 1e-6).all(axis=1)

    # y contains all high-density channels. x selects only the seven corresponding
    # DSI channels from the same clean windows, producing perfectly aligned pairs.
    targets = windows[good]
    inputs = targets[:, dsi_indices, :]

    # flatnonzero records which original windows survived artifact rejection.
    return inputs, targets, np.flatnonzero(good)


def process_file(path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """Run the complete pipeline for one recording and save its paired arrays."""
    # Each function performs one understandable preprocessing stage.
    raw = select_probable_eeg(read_raw(path))
    add_standard_montage(raw)
    dsi_names = find_dsi_channels(raw, tuple(args.dsi_channels))
    preprocess(raw, args.highpass, args.lowpass, args.sfreq)
    inputs, targets, kept_windows = make_windows(raw, dsi_names, args.window_seconds, args.reject_uv)

    # Use the final four folder/file components to create a mostly unique output
    # name without reproducing the entire original directory tree.
    relative_stem = "__".join(path.with_suffix("").parts[-4:])
    destination = output_dir / f"{relative_stem}.npz"

    # Store coordinates in the same order as channel_names so a future graph model
    # can use electrode geometry rather than treating channels as an arbitrary list.
    positions = channel_positions(raw)
    xyz = np.stack([positions[name] for name in raw.ch_names]).astype(np.float32)

    # An NPZ file is a compressed container that can hold several NumPy arrays.
    np.savez_compressed(
        destination,
        x=inputs,
        y=targets,
        kept_window_indices=kept_windows,
        channel_names=np.asarray(raw.ch_names),
        dsi_channel_names=np.asarray(dsi_names),
        requested_dsi_positions=np.asarray(args.dsi_channels),
        channel_xyz=xyz,
        sfreq=np.float32(raw.info["sfreq"]),
    )
    # Return a small summary row. The large EEG arrays remain inside the NPZ file.
    return {
        "source": str(path),
        "output": str(destination),
        "subject": next((part for part in path.parts if part.startswith("sub-")), "unknown"),
        "n_channels": len(raw.ch_names),
        "n_windows_kept": len(inputs),
        "dsi_mapping": json.dumps(dict(zip(args.dsi_channels, dsi_names))),
    }


def parse_args() -> argparse.Namespace:
    """Define the options the user can provide in the Terminal."""
    parser = argparse.ArgumentParser(description=__doc__)

    # required=True means the user must supply these two paths.
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # nargs=7 requires exactly seven electrode labels if this option is changed.
    parser.add_argument("--dsi-channels", nargs=7, default=DEFAULT_DSI_CHANNELS)

    # The remaining settings have reasonable initial defaults but can be overridden.
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--highpass", type=float, default=0.5)
    parser.add_argument("--lowpass", type=float, default=45.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--reject-uv", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    """Find every recording, process each one, and create the manifest."""
    # Read the user's command-line options.
    args = parse_args()

    # parents=True creates missing parent folders; exist_ok=True permits reuse of an
    # existing output directory without deleting its contents.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # rglob searches through every participant/session subfolder. sorted makes the
    # processing order repeatable across runs.
    files = sorted(path for path in args.data_root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported EEG files found under {args.data_root}")

    # One summary dictionary will be appended for each successfully processed file.
    rows = []
    for number, path in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] {path}")
        try:
            rows.append(process_file(path, args.output_dir, args))
        except Exception as error:
            # One problematic participant should not terminate a long dataset run.
            # The printed error explains which file needs individual inspection.
            print(f"  SKIPPED: {error}")

    if not rows:
        raise RuntimeError("Every recording was skipped; review the error messages above")
    # The manifest makes it easy to inspect subject IDs, output files, and the
    # precise mapping between desired DSI positions and high-density electrodes.
    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} recordings and {manifest}")


if __name__ == "__main__":
    main()
