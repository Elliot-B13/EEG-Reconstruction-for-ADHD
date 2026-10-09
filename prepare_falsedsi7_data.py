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

# collects adjustable settings for preprocessing pipeline and puts in 1 place
# so can adjust settings without going in and changing the code
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


def channel_lookup(channel_names: Sequence[str]) -> dict[str, str]:
    """Map canonical channel names to the actual labels present in one EDF."""
    lookup: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for actual_name in channel_names:
        canonical = canonical_channel_name(actual_name)
        if canonical in lookup:
            duplicates.setdefault(canonical, [lookup[canonical]]).append(actual_name)
        else:
            lookup[canonical] = actual_name

    if duplicates:
        raise ValueError(f"Ambiguous duplicate channel labels after cleanup: {duplicates}")
    return lookup


def resolve_channels(
    lookup: dict[str, str],
    requested: Iterable[str],
    purpose: str,
) -> list[str]:
    canonical_requested = [canonical_channel_name(name) for name in requested]
    missing = [name for name in canonical_requested if name not in lookup]
    if missing:
        available = sorted(lookup)
        raise ValueError(
            f"Missing {purpose} channels {missing}. Canonical channels found: {available}"
        )
    return [lookup[name] for name in canonical_requested]


def extract_subject(path: Path) -> str:
    for part in reversed(path.parts):
        if re.fullmatch(r"sub-[A-Za-z0-9]+", part):
            return part
    match = re.search(r"sub-[A-Za-z0-9]+", path.name)
    return match.group(0) if match else "unknown"


def extract_task(path: Path) -> str:
    match = re.search(r"task-([A-Za-z0-9]+)", path.name)
    return match.group(1) if match else "unknown"


def read_and_prepare_raw(path: Path, args: argparse.Namespace) -> tuple[mne.io.BaseRaw, list[str]]:
    """Read, optionally rereference, select scalp channels, filter, and resample."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    original_lookup = channel_lookup(raw.ch_names)

    if args.reference_channels:
        actual_reference_names = resolve_channels(
            original_lookup,
            args.reference_channels,
            "reference",
        )
        raw.set_eeg_reference(
            ref_channels=actual_reference_names,
            projection=False,
            verbose="ERROR",
        )

    # Keep every recognized scalp channel in a fixed anatomical order. This
    # excludes annotations, accelerometry, status, and A1/A2 ear channels.
    available_scalp = [name for name in DSI24_SCALP_CHANNELS if name in original_lookup]
    if len(available_scalp) < 8:
        raise ValueError(
            f"Only {len(available_scalp)} recognized scalp channels were found: "
            f"{available_scalp}"
        )

    requested_inputs = [canonical_channel_name(name) for name in args.dsi7_channels]
    missing_inputs = [name for name in requested_inputs if name not in available_scalp]
    if missing_inputs:
        raise ValueError(
            f"Requested DSI-7 channels are absent: {missing_inputs}. "
            f"Available scalp channels: {available_scalp}"
        )

    actual_scalp_names = [original_lookup[name] for name in available_scalp]
    raw.pick(actual_scalp_names)

    # Rename only after selection so saved files use a consistent canonical order.
    rename_mapping = {
        actual_name: canonical_name
        for actual_name, canonical_name in zip(actual_scalp_names, available_scalp)
        if actual_name != canonical_name
    }
    if rename_mapping:
        raw.rename_channels(rename_mapping)

    raw.set_channel_types({name: "eeg" for name in raw.ch_names}, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="raise", verbose="ERROR")

    if args.inspect_only:
        return raw, requested_inputs

    if not (0 <= args.highpass < args.lowpass < raw.info["sfreq"] / 2):
        raise ValueError(
            "Filtering frequencies must satisfy "
            "0 <= highpass < lowpass < original Nyquist frequency"
        )

    # A notch below the selected low-pass cutoff is useful. With the defaults,
    # the 45 Hz low-pass already removes 60 Hz line noise, so no notch is applied.
    if 0 < args.notch < args.lowpass:
        notch_frequencies = np.arange(
            args.notch,
            raw.info["sfreq"] / 2,
            args.notch,
        )
        notch_frequencies = notch_frequencies[notch_frequencies < args.lowpass]
        if notch_frequencies.size:
            raw.notch_filter(notch_frequencies, verbose="ERROR")

    raw.filter(args.highpass, args.lowpass, verbose="ERROR")

    if not np.isclose(raw.info["sfreq"], args.sfreq):
        raw.resample(args.sfreq, verbose="ERROR")

    return raw, requested_inputs


def make_windows(
    raw: mne.io.BaseRaw,
    input_channel_names: Sequence[str],
    window_seconds: float,
    stride_seconds: float,
    reject_uv: float,
) -> dict[str, np.ndarray]:
    """Create aligned full-montage and seven-channel windows in microvolts."""
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(window_seconds * sfreq))
    stride_samples = int(round(stride_seconds * sfreq))

    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride durations must produce positive sample counts")
    if raw.n_times < window_samples:
        raise ValueError(
            f"Recording has {raw.n_times} samples, shorter than one "
            f"{window_samples}-sample window"
        )

    full_data_uv = (raw.get_data() * 1e6).astype(np.float32, copy=False)
    starts = np.arange(
        0,
        raw.n_times - window_samples + 1,
        stride_samples,
        dtype=np.int64,
    )

    # Shape: windows x channels x samples.
    windows = np.stack(
        [full_data_uv[:, start : start + window_samples] for start in starts],
        axis=0,
    )

    finite = np.isfinite(windows).all(axis=(1, 2))
    peak_to_peak_uv = np.ptp(windows, axis=2)
    good = finite.copy()
    if reject_uv > 0:
        good &= (peak_to_peak_uv <= reject_uv).all(axis=1)

    input_indices = np.asarray(
        [raw.ch_names.index(name) for name in input_channel_names],
        dtype=np.int64,
    )
    target_channel_names = [
        name for name in raw.ch_names if name not in input_channel_names
    ]

    if not target_channel_names:
        raise ValueError("No reconstruction targets remain after input selection")

    return {
        "full_windows_uv": windows[good],
        "input_windows_uv": windows[good][:, input_indices, :],
        "window_start_samples": starts[good],
        "kept_window_indices": np.flatnonzero(good).astype(np.int64),
        "peak_to_peak_uv": peak_to_peak_uv[good].astype(np.float32),
        "all_window_good": good,
        "target_channel_names": np.asarray(target_channel_names),
    }


def atomic_savez(path: Path, **arrays: object) -> None:
    """Write one NPZ completely before replacing its final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def output_paths(
    source: Path,
    input_root: Path,
    dsi24_root: Path,
    dsi7_root: Path,
) -> tuple[Path, Path, Path]:
    relative = source.relative_to(input_root).with_suffix(".npz")
    return dsi24_root / relative, dsi7_root / relative, relative


def process_recording(
    source: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    full_path, input_path, relative_path = output_paths(
        source,
        args.input_root,
        args.dsi24_output,
        args.dsi7_output,
    )

    if not args.overwrite and (full_path.exists() or input_path.exists()):
        if full_path.exists() and input_path.exists():
            return {
                "source": str(source),
                "relative_output": str(relative_path),
                "subject": extract_subject(source),
                "task": extract_task(source),
                "status": "skipped_existing",
                "n_windows": "",
                "input_channels": "",
                "target_channels": "",
                "message": "Both paired outputs already exist",
            }
        raise FileExistsError(
            "Only one member of the output pair exists. Use --overwrite after "
            f"checking these paths: {full_path}, {input_path}"
        )

    raw, input_channel_names = read_and_prepare_raw(source, args)
    target_channel_names = [
        name for name in raw.ch_names if name not in input_channel_names
    ]

    mapping_message = (
        f"full={raw.ch_names}; inputs={input_channel_names}; "
        f"targets={target_channel_names}"
    )
    if args.inspect_only:
        print("  " + mapping_message)
        return {
            "source": str(source),
            "relative_output": str(relative_path),
            "subject": extract_subject(source),
            "task": extract_task(source),
            "status": "inspected",
            "n_windows": "",
            "input_channels": json.dumps(input_channel_names),
            "target_channels": json.dumps(target_channel_names),
            "message": "No files written",
        }

    windowed = make_windows(
        raw,
        input_channel_names,
        args.window_seconds,
        args.stride_seconds,
        args.reject_uv,
    )
    full_windows = windowed["full_windows_uv"]
    input_windows = windowed["input_windows_uv"]
    if len(full_windows) == 0:
        raise ValueError("Every window was rejected")

    common_metadata = {
        "source_file": np.asarray(str(source)),
        "relative_id": np.asarray(str(relative_path)),
        "subject": np.asarray(extract_subject(source)),
        "task": np.asarray(extract_task(source)),
        "sfreq": np.float32(raw.info["sfreq"]),
        "window_seconds": np.float32(args.window_seconds),
        "stride_seconds": np.float32(args.stride_seconds),
        "window_start_samples": windowed["window_start_samples"],
        "window_start_seconds": (
            windowed["window_start_samples"] / raw.info["sfreq"]
        ).astype(np.float32),
        "kept_window_indices": windowed["kept_window_indices"],
        "annotation_onsets": np.asarray(raw.annotations.onset, dtype=np.float64),
        "annotation_durations": np.asarray(raw.annotations.duration, dtype=np.float64),
        "annotation_descriptions": np.asarray(raw.annotations.description),
        "unit": np.asarray("microvolts"),
        "preprocessing_json": np.asarray(
            json.dumps(
                {
                    "highpass": args.highpass,
                    "lowpass": args.lowpass,
                    "notch": args.notch,
                    "output_sfreq": args.sfreq,
                    "reference_channels": args.reference_channels,
                    "reject_uv": args.reject_uv,
                },
                sort_keys=True,
            )
        ),
    }

    atomic_savez(
        full_path,
        data_uv=full_windows,
        channel_names=np.asarray(raw.ch_names),
        input_channel_names=np.asarray(input_channel_names),
        target_channel_names=windowed["target_channel_names"],
        peak_to_peak_uv=windowed["peak_to_peak_uv"],
        **common_metadata,
    )
    atomic_savez(
        input_path,
        data_uv=input_windows,
        channel_names=np.asarray(input_channel_names),
        full_channel_names=np.asarray(raw.ch_names),
        target_channel_names=windowed["target_channel_names"],
        **common_metadata,
    )

    return {
        "source": str(source),
        "relative_output": str(relative_path),
        "subject": extract_subject(source),
        "task": extract_task(source),
        "status": "processed",
        "n_windows": len(full_windows),
        "input_channels": json.dumps(input_channel_names),
        "target_channels": json.dumps(target_channel_names),
        "message": mapping_message,
    }


def write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "relative_output",
        "subject",
        "task",
        "status",
        "n_windows",
        "input_channels",
        "target_channels",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.dsi24_output = args.dsi24_output.expanduser().resolve()
    args.dsi7_output = args.dsi7_output.expanduser().resolve()

    if args.dsi24_output == args.dsi7_output:
        raise ValueError("DSI-24 and DSI-7 output directories must be different")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    files = sorted(args.input_root.rglob(args.file_pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.file_pattern!r} found under {args.input_root}"
        )

    print(f"Found {len(files)} recording(s)")
    print("Requested DSI-7 channels:", list(args.dsi7_channels))
    print("Reference:", args.reference_channels or "preserve EDF acquisition reference")

    rows: list[dict[str, object]] = []
    processed = 0
    failed = 0

    for number, source in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] {source}")
        try:
            row = process_recording(source, args)
            rows.append(row)
            if row["status"] == "processed":
                processed += 1
                print(f"  saved {row['n_windows']} aligned windows")
        except Exception as error:
            failed += 1
            print(f"  ERROR: {error}")
            rows.append(
                {
                    "source": str(source),
                    "relative_output": "",
                    "subject": extract_subject(source),
                    "task": extract_task(source),
                    "status": "failed",
                    "n_windows": "",
                    "input_channels": "",
                    "target_channels": "",
                    "message": repr(error),
                }
            )

    if not args.inspect_only:
        manifest_path = args.dsi24_output / "manifest.csv"
        write_manifest(manifest_path, rows)
        print("Manifest:", manifest_path)

    print(f"Finished: processed={processed}, failed={failed}, total={len(files)}")
    if failed:
        raise RuntimeError(
            f"{failed} recording(s) failed. Review the errors before training."
        )


if __name__ == "__main__":
    main()
