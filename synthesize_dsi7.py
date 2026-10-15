from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import mne


DSI7_CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "FZ")
DEFAULT_INPUT_ROOT = Path(
    "/Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-24.Sourcedata"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-7.Artificial"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def electrode_name(channel_label: str) -> str:
    cleaned = re.sub(r"^EEG[ .:_-]*", "", channel_label.strip(), flags=re.IGNORECASE)
    cleaned = re.split(r"[-/]", cleaned, maxsplit=1)[0]
    return re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()


def select_dsi7_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    matches: dict[str, str] = {}

    for channel_label in raw.ch_names:
        electrode = electrode_name(channel_label)
        if electrode in DSI7_CHANNELS:
            if electrode in matches:
                raise ValueError(
                    f"More than one EDF channel matched {electrode}: "
                    f"{matches[electrode]!r} and {channel_label!r}"
                )
            matches[electrode] = channel_label

    missing = [electrode for electrode in DSI7_CHANNELS if electrode not in matches]
    if missing:
        raise ValueError(
            f"Missing DSI-7 channels {missing}. EDF channels were {raw.ch_names}"
        )

    selected_labels = [matches[electrode] for electrode in DSI7_CHANNELS]
    dsi7 = raw.copy().pick(selected_labels)
    dsi7.reorder_channels(selected_labels)
    return dsi7


def verify_saved_file(path: Path, expected: mne.io.BaseRaw) -> None:
    saved = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")

    if saved.ch_names != expected.ch_names:
        raise RuntimeError(
            f"Saved channel order differs: {saved.ch_names} != {expected.ch_names}"
        )
    if saved.n_times != expected.n_times:
        raise RuntimeError(
            f"Saved sample count differs: {saved.n_times} != {expected.n_times}"
        )
    if saved.info["sfreq"] != expected.info["sfreq"]:
        raise RuntimeError(
            f"Saved sampling rate differs: {saved.info['sfreq']} != "
            f"{expected.info['sfreq']}"
        )


def synthesize_recording(
    source: Path,
    destination: Path,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return "skipped"

    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    dsi7 = select_dsi7_channels(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(f"{destination.stem}.temporary.edf")
    dsi7.export(
        temporary,
        fmt="edf",
        physical_range="auto",
        digital_range="orig",
        overwrite=True,
        verbose="ERROR",
    )

    verify_saved_file(temporary, dsi7)
    os.replace(temporary, destination)
    return "saved"


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("The output directory cannot be inside the input directory")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    sources = sorted(
        path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".edf"
    )
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError(f"No EDF files found under {input_root}")

    saved_count = 0
    skipped_count = 0

    for number, source in enumerate(sources, start=1):
        destination = output_root / source.relative_to(input_root)
        result = synthesize_recording(source, destination, args.overwrite)
        if result == "saved":
            saved_count += 1
        else:
            skipped_count += 1
        print(f"[{number}/{len(sources)}] {result}: {destination}")

    print(f"Finished: saved={saved_count}, skipped={skipped_count}")


if __name__ == "__main__":
    main()
