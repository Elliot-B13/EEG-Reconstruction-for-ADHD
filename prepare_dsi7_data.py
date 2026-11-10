"""Prepare one continuous NPZ per recording; preserve the acquisition reference.

Run separately for each dataset. Supported entry files: EDF, BrainVision VHDR,
and continuous EEGLAB SET. Output arrays have shape (channels, time), in uV.
Windowing, input selection and training-only normalization belong to the trainer.
Legacy paired/windowed files are not overwritten by this format.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--dataset-id', required=True, help='Unique dataset label, e.g. AhnLee26')
    p.add_argument('--file-pattern', default=None, help='Optional filename glob, e.g. *_acq-DSI_eeg.edf')
    p.add_argument('--highpass', type=float, default=0.5)
    p.add_argument('--lowpass', type=float, default=45.0)
    p.add_argument('--sfreq', type=float, default=300.0)
    p.add_argument('--reference-description', default=None,
                   help='Documented reference text only; does NOT rereference the signals')
    p.add_argument('--limit', type=int)
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def read_source(path, preload=False):
    readers = {'.edf': mne.io.read_raw_edf,
               '.vhdr': mne.io.read_raw_brainvision,
               '.set': mne.io.read_raw_eeglab}
    return readers[path.suffix.lower()](path, preload=preload, verbose='ERROR')


def sidecar_metadata(source, root):
    """Apply matching ancestor EEG JSON files from broad to specific (BIDS inheritance)."""
    combined, paths = {}, []
    source_entities = dict(token.split('-', 1) for token in source.stem.split('_')
                           if '-' in token)
    directories = list(reversed(source.parent.relative_to(root).parents))
    directories = [root / d for d in directories] + [source.parent]
    for directory in dict.fromkeys(directories):
        candidates = []
        for path in directory.glob('*.json'):
            if path.stem != 'eeg' and not path.stem.endswith('_eeg'):
                continue
            entities = dict(token.split('-', 1) for token in path.stem.split('_') if '-' in token)
            if all(source_entities.get(k) == v for k, v in entities.items()):
                candidates.append((len(entities), path))
        for _, path in sorted(candidates):
            combined.update(json.loads(path.read_text(encoding='utf-8-sig')))
            paths.append(str(path))
    return combined, paths


def atomic_savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    # Every field must load without pickle, including annotation descriptions.
    with np.load(temporary, allow_pickle=False) as archive:
        for key in archive.files:
            archive[key]
    os.replace(temporary, path)


def process(source, args):
    relative = source.relative_to(args.input_root)
    # Keep the original extension in the name to avoid format-name collisions.
    destination = args.output_root / args.dataset_id / (str(relative) + '.npz')
    subject = extract_subject(relative)
    if subject == 'unknown':
        raise ValueError('Cannot identify participant; add a sub-... directory or filename')
    metadata, metadata_paths = sidecar_metadata(source, args.input_root)
    reference = args.reference_description or metadata.get('EEGReference') or 'unknown'
    if str(reference).strip().lower() in ('n/a', 'na', ''):
        reference = 'unknown'
    config = {'highpass': args.highpass, 'lowpass': args.lowpass,
              'sfreq': args.sfreq, 'reference_action': 'preserve',
              'reference_description': str(reference)}
    signature = json.dumps(config, sort_keys=True)
    if destination.exists() and not args.overwrite and not args.inspect_only:
        with np.load(destination, allow_pickle=False) as saved:
            if int(saved['schema_version']) != 2 or str(saved['preprocessing_json'].item()) != signature:
                raise ValueError('Existing output uses different settings; use a new output root or --overwrite')
        return {'source': str(source), 'output': str(destination), 'status': 'skipped',
                'subject': args.dataset_id + '/' + subject, 'reference': reference}
    raw = read_source(source)
    try:
        original_labels = list(raw.ch_names)
        lookup = channel_lookup(original_labels)
        missing = [ch for ch in DSI24_SCALP_CHANNELS if ch not in lookup]
        if missing:
            raise ValueError(f'Missing required scalp channels: {missing}')
        raw.pick([lookup[ch] for ch in DSI24_SCALP_CHANNELS])
        raw.reorder_channels([lookup[ch] for ch in DSI24_SCALP_CHANNELS])
        raw.rename_channels({lookup[ch]: ch for ch in DSI24_SCALP_CHANNELS})
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names}, verbose='ERROR')
        row = {'source': str(source), 'output': str(destination),
               'status': 'inspected', 'subject': args.dataset_id + '/' + subject,
               'reference': str(reference), 'seconds': raw.n_times / raw.info['sfreq']}
        if args.inspect_only:
            return row
        if not 0 <= args.highpass < args.lowpass < min(raw.info['sfreq'], args.sfreq) / 2:
            raise ValueError('Require 0 <= highpass < lowpass < both input/output Nyquist frequencies')
        raw.load_data()
        if not np.isfinite(raw.get_data()).all():
            raise ValueError('Nonfinite source samples; resolve before filtering')
        raw.filter(args.highpass or None, args.lowpass, verbose='ERROR')
        if not np.isclose(raw.info['sfreq'], args.sfreq):
            raw.resample(args.sfreq, verbose='ERROR')
        data = (raw.get_data() * 1e6).astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite prepared samples')
        atomic_savez(destination,
            schema_version=np.int64(2), data_uv=data,
            channel_names=np.asarray(raw.ch_names, dtype=str),
            original_channel_names=np.asarray(original_labels, dtype=str),
            sfreq=np.float64(raw.info['sfreq']), unit=np.asarray('microvolts'),
            subject=np.asarray(args.dataset_id + '/' + subject),
            dataset_id=np.asarray(args.dataset_id), task=np.asarray(extract_task(source)),
            source_file=np.asarray(str(source)), relative_id=np.asarray(str(relative)),
            reference_description=np.asarray(str(reference)),
            reference_action=np.asarray('preserve'),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            metadata_sources=np.asarray(metadata_paths, dtype=str),
            preprocessing_json=np.asarray(signature),
            # Convert annotation onsets to seconds from the first saved sample.
            annotation_onsets=np.asarray(raw.annotations.onset - raw.first_time, dtype=np.float64),
            annotation_durations=np.asarray(raw.annotations.duration, dtype=np.float64),
            annotation_descriptions=np.asarray([str(description) for description in raw.annotations.description],dtype=str))
        row.update(status='prepared', samples=data.shape[1])
        return row
    finally:
        raw.close()


def main():
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.input_root.is_dir():
        raise NotADirectoryError(args.input_root)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.dataset_id) or args.dataset_id in ('.', '..'):
        raise ValueError('dataset-id must be a simple folder label')
    if (args.output_root == args.input_root or args.input_root in args.output_root.parents
            or args.output_root in args.input_root.parents):
        raise ValueError('Use separate, non-nested source and output roots')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    files = sorted(p for p in args.input_root.rglob('*')
        if p.is_file() and p.suffix.lower() in ('.edf', '.vhdr', '.set')
        and not any(part.startswith('.') or part in ('eye_tracker', 'derivatives', 'processed')
                    for part in p.relative_to(args.input_root).parts)
        and (args.file_pattern is None or p.match(args.file_pattern)))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError('No supported recording entry files found')
    rows = []
    for i, source in enumerate(files, 1):
        try:
            row = process(source, args)
        except Exception as error:
            row = {'source': str(source), 'status': 'failed', 'error': str(error)}
        rows.append(row)
        print(f"[{i}/{len(files)}] {row['status']}: {source.name} {row.get('error', '')}", flush=True)
    if not args.inspect_only:
        directory = args.output_root / args.dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        # Separate audit per invocation; previous manifests are retained.
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with (directory / f'manifest_{stamp}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
    failed = sum(row['status'] == 'failed' for row in rows)
    print(f'Finished: recordings={len(rows)}, failed={failed}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
