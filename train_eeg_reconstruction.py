"""Train a modular EEG missing-channel reconstruction model.

The data loader expects matching directory trees created by
``prepare_dsi7_data.py``. The training and evaluation pipeline is independent of
the model architecture: model classes only need to accept tensors shaped
``(batch, input_channels, time)`` and return
``(batch, target_channels, time)``.

The default model is a temporal convolutional network (TCN). A pointwise linear
baseline is also included to demonstrate how future architectures can be added
without rewriting the data pipeline or training loop.

Example
-------
python train_eeg_reconstruction.py \
    --dsi24-root /Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-24.Clean \
    --dsi7-root /Users/elliotbroth/Data/EEG_Reconstruction_Data/DSI-7.Artificial \
    --output-dir /Users/elliotbroth/Projects/Reconstruction_EEG/results/tcn-v1
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class PairRecord:
    """Metadata needed to lazily load one paired recording."""

    relative_path: str
    subject: str
    input_path: Path
    full_path: Path
    n_windows: int
    n_samples: int
    input_channel_names: tuple[str, ...]
    full_channel_names: tuple[str, ...]
    target_channel_names: tuple[str, ...]
    target_indices: tuple[int, ...]


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsi24-root", type=Path, required=True)
    parser.add_argument("--dsi7-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-channels",
        nargs="*",
        default=None,
        help=(
            "Optional ordered target-channel subset. By default, predict every "
            "full-montage channel not present in the DSI-7 input."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("tcn", "pointwise-linear"),
        default="tcn",
        help="Architecture to insert into the shared training pipeline.",
    )
    parser.add_argument(
        "--hidden-channels",
        nargs="+",
        type=int,
        default=[64, 64, 64, 64],
        help="Output width of each TCN residual block.",
    )
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is the safest initial value on macOS.",
    )
    parser.add_argument(
        "--cache-recordings",
        type=int,
        default=2,
        help="Number of decompressed paired recordings cached per dataset worker.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow an already populated output directory.",
    )
    return parser.parse_args()


def string_scalar(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def discover_pairs(
    dsi24_root: Path,
    dsi7_root: Path,
    requested_targets: Sequence[str] | None,
) -> list[PairRecord]:
    """Find matching files and validate all shapes and channel orders."""
    input_files = sorted(dsi7_root.rglob("*.npz"))
    if not input_files:
        raise FileNotFoundError(f"No NPZ files found under {dsi7_root}")

    records: list[PairRecord] = []
    expected_inputs: tuple[str, ...] | None = None
    expected_full: tuple[str, ...] | None = None
    expected_targets: tuple[str, ...] | None = None
    expected_samples: int | None = None

    for input_path in input_files:
        relative = input_path.relative_to(dsi7_root)
        full_path = dsi24_root / relative
        if not full_path.exists():
            raise FileNotFoundError(
                f"Missing DSI-24 partner for {input_path}: expected {full_path}"
            )

        with np.load(input_path, allow_pickle=False) as input_file:
            input_shape = tuple(input_file["data_uv"].shape)
            input_names = tuple(str(name) for name in input_file["channel_names"])
            subject = string_scalar(input_file["subject"])

        with np.load(full_path, allow_pickle=False) as full_file:
            full_shape = tuple(full_file["data_uv"].shape)
            full_names = tuple(str(name) for name in full_file["channel_names"])
            stored_targets = tuple(
                str(name) for name in full_file["target_channel_names"]
            )
            full_subject = string_scalar(full_file["subject"])

        if len(input_shape) != 3 or len(full_shape) != 3:
            raise ValueError(
                f"Expected 3-D arrays in {relative}; got {input_shape} and {full_shape}"
            )
        if input_shape[0] != full_shape[0] or input_shape[2] != full_shape[2]:
            raise ValueError(
                f"Window alignment mismatch in {relative}: "
                f"input={input_shape}, full={full_shape}"
            )
        if input_shape[1] != len(input_names):
            raise ValueError(f"Input channel metadata mismatch in {relative}")
        if full_shape[1] != len(full_names):
            raise ValueError(f"Full channel metadata mismatch in {relative}")
        if subject != full_subject:
            raise ValueError(f"Subject metadata mismatch in {relative}")
        if not set(input_names).issubset(full_names):
            raise ValueError(
                f"Input channels are not a subset of the full montage in {relative}"
            )

        if requested_targets:
            target_names = tuple(requested_targets)
            absent = [name for name in target_names if name not in full_names]
            if absent:
                raise ValueError(f"Requested targets absent from {relative}: {absent}")
        else:
            target_names = stored_targets

        overlap = sorted(set(input_names).intersection(target_names))
        if overlap:
            raise ValueError(f"Input/target channel overlap in {relative}: {overlap}")
        target_indices = tuple(full_names.index(name) for name in target_names)

        current_samples = input_shape[2]
        if expected_inputs is None:
            expected_inputs = input_names
            expected_full = full_names
            expected_targets = target_names
            expected_samples = current_samples
        elif (
            input_names != expected_inputs
            or full_names != expected_full
            or target_names != expected_targets
            or current_samples != expected_samples
        ):
            raise ValueError(
                "All recordings must have identical channel order and window length. "
                f"First input/full/target/sample definition differs at {relative}."
            )

        records.append(
            PairRecord(
                relative_path=str(relative),
                subject=subject,
                input_path=input_path,
                full_path=full_path,
                n_windows=input_shape[0],
                n_samples=current_samples,
                input_channel_names=input_names,
                full_channel_names=full_names,
                target_channel_names=target_names,
                target_indices=target_indices,
            )
        )

    return records


def split_by_subject(
    records: Sequence[PairRecord],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord], list[PairRecord], dict[str, list[str]]]:
    """Create leakage-safe partitions in which each subject appears once."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("Validation and test fractions must be between zero and one")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fraction must be less than one")

    subjects = sorted({record.subject for record in records})
    if len(subjects) < 3:
        raise ValueError(
            "At least three subjects are required for participant-level "
            "train/validation/test splits. Process more subjects before training."
        )

    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    n_test = max(1, round(n_subjects * test_fraction))
    n_val = max(1, round(n_subjects * val_fraction))
    if n_test + n_val >= n_subjects:
        raise ValueError("Split fractions leave no training subjects")

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    train = [record for record in records if record.subject in train_subjects]
    val = [record for record in records if record.subject in val_subjects]
    test = [record for record in records if record.subject in test_subjects]
    split = {
        "train": sorted(train_subjects),
        "validation": sorted(val_subjects),
        "test": sorted(test_subjects),
    }
    return train, val, test, split


def compute_channel_stats(
    records: Sequence[PairRecord],
    source: str,
) -> ChannelStats:
    """Compute training-only per-channel mean and standard deviation."""
    total_sum: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    count = 0

    for record in records:
        if source == "input":
            with np.load(record.input_path, allow_pickle=False) as npz_file:
                array = np.asarray(npz_file["data_uv"], dtype=np.float64)
        elif source == "target":
            with np.load(record.full_path, allow_pickle=False) as npz_file:
                array = np.asarray(
                    npz_file["data_uv"][:, record.target_indices, :],
                    dtype=np.float64,
                )
        else:
            raise ValueError(f"Unknown source: {source}")

        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite {source} values found in {record.relative_path}")

        channel_sum = array.sum(axis=(0, 2))
        channel_squared = np.square(array).sum(axis=(0, 2))
        total_sum = channel_sum if total_sum is None else total_sum + channel_sum
        total_squared = (
            channel_squared
            if total_squared is None
            else total_squared + channel_squared
        )
        count += array.shape[0] * array.shape[2]

    if total_sum is None or total_squared is None or count == 0:
        raise ValueError(f"Cannot calculate {source} statistics from an empty split")

    mean = total_sum / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-6)
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


class PairedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy paired dataset with a small recording-level LRU cache."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        input_stats: ChannelStats,
        target_stats: ChannelStats,
        cache_recordings: int,
    ) -> None:
        self.records = list(records)
        self.input_stats = input_stats
        self.target_stats = target_stats
        self.cache_recordings = max(1, cache_recordings)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.offsets: list[int] = []
        running_total = 0
        for record in self.records:
            running_total += record.n_windows
            self.offsets.append(running_total)

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def _load_recording(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        if record_index in self._cache:
            arrays = self._cache.pop(record_index)
            self._cache[record_index] = arrays
            return arrays

        record = self.records[record_index]
        with np.load(record.input_path, allow_pickle=False) as input_file:
            inputs = np.asarray(input_file["data_uv"], dtype=np.float32)
        with np.load(record.full_path, allow_pickle=False) as full_file:
            targets = np.asarray(
                full_file["data_uv"][:, record.target_indices, :],
                dtype=np.float32,
            )

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Pair became misaligned: {record.relative_path}")
        self._cache[record_index] = (inputs, targets)
        while len(self._cache) > self.cache_recordings:
            self._cache.popitem(last=False)
        return inputs, targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        record_index = bisect.bisect_right(self.offsets, index)
        previous_offset = 0 if record_index == 0 else self.offsets[record_index - 1]
        window_index = index - previous_offset
        inputs, targets = self._load_recording(record_index)

        x = (inputs[window_index] - self.input_stats.mean[:, None]) / (
            self.input_stats.std[:, None]
        )
        y = (targets[window_index] - self.target_stats.mean[:, None]) / (
            self.target_stats.std[:, None]
        )
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


class RecordingBatchSampler(Sampler[list[int]]):
    """Shuffle recordings/windows while keeping batches recording-local.

    Grouping batches by recording avoids repeatedly decompressing many different
    NPZ files inside one batch, which keeps the first storage format usable as the
    dataset grows.
    """

    def __init__(
        self,
        dataset: PairedWindowDataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        record_indices = list(range(len(self.dataset.records)))
        if self.shuffle:
            rng.shuffle(record_indices)

        starts = [0] + self.dataset.offsets[:-1]
        for record_index in record_indices:
            start = starts[record_index]
            count = self.dataset.records[record_index].n_windows
            indices = list(range(start, start + count))
            if self.shuffle:
                rng.shuffle(indices)
            for batch_start in range(0, count, self.batch_size):
                yield indices[batch_start : batch_start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        return sum(
            math.ceil(record.n_windows / self.batch_size)
            for record in self.dataset.records
        )


class TemporalResidualBlock(nn.Module):
    """A same-length, non-causal dilated temporal residual block."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Use an odd TCN kernel size to preserve exact length")
        padding = dilation * (kernel_size - 1) // 2

        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.skip(inputs))


class TCNReconstructor(nn.Module):
    """Dilated temporal CNN that preserves the input window length."""

    def __init__(
        self,
        input_channels: int,
        target_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("TCN requires at least one hidden block")

        blocks: list[nn.Module] = []
        previous_channels = input_channels
        for level, width in enumerate(hidden_channels):
            blocks.append(
                TemporalResidualBlock(
                    input_channels=previous_channels,
                    output_channels=width,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            previous_channels = width

        self.temporal_blocks = nn.Sequential(*blocks)
        self.output_projection = nn.Conv1d(
            previous_channels,
            target_channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.temporal_blocks(inputs))


class PointwiseLinearReconstructor(nn.Module):
    """Instantaneous multi-output linear baseline implemented as a 1x1 convolution."""

    def __init__(self, input_channels: int, target_channels: int) -> None:
        super().__init__()
        self.mapping = nn.Conv1d(input_channels, target_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mapping(inputs)


def build_model(
    model_name: str,
    input_channels: int,
    target_channels: int,
    args: argparse.Namespace,
) -> nn.Module:
    """Architecture factory: add future models here without changing training."""
    if model_name == "tcn":
        return TCNReconstructor(
            input_channels=input_channels,
            target_channels=target_channels,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if model_name == "pointwise-linear":
        return PointwiseLinearReconstructor(input_channels, target_channels)
    raise ValueError(f"Unknown model: {model_name}")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    element_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_sum += loss.item() * targets.numel()
        element_count += targets.numel()

    return loss_sum / max(element_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    target_stats: ChannelStats,
    target_names: Sequence[str],
) -> dict[str, object]:
    model.eval()
    channel_count = len(target_names)
    standardized_sse = 0.0
    standardized_count = 0

    sums = {
        "prediction": np.zeros(channel_count, dtype=np.float64),
        "target": np.zeros(channel_count, dtype=np.float64),
        "prediction_squared": np.zeros(channel_count, dtype=np.float64),
        "target_squared": np.zeros(channel_count, dtype=np.float64),
        "cross": np.zeros(channel_count, dtype=np.float64),
        "squared_error": np.zeros(channel_count, dtype=np.float64),
        "absolute_error": np.zeros(channel_count, dtype=np.float64),
    }
    sample_count = 0
    target_mean = torch.as_tensor(
        target_stats.mean,
        dtype=torch.float32,
        device=device,
    )[None, :, None]
    target_std = torch.as_tensor(
        target_stats.std,
        dtype=torch.float32,
        device=device,
    )[None, :, None]

    for inputs, targets_standardized in loader:
        inputs = inputs.to(device)
        targets_standardized = targets_standardized.to(device)
        predictions_standardized = model(inputs)
        loss = loss_function(predictions_standardized, targets_standardized)
        standardized_sse += loss.item() * targets_standardized.numel()
        standardized_count += targets_standardized.numel()

        predictions = (
            (predictions_standardized * target_std + target_mean)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        targets = (
            (targets_standardized * target_std + target_mean)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        axes = (0, 2)
        difference = predictions - targets

        sums["prediction"] += predictions.sum(axis=axes)
        sums["target"] += targets.sum(axis=axes)
        sums["prediction_squared"] += np.square(predictions).sum(axis=axes)
        sums["target_squared"] += np.square(targets).sum(axis=axes)
        sums["cross"] += (predictions * targets).sum(axis=axes)
        sums["squared_error"] += np.square(difference).sum(axis=axes)
        sums["absolute_error"] += np.abs(difference).sum(axis=axes)
        sample_count += predictions.shape[0] * predictions.shape[2]

    if sample_count == 0:
        raise ValueError("Evaluation loader produced no samples")

    rmse = np.sqrt(sums["squared_error"] / sample_count)
    mae = sums["absolute_error"] / sample_count
    target_variance = np.maximum(
        sums["target_squared"] / sample_count
        - np.square(sums["target"] / sample_count),
        0.0,
    )
    target_scale = np.sqrt(target_variance)
    nrmse = rmse / np.maximum(target_scale, 1e-12)

    covariance_numerator = (
        sums["cross"]
        - sums["prediction"] * sums["target"] / sample_count
    )
    prediction_ss = (
        sums["prediction_squared"]
        - np.square(sums["prediction"]) / sample_count
    )
    target_ss = (
        sums["target_squared"]
        - np.square(sums["target"]) / sample_count
    )
    correlation = covariance_numerator / np.sqrt(
        np.maximum(prediction_ss * target_ss, 1e-24)
    )

    per_channel = {
        name: {
            "rmse_uv": float(rmse[index]),
            "mae_uv": float(mae[index]),
            "nrmse": float(nrmse[index]),
            "pearson_r": float(correlation[index]),
        }
        for index, name in enumerate(target_names)
    }
    return {
        "standardized_mse": standardized_sse / standardized_count,
        "mean_rmse_uv": float(np.mean(rmse)),
        "mean_mae_uv": float(np.mean(mae)),
        "mean_nrmse": float(np.mean(nrmse)),
        "mean_pearson_r": float(np.mean(correlation)),
        "per_channel": per_channel,
    }


def make_loader(
    dataset: PairedWindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    sampler = RecordingBatchSampler(dataset, batch_size, shuffle, seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def save_history(path: Path, history: Sequence[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.dsi24_root = args.dsi24_root.expanduser().resolve()
    args.dsi7_root = args.dsi7_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Choose a new run directory or pass --overwrite-output."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print("Device:", device)

    records = discover_pairs(
        args.dsi24_root,
        args.dsi7_root,
        args.target_channels,
    )
    train_records, val_records, test_records, split = split_by_subject(
        records,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    save_json(args.output_dir / "subject_split.json", split)

    print(
        "Recordings:",
        f"train={len(train_records)},",
        f"validation={len(val_records)},",
        f"test={len(test_records)}",
    )
    print(
        "Subjects:",
        f"train={len(split['train'])},",
        f"validation={len(split['validation'])},",
        f"test={len(split['test'])}",
    )

    input_stats = compute_channel_stats(train_records, "input")
    target_stats = compute_channel_stats(train_records, "target")
    first_record = records[0]
    input_names = first_record.input_channel_names
    target_names = first_record.target_channel_names

    normalization = {
        "input_channel_names": list(input_names),
        "input_mean_uv": input_stats.mean.tolist(),
        "input_std_uv": input_stats.std.tolist(),
        "target_channel_names": list(target_names),
        "target_mean_uv": target_stats.mean.tolist(),
        "target_std_uv": target_stats.std.tolist(),
    }
    save_json(args.output_dir / "normalization.json", normalization)

    train_dataset = PairedWindowDataset(
        train_records,
        input_stats,
        target_stats,
        args.cache_recordings,
    )
    val_dataset = PairedWindowDataset(
        val_records,
        input_stats,
        target_stats,
        args.cache_recordings,
    )
    test_dataset = PairedWindowDataset(
        test_records,
        input_stats,
        target_stats,
        args.cache_recordings,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.seed,
        args.num_workers,
        device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
        device,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
        device,
    )

    model = build_model(
        args.model,
        input_channels=len(input_names),
        target_channels=len(target_names),
        args=args,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Model:", args.model)
    print("Input channels:", list(input_names))
    print("Target channels:", list(target_names))
    print("Trainable parameters:", f"{parameter_count:,}")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            args.gradient_clip,
        )
        validation_metrics = evaluate(
            model,
            val_loader,
            loss_function,
            device,
            target_stats,
            target_names,
        )
        validation_loss = float(validation_metrics["standardized_mse"])
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": float(epoch),
                "train_standardized_mse": float(train_loss),
                "validation_standardized_mse": validation_loss,
                "validation_mean_nrmse": float(validation_metrics["mean_nrmse"]),
                "validation_mean_pearson_r": float(
                    validation_metrics["mean_pearson_r"]
                ),
                "learning_rate": float(current_lr),
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train MSE={train_loss:.6f} | "
            f"val MSE={validation_loss:.6f} | "
            f"val nRMSE={validation_metrics['mean_nrmse']:.4f} | "
            f"val r={validation_metrics['mean_pearson_r']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid model state")
    model.load_state_dict(best_state)

    validation_metrics = evaluate(
        model,
        val_loader,
        loss_function,
        device,
        target_stats,
        target_names,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        loss_function,
        device,
        target_stats,
        target_names,
    )

    checkpoint = {
        "model_name": args.model,
        "model_state_dict": best_state,
        "input_channel_names": list(input_names),
        "target_channel_names": list(target_names),
        "input_mean_uv": input_stats.mean.tolist(),
        "input_std_uv": input_stats.std.tolist(),
        "target_mean_uv": target_stats.mean.tolist(),
        "target_std_uv": target_stats.std.tolist(),
        "hidden_channels": list(args.hidden_channels),
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "window_samples": first_record.n_samples,
        "best_epoch": best_epoch,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")
    save_history(args.output_dir / "training_history.csv", history)
    save_json(
        args.output_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_validation_standardized_mse": best_validation_loss,
            "validation": validation_metrics,
            "test": test_metrics,
        },
    )

    print("Best epoch:", best_epoch)
    print("Validation mean correlation:", validation_metrics["mean_pearson_r"])
    print("Test mean correlation:", test_metrics["mean_pearson_r"])
    print("Saved run outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
