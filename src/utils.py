from typing import Iterable
from argparse import ArgumentParser

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def iter_progress(iterable: Iterable, desc: str | None = None, leave: bool = True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)

def log_stage(message: str):
    print(f"[stage] {message}")

def parse_args():
    parser = ArgumentParser(description="Generate a synthetic dataset using a fitted GMM.")
    parser.add_argument(
        "input",
        nargs="?",
        default="../datasets/filtered.csv",
        help="Path to the CSV containing the real transfer records. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output",
        default="../output/output.csv",
        help="Path to write the synthetic CSV. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-device",
        default="cuda",
        help="Torch device string to use when --use-gpu is set. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="Batch size (in samples) for EM responsibilities. Use 0 to process the full dataset. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--max-cap",
        action="store_true",
        help="Apply max-capping during GPU sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training and sampling. Defaults to %(default)s.",
    )
    return parser.parse_args()
