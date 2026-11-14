"""Public injection sets for the tool-result screen: read from pinned upstream files, never committed.

Each source is a file at an immutable upstream revision, used only when its bytes hash to the pin (the pinned
download of ``benchmarks/public_memory``) and only under a license on ``LICENSES``. Every item a source gives is a
text written to make an AI assistant do something it was not asked to do, so under the screen's question ("does
this text contain instructions aimed at an AI assistant?") each one is labelled yes; the screen's clean results
are the synthetic set's. Where a source also labels ordinary requests to an assistant as benign, those are left
out: they are instructions to an assistant too, not clean tool results.

    python benchmarks/decisions/public_sets.py fetch        # once, into the cache (``COLONYMIND_BENCH_DATA_DIR``)

The items carry ``subset: public`` and ``source: public:<name>``; they are reported apart from the frozen sets.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
LICENSES = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
SCREEN_CHARS = 1200


def _pinned():
    """The pinned download of ``benchmarks/public_memory``, loaded once per process (one ``PinError``)."""
    if "decisions_pinned" in sys.modules:
        return sys.modules["decisions_pinned"]
    path = HERE.parent / "public_memory" / "datasets" / "pinned.py"
    spec = importlib.util.spec_from_file_location("decisions_pinned", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module             # a dataclass reads its module while it is built
    spec.loader.exec_module(module)
    return module


pinned = _pinned()


@dataclass(frozen=True)
class Source:
    name: str
    license: str
    pins: Tuple[Any, ...]
    label_column: Optional[str]     # None: every row is an injection; else the rows whose label is ``injection``
    injection: Any = 1
    text_column: str = "text"


SOURCES: Tuple[Source, ...] = (
    Source(name="prompt-injections", license="apache-2.0", label_column="label", pins=(
        pinned.Pin(name="prompt-injections-test.parquet",
                   url="https://huggingface.co/datasets/deepset/prompt-injections/resolve/"
                       "4f61ecb038e9c3fb77e21034b22511b523772cdd/data/test-00000-of-00001-701d16158af87368.parquet",
                   sha256="39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e", size=10892),
        pinned.Pin(name="prompt-injections-train.parquet",
                   url="https://huggingface.co/datasets/deepset/prompt-injections/resolve/"
                       "4f61ecb038e9c3fb77e21034b22511b523772cdd/data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                   sha256="2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04", size=40323))),
    Source(name="jailbreak-classification", license="apache-2.0", label_column="type", injection="jailbreak",
           text_column="prompt", pins=(
               pinned.Pin(name="jailbreak-classification-test.csv",
                          url="https://huggingface.co/datasets/jackhhao/jailbreak-classification/resolve/"
                              "2f2ceeb39658696fd3f462403562b6eea5306287/default/jailbreak_dataset_test.csv",
                          sha256="a546ce540973d8f5ef87ce3be4bd21ee5923154b9dda86eba5a12c7fa00829da", size=453350),)),
)


class LicenseError(ValueError):
    """A source whose license is not one the bake-off may read."""


def _rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    import pyarrow.parquet as parquet
    return parquet.read_table(path).to_pylist()


def items(source: Source, directory: Optional[Path] = None, *, download: bool = False) -> List[Dict[str, Any]]:
    """``source``'s screen items from its pinned files (``PinError`` for a missing or changed file)."""
    if source.license not in LICENSES:
        raise LicenseError(f"{source.name}: license {source.license} is not one the bake-off reads")
    out: List[Dict[str, Any]] = []
    seen = set()
    for pin in source.pins:
        for row in _rows(pinned.fetch(pin, directory, download=download)):
            if source.label_column is not None and row.get(source.label_column) != source.injection:
                continue
            text = " ".join(str(row.get(source.text_column) or "").split())
            if not text or len(text) > SCREEN_CHARS or text in seen:
                continue
            seen.add(text)
            out.append({"point": "tool_screen", "fields": {"tool": "web_extract", "text": text}, "gold": "yes",
                        "source": f"public:{source.name}", "subset": "public"})
    return out


def screen(directory: Optional[Path] = None, *, download: bool = False) -> List[Dict[str, Any]]:
    return [item for source in SOURCES for item in items(source, directory, download=download)]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("fetch", "verify"))
    parser.add_argument("--dir")
    args = parser.parse_args(argv)
    got = screen(Path(args.dir) if args.dir else None, download=args.step == "fetch")
    print(f"{args.step}: {len(got)} public screen items from {len(SOURCES)} pinned sources")


if __name__ == "__main__":
    sys.exit(main())
