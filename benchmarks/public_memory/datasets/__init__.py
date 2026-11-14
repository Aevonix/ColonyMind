"""Public memory datasets, read from their upstream files.

Each adapter downloads its upstream files once into a local cache, uses them only when their SHA-256 matches the
pin, and turns them into the same records (``records.py``): conversations of dated sessions in time order,
questions with the time they are asked, gold sessions and gold turns, and abstention ids. ``subsets.py`` draws
seeded, stratified subsets whose ids are recorded. The data itself is never committed.

    python -m benchmarks.public_memory.datasets fetch            # from the repository root
    python -m benchmarks.public_memory.datasets verify longmemeval_s
    python -m benchmarks.public_memory.datasets subset longmemeval_s --size 60 --seed 1 --out subset.json

The adapters and their sources are listed in ``README.md`` beside this file.
"""

NAMES = ("longmemeval_s", "beam", "precisionmembench")
