"""The faculty suite: every faculty's paired contrast on chosen families, in one command.

The policy is that every faculty is on unless it has shown harm, so the suite's reference arm is ``all-on`` and
its treatment arms are every faculty's arm (``all-on-<faculty>``; ``all-on-initiative``, all-on with the mind off,
for initiative, whose measure is the mind itself). A plan takes at most eight arms, so each family's
ablations are spread over balanced plans of at most seven, each with the reference. Each plan is the ordinary
``colonymind models paired plan`` and ``run`` (frozen, resumable, never replayed): the suite only writes their
commands and runs them, at most ``--streams`` plans at once, each plan's steps in order in one stream. A plan
already made is not made again, a run already started resumes its untouched episodes, and a plan is done when it
has a report and every arm-episode has started (a run that stops early still writes its report, so a report
alone is not completion; such a plan is reported as not finished). A plan's label comes from its family and its
position, not from its arms, so a plan already in the output directory is checked against the one asked for
(``frozen_mismatch``: its arms, dataset and settings) before it is skipped or resumed, and a suite with a plan
that differs runs nothing. ``colonymind report bench`` reads the results directory afterwards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REFERENCE = "all-on"
MAX_ARMS = 8
SUITE_VERSION = "paired-faculty-suite-1"


@dataclass(frozen=True)
class Spec:
    label: str
    dataset_dir: Optional[Path]
    dataset_version: Optional[str]
    arms: List[str]
    reference: str = REFERENCE


@dataclass(frozen=True)
class Resources:
    native_config: Path
    comparison_policy: Path
    container_image: str
    docker_host: Optional[str]
    native_binding: str
    repetitions: int = 1
    temperature: Optional[float] = None
    seeds: Optional[str] = None
    embedding_config: Optional[Path] = None


def faculty_arms() -> List[Tuple[str, str]]:
    """``(faculty, arm)`` for every faculty that can be switched off: the arm that differs from all-on in it alone.
    A mandatory faculty (no setting, no arm) has nothing to measure against."""
    from colonymind.reports.registry import FACULTIES
    from .paired import PROFILES
    arms = [(faculty.key, faculty.arm) for faculty in FACULTIES if faculty.arm]
    missing = [str(arm) for _, arm in arms if arm not in PROFILES]
    if missing:
        raise ValueError(f"no built-in profile for {', '.join(missing)}")
    return arms


def _chunks(items: Sequence[str], size: int) -> List[List[str]]:
    """``items`` in the fewest groups of at most ``size``, as even as they can be."""
    if not items:
        return []
    count = -(-len(items) // size)
    base, extra = divmod(len(items), count)
    groups, start = [], 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        groups.append(list(items[start:end]))
        start = end
    return groups


def _name(family: Any) -> str:
    return (Path(family).name if isinstance(family, Path) or os.sep in str(family) else str(family)) or "family"


def specs(families: Sequence[Any], *, label: str, faculties: Optional[Sequence[str]] = None,
          versions: Sequence[str] = ()) -> List[Spec]:
    """The suite's plans: for each family (a generated family directory, or an installed dataset version in
    ``versions``), the chosen faculties' ablations against all-on, in balanced plans of at most eight arms."""
    arms = faculty_arms()
    if faculties:
        known = dict(arms)
        unknown = [name for name in faculties if name not in known]
        if unknown:
            raise ValueError(f"unknown faculty: {', '.join(unknown)} (known: {', '.join(known)})")
        arms = [(name, known[name]) for name in dict.fromkeys(faculties)]   # in the order asked for
    result = []
    chosen = [*(Path(item) for item in families), *versions]
    names = [_name(family) for family in chosen]
    shared = sorted({name for name in names if names.count(name) > 1})
    if shared:
        # A plan's label and directory come from the family's name: two families sharing one would share a plan.
        raise ValueError(f"two families have the same name ({', '.join(shared)}); give each a distinct name")
    for family in chosen:
        for index, group in enumerate(_chunks([arm for _, arm in arms], MAX_ARMS - 1), start=1):
            result.append(Spec(label=f"{label}-{_name(family)}-{index}",
                               dataset_dir=family if isinstance(family, Path) else None,
                               dataset_version=None if isinstance(family, Path) else str(family),
                               arms=[REFERENCE, *group]))
    return result


def untouched(directory: Path) -> Optional[int]:
    """How many of a plan's arm-episodes have not started (no run directory yet); None without a readable plan."""
    try:
        manifest = json.loads((Path(directory) / "paired.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pairs = manifest.get("pairs") if isinstance(manifest, dict) else None
    paths = [member["path"] for pair in pairs or [] if isinstance(pair, dict)
             for member in (pair.get("arms") or {}).values() if isinstance(member, dict) and member.get("path")]
    return sum(not (Path(directory) / str(path)).exists() for path in paths)


def _seeds(seeds: Optional[str]) -> Any:
    try:
        return [int(value) for value in seeds.split(",")] if seeds else []
    except ValueError:
        return seeds


def _embedding(path: Optional[Path]) -> Any:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable"
    from .paired import EMBEDDING_KEYS
    return ({key: value[key] for key in (*EMBEDDING_KEYS, "api_key_env") if key in value}
            if isinstance(value, dict) else value)


def _policy(path: Path) -> Any:
    from .paired import _policy as read_policy
    try:
        return read_policy(path)
    except (OSError, ValueError):
        return "unreadable"


def frozen_mismatch(spec: Spec, directory: Path, resources: Resources) -> List[str]:
    """What the plan frozen in ``directory`` differs in from the one ``spec`` and ``resources`` ask for, as
    ``colonymind models paired plan`` would freeze it: its label, arms, reference arm, dataset (a generated family's
    path and content), native binding, repetitions, temperature, seeds, embedding endpoint, container image,
    comparison policy, arm profiles and evidence mode. Empty when they agree or no plan is frozen there.

    A plan's label comes from its family and position, not its arms: a suite for other faculties, another dataset
    or other settings names the same directory, and its plan would be skipped as done or resumed with its own arms.
    """
    path = Path(directory) / "paired.json"
    if not path.exists():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    options = manifest.get("options") if isinstance(manifest, dict) else None
    if not isinstance(options, dict):
        return [UNREADABLE]
    recipe = manifest.get("recipe") if isinstance(manifest.get("recipe"), dict) else {}
    container = recipe.get("container") if isinstance(recipe.get("container"), dict) else {}
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    comparison = manifest.get("comparison") if isinstance(manifest.get("comparison"), dict) else {}
    if spec.dataset_dir is not None:
        from .paired_cases import generated_content_hash
        family = Path(spec.dataset_dir).resolve()
        try:
            content: Any = generated_content_hash(family)
        except OSError:
            content = "unreadable"
        wanted: Any = (str(family), None, content)
        frozen: Any = (options.get("dataset_dir"), options.get("dataset_version"), dataset.get("source_sha256"))
    else:
        wanted = (None, spec.dataset_version)
        frozen = (options.get("dataset_dir"), options.get("dataset_version"))
    checks = [
        ("label", spec.label, manifest.get("label")),
        ("arms", list(spec.arms), options.get("arms")),
        ("reference arm", spec.reference, options.get("reference_arm")),
        ("dataset", wanted, frozen),
        ("native binding", resources.native_binding, options.get("native_binding")),
        ("repetitions", resources.repetitions, options.get("repetitions", 1)),
        ("temperature", None if resources.temperature is None else float(resources.temperature),
         options.get("temperature")),
        ("seeds", _seeds(resources.seeds), options.get("seeds")),
        ("embedding", _embedding(resources.embedding_config), options.get("embedding")),
        ("container image", (resources.container_image, resources.docker_host),
         (container.get("image"), container.get("docker_host"))),
        ("comparison policy", _policy(resources.comparison_policy), comparison.get("policy")),
        ("arm profiles", None, options.get("profiles")),
        ("evidence mode", "actual_inference", manifest.get("evidence_mode")),
    ]
    return [name for name, asked, found in checks if asked != found]


UNREADABLE = "unreadable plan"


def _refusal(spec: Spec, directory: Path, differs: List[str]) -> str:
    if differs == [UNREADABLE]:
        return f"{spec.label}: {directory} holds a plan the suite cannot read; choose another --output or --label"
    return (f"{spec.label}: {directory} holds a plan with another {', '.join(differs)} than the one asked for; "
            "choose another --output or --label")


def steps(spec: Spec, output: Path, resources: Resources, *, program: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """What is left of one plan: ``[(step, argv)]`` for ``plan`` (unless it is made) and ``run`` (with ``--resume``
    once started), nothing once it has a report and every arm-episode has started. A plan already made that is
    not the one asked for (``frozen_mismatch``) raises ``ValueError``: it is neither skipped nor resumed."""
    directory = Path(output) / spec.label
    differs = frozen_mismatch(spec, directory, resources)
    if differs:
        raise ValueError(_refusal(spec, directory, differs))
    if any(directory.glob("report-*.json")) and untouched(directory) == 0:
        return []
    common = ["--native-config", str(resources.native_config), "--comparison-policy", str(resources.comparison_policy),
              "--container-image", resources.container_image,
              *(["--docker-host", resources.docker_host] if resources.docker_host else []),
              "--output", str(directory)]
    found: List[Tuple[str, List[str]]] = []
    if not (directory / "paired.json").is_file():
        dataset = (["--dataset-dir", str(spec.dataset_dir)] if spec.dataset_dir is not None
                   else ["--dataset-version", str(spec.dataset_version)])
        extra = [*(["--repetitions", str(resources.repetitions)] if resources.repetitions != 1 else []),
                 *(["--temperature", str(resources.temperature)] if resources.temperature is not None else []),
                 *(["--seeds", resources.seeds] if resources.seeds else []),
                 *(["--embedding-config", str(resources.embedding_config)] if resources.embedding_config else [])]
        found.append(("plan", [*program, "models", "paired", "plan", *common, "--native-binding",
                               resources.native_binding, *dataset, "--arms", ",".join(spec.arms), "--reference-arm",
                               spec.reference, "--label", spec.label, *extra]))
    started = (directory / "runs").is_dir() and any((directory / "runs").iterdir())
    found.append(("run", [*program, "models", "paired", "run", *common, *(["--resume"] if started else [])]))
    return found


def _execute(argv: List[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(argv) + "\n")
        stream.flush()
        return subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def run_suite(plans: Sequence[Spec], output: Path, resources: Resources, *, streams: int,
              execute: Callable[[List[str], Path], int] = _execute,
              program: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Run every plan's remaining steps, at most ``streams`` plans at once, each plan's steps in order; a failed
    step ends its plan and the others go on. Writes ``suite.json`` (the suite as run) into ``output``.

    Raises ``ValueError``, before running or writing anything, when a plan already in ``output`` is not the one
    asked for (``frozen_mismatch``)."""
    if streams < 1:
        raise ValueError("--streams is at least 1")
    refused = [_refusal(spec, Path(output) / spec.label, differs) for spec in plans
               if (differs := frozen_mismatch(spec, Path(output) / spec.label, resources))]
    if refused:
        raise ValueError("\n".join(refused))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    program = list(program or [sys.executable, "-m", "colonymind"])
    (output / "suite.json").write_text(json.dumps({
        "version": SUITE_VERSION, "reference": REFERENCE, "streams": streams,
        "plans": [{**asdict(spec), "dataset_dir": str(spec.dataset_dir) if spec.dataset_dir else None}
                  for spec in plans]}, indent=2) + "\n")

    def one(spec: Spec) -> Dict[str, Any]:
        code = 0
        for step, argv in steps(spec, output, resources, program=program):
            code = execute(argv, output / f"{spec.label}.log")
            if code != 0:
                return {"label": spec.label, "ok": False, "step": step, "exit": code}
        left = untouched(output / spec.label)
        if left:
            # The run stopped early (its report is written all the same): not finished, whatever it exited with.
            return {"label": spec.label, "ok": False, "step": "run", "exit": code, "untouched": left}
        return {"label": spec.label, "ok": True, "step": None, "exit": 0}

    with ThreadPoolExecutor(max_workers=streams) as pool:
        return list(pool.map(one, plans))


def add_arguments(item: Any) -> None:
    item.add_argument("--dataset-dir", type=Path, action="append", default=[],
                      help="A generated family directory; repeat for several families")
    item.add_argument("--dataset-version", action="append", default=[],
                      help="An installed dataset version; repeat for several")
    item.add_argument("--output", type=Path, required=True, help="Private results directory: one plan directory each")
    item.add_argument("--native-config", type=Path, required=True, help="Private common Hermes provider configuration")
    item.add_argument("--comparison-policy", type=Path, required=True,
                      help="Versioned budget and environment declarations")
    item.add_argument("--container-image", required=True, help="Same digest-pinned image for every arm")
    item.add_argument("--docker-host", help="Declared Docker daemon; never mounted inside the benchmark")
    item.add_argument("--native-binding", required=True)
    item.add_argument("--faculties", help="Comma-separated faculties to measure (default: every faculty)")
    item.add_argument("--streams", type=int, default=1, help="Plans run at once: the benchmark streams the host "
                                                              "takes (default 1)")
    item.add_argument("--label", default="faculties", help="Prefix of every plan's label")
    item.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=1)
    item.add_argument("--temperature", type=float)
    item.add_argument("--seeds")
    item.add_argument("--embedding-config", type=Path)
    item.add_argument("--dry-run", action="store_true", help="Print the plans and their commands; run nothing")


def cli(args: Any) -> int:
    if not args.dataset_dir and not args.dataset_version:
        print("colonymind models paired suite: name at least one family (--dataset-dir or --dataset-version)",
              file=sys.stderr)
        return 2
    faculties = [name.strip() for name in args.faculties.split(",") if name.strip()] if args.faculties else None
    try:
        plans = specs(args.dataset_dir, label=args.label, faculties=faculties, versions=args.dataset_version)
    except ValueError as error:
        print(f"colonymind models paired suite: {error}", file=sys.stderr)
        return 2
    resources = Resources(native_config=args.native_config, comparison_policy=args.comparison_policy,
                          container_image=args.container_image, docker_host=args.docker_host,
                          native_binding=args.native_binding, repetitions=args.repetitions,
                          temperature=args.temperature, seeds=args.seeds, embedding_config=args.embedding_config)
    streams = max(1, int(args.streams))
    print(f"{len(plans)} plans, at most {streams} at once; reference arm {REFERENCE}.")
    if args.dry_run:
        refused = False
        for spec in plans:
            print(f"\n{spec.label}: {', '.join(spec.arms)}")
            try:
                for _, argv in steps(spec, args.output, resources, program=["colonymind"]):
                    print("  " + " ".join(argv))
            except ValueError as error:
                refused = True
                print(f"  refused: {error}")
        return 2 if refused else 0
    from .paired import ENDPOINT_WARNING
    print(ENDPOINT_WARNING, file=sys.stderr)
    try:
        results = run_suite(plans, args.output, resources, streams=streams)
    except ValueError as error:
        print(f"colonymind models paired suite: {error}", file=sys.stderr)
        return 2
    failed = [item for item in results if not item["ok"]]
    for item in failed:
        log = args.output / (item["label"] + ".log")
        if item.get("untouched"):
            print(f"{item['label']}: the run stopped with {item['untouched']} arm-episodes not started (see {log})")
        else:
            print(f"{item['label']}: {item['step']} exited {item['exit']} (see {log})")
    print(f"{len(results) - len(failed)} of {len(results)} plans finished. Read them with: "
          f"colonymind report bench {args.output}")
    return 0 if not failed else 1


__all__ = ["MAX_ARMS", "REFERENCE", "Resources", "SUITE_VERSION", "Spec", "add_arguments", "cli", "faculty_arms",
           "frozen_mismatch", "run_suite", "specs", "steps", "untouched"]
