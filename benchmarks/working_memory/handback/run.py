"""The runner: every arm for every question of a frozen set, against one model on a loopback server.

    python -m handback.run --base-url http://127.0.0.1:8000/v1 --model <served name> --tokenizer <dir> \\
        --out <dir> --model-revision <rev> [--seeds 1301 ...]

One model answers every arm (greedy decoding, thinking off). The runner refuses any base URL that is not loopback,
ignores proxy settings and follows no redirect. For each child, in order: its handoff summary and its excerpt
selection (both under the child's cache salt), then every consult of that child (the same salt, so a prefix cache
can serve the child's exact context). Then every question through the parent prompt, each arm under a salt of its
own (a cold read). Every salt carries the run's nonce (drawn when the run starts and kept in ``run.json``), so two
runs of the same sets against one server never share a cached prefix; a parent request draws a fresh salt on every
try, a retry and a resumed request included, so a request that reached the server before it failed cannot warm its
own cold read. The server must report cached prompt tokens (vLLM: ``--enable-prompt-tokens-details``) for the
consult's cost line; without them the cached count is 0. Outputs, resumable: ``handbacks.json``, ``consults.jsonl``,
``answers.jsonl`` per set.

A run has an identity: the served model and its revision, the tokenizer's files, the set's version, seed and hash,
a fingerprint of every prompt the run sends, the decoding settings, the output caps and the excerpt budget. It is
written to ``run.json`` in the set's directory before the first request, and a set's outputs are resumed only under
the same identity (and never when they have none beside them), so no run mixes or relabels another's rows. The
output directory's ``manifest.json`` holds the identity once and every start of the run.

Output token caps (my assumptions): ``MAX_TOKENS``.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlsplit

if __package__ in (None, ""):  # run as a script: import the package from its parent directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handback import arms  # noqa: E402
from handback.sets import load_frozen_set, set_hash, task_text, WORKER_SYSTEM  # noqa: E402

MAX_TOKENS = {"summary": 1024, "excerpts": 512, "consult": 256, "parent": 64}
SEED = 0
#: Greedy decoding with thinking off, the same for every request of a run.
DECODING = {"temperature": 0.0, "top_p": 1.0, "seed": SEED, "enable_thinking": False}
RUN_PROTOCOL = "wm-handback-run-1"
#: The tokenizer files that decide its token counts (weights and the rest are not read).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "tokenizer.model", "chat_template.jinja", "chat_template.json")
TIMEOUT_S = 900
#: Tries per request before it is recorded as an error, which scores as a wrong answer (my assumption).
TRIES = 3


def check_base_url(url: str) -> str:
    """``url`` without its trailing slash when it names a loopback host over HTTP; otherwise ValueError."""
    parts = urlsplit(str(url))
    host = parts.hostname
    loopback = False
    if parts.scheme in ("http", "https") and host:
        if host == "localhost":
            loopback = True
        else:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
    if not loopback:
        raise ValueError("the runner only talks to a loopback base URL (127.0.0.0/8, ::1 or localhost)")
    return str(url).rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("the runner follows no redirect (loopback only)")


def request_body(model: str, messages: List[Dict[str, Any]], *, max_tokens: int, salt: Optional[str]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": DECODING["temperature"],
                            "top_p": DECODING["top_p"], "seed": DECODING["seed"], "max_tokens": max_tokens,
                            "chat_template_kwargs": {"enable_thinking": DECODING["enable_thinking"]}}
    if salt:
        body["cache_salt"] = salt
    return body


class Client:
    """An OpenAI-compatible chat endpoint on a loopback host."""

    def __init__(self, base_url: str, model: str, *, timeout: float = TIMEOUT_S):
        self.base_url = check_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def complete(self, messages: List[Dict[str, Any]], *, max_tokens: int, salt: Optional[str]) -> Dict[str, Any]:
        data = json.dumps(request_body(self.model, messages, max_tokens=max_tokens, salt=salt)).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with self._opener.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        choice = (payload.get("choices") or [{}])[0]
        return {"text": (choice.get("message") or {}).get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": details.get("cached_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens")}


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prompts_fingerprint() -> str:
    """A hash of every prompt a run sends, rendered for a tiny probe child and question through the same code the
    run uses (the instructions, the parent's system prompt and its layout, for every arm)."""
    units = [{"handle": "t1", "kind": "tool_result", "label": "Tool result (probe)", "text": "Probe text."}]
    child = {"id": "w1", "project": "Project Probe", "units": units, "work": "Probe work."}
    child["context"] = [{"role": "system", "content": WORKER_SYSTEM}, {"role": "user", "content": task_text(child)},
                        {"role": "assistant", "content": child["work"]}]
    question = {"id": "probe-q", "question": "Probe question?", "children": ["w1"]}
    the_set = {"children": [child], "questions": [question]}
    handbacks = {"w1": "Probe handback."}
    rendered = {"summary": arms.summary_request(child), "excerpts": arms.excerpt_request(child),
                "consult": arms.consult_request(child, question),
                "parent": {arm: arms.parent_messages(arm, the_set, question, summaries=handbacks, excerpts=handbacks,
                                                     consults=handbacks) for arm in arms.ARMS}}
    return _sha256(rendered)


def tokenizer_fingerprint(path: Any) -> str:
    """A hash of a local tokenizer directory's tokenizer files (their names and contents)."""
    folder = Path(path)
    files = [folder / name for name in TOKENIZER_FILES if (folder / name).is_file()]
    if not files:
        raise ValueError(f"no tokenizer files in {folder}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8") + b"\0" + hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def run_identity(model: str, model_revision: str, tokenizer: str) -> Dict[str, Any]:
    """What a run's rows depend on besides its set: ``tokenizer`` is the tokenizer's fingerprint."""
    return {"protocol": RUN_PROTOCOL, "model": model, "model_revision": model_revision, "tokenizer": tokenizer,
            "prompts_sha256": prompts_fingerprint(), "decoding": dict(DECODING), "max_tokens": dict(MAX_TOKENS),
            "excerpt_budget": arms.EXCERPT_BUDGET, "excerpt_handles": arms.EXCERPT_HANDLES, "tries": TRIES}


def _differences(stored: Dict[str, Any], wanted: Dict[str, Any]) -> List[str]:
    return sorted(key for key in set(stored) | set(wanted) if stored.get(key) != wanted.get(key))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def open_set_run(the_set: Dict[str, Any], identity: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """The set's run record in ``out_dir``: written before the first request, or read back when its identity is
    the same; ValueError for another identity, or for outputs with no identity beside them."""
    wanted = json.loads(json.dumps({**identity, "set_version": the_set["version"], "set_seed": the_set["seed"],
                                    "set_sha256": set_hash(the_set)}))
    path = out_dir / "run.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        differ = _differences(record.get("identity") or {}, wanted)
        if differ:
            raise ValueError(f"{path} holds another run identity ({', '.join(differ)} differ): refusing to resume")
        return record
    if any((out_dir / name).exists() for name in ("handbacks.json", "consults.jsonl", "answers.jsonl")):
        raise ValueError(f"{out_dir} holds outputs with no run identity beside them: refusing to resume")
    record = {"identity": wanted, "nonce": secrets.token_hex(8),
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(path, record)
    return record


def record_start(out_dir: Path, identity: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    """The output directory's manifest: the run identity, written once, and every start (ValueError when the
    directory belongs to another identity)."""
    path = Path(out_dir) / "manifest.json"
    wanted = json.loads(json.dumps(identity))
    manifest = {"identity": wanted, "starts": []}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        differ = _differences(manifest.get("identity") or {}, wanted)
        if differ:
            raise ValueError(f"{path} holds another run identity ({', '.join(differ)} differ): refusing to start")
    manifest["starts"] = list(manifest.get("starts") or []) + [
        {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seeds": list(seeds)}]
    _write_json(path, manifest)
    return manifest


def child_salt(the_set: Dict[str, Any], cid: str, nonce: str) -> str:
    """The salt a child's summary, selection and consults share within one run."""
    return f"wm-{the_set['version']}-{the_set['seed']}-{nonce}-{cid}"


def parent_salt(the_set: Dict[str, Any], qid: str, arm: str, nonce: str) -> Callable[[], str]:
    """A parent request's salt: a fresh one on every call, so every try is a cold read."""
    base = f"wm-{the_set['version']}-{the_set['seed']}-{nonce}-parent-{qid}-{arm}"
    return lambda: f"{base}-{secrets.token_hex(8)}"


def _timed(client: Any, messages: List[Dict[str, Any]], *, max_tokens: int,
           salt: Union[str, Callable[[], str]]) -> Dict[str, Any]:
    """One request, timed; a transport failure is tried again, and after ``TRIES`` it is an empty answer with
    its error recorded (never a refusal to go on). A refused URL is not a transport failure. A callable ``salt``
    gives each try its own salt."""
    for attempt in range(1, TRIES + 1):
        started = time.monotonic()
        try:
            result = dict(client.complete(messages, max_tokens=max_tokens, salt=salt() if callable(salt) else salt))
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == TRIES:
                return {"text": "", "error": type(error).__name__, "prompt_tokens": None, "cached_tokens": 0,
                        "completion_tokens": None, "wall_s": round(time.monotonic() - started, 3)}
            time.sleep(5 * attempt)
            continue
        result.setdefault("wall_s", round(time.monotonic() - started, 3))
        return result
    raise AssertionError("unreachable")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _usage(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: result.get(key) for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "wall_s",
                                             "finish_reason", "error") if key in result}


def run_set(the_set: Dict[str, Any], client: Any, count: Callable[[str], int], out_dir: Path, *,
            identity: Dict[str, Any], log: Callable[[str], None] = lambda message: None) -> Dict[str, Any]:
    if getattr(client, "model", identity["model"]) != identity["model"]:
        raise ValueError(f"the client serves model {client.model!r}, not the run identity's {identity['model']!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonce = open_set_run(the_set, identity, out_dir)["nonce"]
    handbacks_path, consults_path, answers_path = (out_dir / "handbacks.json", out_dir / "consults.jsonl",
                                                   out_dir / "answers.jsonl")
    handbacks: Dict[str, Any] = {}
    if handbacks_path.exists():
        handbacks = json.loads(handbacks_path.read_text(encoding="utf-8"))
    consults = {(row["question_id"], row["child"]): row for row in _jsonl(consults_path)}
    for child in the_set["children"]:
        cid = child["id"]
        salt = child_salt(the_set, cid, nonce)
        if cid not in handbacks:
            summary = _timed(client, arms.summary_request(child), max_tokens=MAX_TOKENS["summary"], salt=salt)
            selection = _timed(client, arms.excerpt_request(child), max_tokens=MAX_TOKENS["excerpts"], salt=salt)
            handles = arms.parse_handles(selection["text"], child)
            block, used = arms.excerpt_block(child, handles, count)
            handbacks[cid] = {"summary": summary["text"], "summary_usage": _usage(summary),
                              "selection": selection["text"], "selection_usage": _usage(selection),
                              "handles": handles, "handles_used": used, "excerpts": block,
                              "excerpt_tokens": count(block) if block else 0,
                              "context_tokens": summary.get("prompt_tokens")}
            handbacks_path.write_text(json.dumps(handbacks, indent=1, ensure_ascii=False), encoding="utf-8")
            log(f"{cid}: summary and excerpts ({len(used)} passages)")
        for question in the_set["questions"]:
            if cid not in question["children"] or (question["id"], cid) in consults:
                continue
            result = _timed(client, arms.consult_request(child, question), max_tokens=MAX_TOKENS["consult"], salt=salt)
            row = {"question_id": question["id"], "child": cid, "answer": result["text"], **_usage(result)}
            consults[(question["id"], cid)] = row
            _append(consults_path, row)
        log(f"{cid}: consults done")
    done = {(row["question_id"], row["arm"]) for row in _jsonl(answers_path)}
    summaries = {cid: entry["summary"] for cid, entry in handbacks.items()}
    excerpts = {cid: entry["excerpts"] for cid, entry in handbacks.items()}
    for question in the_set["questions"]:
        answers = {cid: consults[(question["id"], cid)]["answer"] for cid in question["children"]}
        for arm in arms.ARMS:
            if (question["id"], arm) in done:
                continue
            messages = arms.parent_messages(arm, the_set, question, summaries=summaries, excerpts=excerpts,
                                            consults=answers)
            result = _timed(client, messages, max_tokens=MAX_TOKENS["parent"],
                            salt=parent_salt(the_set, question["id"], arm, nonce))
            row = {"question_id": question["id"], "set_seed": the_set["seed"], "type": question["type"], "arm": arm,
                   "answer": result["text"], **_usage(result)}
            if arm == "consult":
                calls = [consults[(question["id"], cid)] for cid in question["children"]]
                row["consult_calls"] = [{key: call.get(key) for key in ("child", "prompt_tokens", "cached_tokens",
                                                                       "completion_tokens", "wall_s")}
                                        for call in calls]
                for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "wall_s"):
                    row[key] = round(sum(float(c.get(key) or 0) for c in calls) + float(row.get(key) or 0), 3)
            _append(answers_path, row)
    rows = _jsonl(answers_path)
    return {"seed": the_set["seed"], "answers": len(rows), "consults": len(consults), "children": len(handbacks)}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from handback.sets import frozen
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--tokenizer", required=True, help="the model's local tokenizer directory (excerpt budget)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--model-revision", required=True, help="the served weights' revision (run identity)")
    args = parser.parse_args(argv)
    client = Client(args.base_url, args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    seeds = args.seeds or [entry["seed"] for entry in frozen()["sets"]]
    identity = run_identity(args.model, args.model_revision, tokenizer_fingerprint(args.tokenizer))
    args.out.mkdir(parents=True, exist_ok=True)
    record_start(args.out, identity, seeds)
    for seed in seeds:
        the_set = load_frozen_set(seed)
        def log(message: str, seed: int = seed) -> None:
            print(f"[{time.strftime('%H:%M:%S')}] {seed} {message}", flush=True)

        summary = run_set(the_set, client, count, args.out / f"set-{seed}", identity=identity, log=log)
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
