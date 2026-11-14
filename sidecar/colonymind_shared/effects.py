"""Classify tool effects and validate declarations for the sidecar and host adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Recognizers of the four floor classes in what an executor runs (a shell command, code, a browser or delegated
# instruction). They only ever hold: a command they name is that floor effect wherever its paths point, and a
# command they do not name is an unbounded effect, which runs only in a task the owner approved. Nothing here reads
# a command to set it free (a shell spells a path, a hook or a redirection in more ways than a reader sees): no
# command is the worker's own, and no standing grant covers a shell, code, browser or delegation call, each of
# which is the owner's one-tap ask whose yes lets exactly that call run once. A recurring action the owner wants
# to grant is a declared action (``mind.actions``, below): a fixed command with checked arguments.
FLOOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "money_movement": re.compile(
        r"\b(?:wire|transfer|send|move)\s+(?:\$|money|funds|payment)|"
        r"\b(?:purchase|buy|pay|spend|subscribe)\b.{0,30}\$|"
        r"\bpayment\s+(?:of|for)\b|\$\d{2,}|"
        r"\bstripe\s+(?:charges|payment_intents|payouts|transfers|refunds|invoices|subscriptions|topups)\s+"
        r"(?:create|pay|confirm|capture)\b", re.IGNORECASE),
    "irreversible_deletion": re.compile(
        r"\b(?:rm\s+-rf|drop\s+(?:table|database)|delete\s+permanently|"
        r"wipe|purge\s+all|force[- ]?push|push\s+--force(?:-with-lease)?|"
        r"delete\s+(?:the\s+)?(?:repo|repository|database|account)|"
        r"erase\s+(?:all|everything))\b|"
        r"(?:^|[\s;&|`(])(?:rm|rmdir|unlink|shred)\s+\S|(?:^|[\s;&|`(])truncate\s+-|"
        r"\bfind\b[^;&|\n]*\s-delete\b|"
        r"\bgit\s+(?:-C\s+\S+\s+)?(?:clean\s+-\w*f|branch\s+-D\b|reset\s+--hard\b|"
        r"push\b[^;&|\n]*(?:\s--force\b|\s-f\b|\s\+\S))|"
        r"\b(?:drop|truncate)\s+(?:table|database|schema)\b|\bdelete\s+from\s+\w+(?![^;\n]*\bwhere\b)|"
        r"\b(?:docker|podman)\s+(?:rm|rmi|volume\s+rm|system\s+prune|container\s+rm|image\s+rm)\b|"
        r"\bkubectl\s+delete\b|\bhelm\s+(?:uninstall|delete)\b|"
        r"\baws\s+s3\s+rm\b|\baws\s+s3api\s+delete-|\bgsutil\s+(?:-m\s+)?rm\b|\bgcloud\b[^;&|\n]*\sdelete\b|"
        r"\baz\b[^;&|\n]*\sdelete\b|\brclone\s+(?:delete|purge|deletefile)\b|\bflush(?:all|db)\b|"
        r"\bos\.(?:remove|unlink|rmdir|removedirs)\s*\(|\bshutil\.rmtree\s*\(|\.unlink\s*\(|\.rmdir\s*\(",
        re.IGNORECASE),
    "credential_change": re.compile(
        r"\b(?:rotate|change|reset|revoke|create)\b.{0,40}\b(?:credential|"
        r"password|api[_ ]?key|secret|token|ssh[- ]?key|certificate)\b|"
        r"\bsecurity\s+settings?\b|"
        r"\bssh-keygen\b|\bssh-add\s+-[dD]\b|\bopenssl\s+(?:genrsa|genpkey|ca\b|req\b[^;&|\n]*\s-(?:new|x509)\b)|"
        r"\b(?:certbot|acme\.sh|lego)\b[^;&|\n]*(?:\bcertonly\b|\brenew\b|\brevoke\b|\bdelete\b|--issue\b|--renew\b)|"
        r"\bgpg2?\b[^;&|\n]*--(?:gen-key|full-gen-key|quick-gen-key|generate-key|delete-secret-keys?|delete-keys?)\b|"
        r"\bsecurity\s+(?:add|delete)-(?:generic|internet)-password\b|\bsecurity\s+delete-certificate\b|"
        r"\bsecret-tool\s+(?:store|clear)\b|\bop\s+item\s+(?:create|edit|delete)\b|\bbw\s+(?:create|edit|delete)\b|"
        r"\bpass\s+(?:insert|edit|rm|generate|mv)\b|"
        r"\bgh\s+(?:auth\s+(?:login|logout|refresh|token)|secret\s+(?:set|delete|remove)|"
        r"(?:ssh|gpg)-key\s+(?:add|delete))\b|\bglab\s+auth\b|"
        r"\baws\s+(?:iam\s+(?:create|delete|update)-(?:access-key|login-profile|user)|"
        r"secretsmanager\s+(?:create|put|update|delete)|configure\s+set)\b|"
        r"\bgcloud\s+(?:auth\s+\S+|iam\s+service-accounts\s+keys\s+(?:create|delete)|"
        r"secrets\s+(?:create|delete|versions\s+add))\b|"
        r"\baz\s+(?:login|ad\s+(?:sp|app)\s+credential|keyvault\s+secret\s+(?:set|delete))\b|"
        r"\bkubectl\s+(?:create|apply)\b[^;&|\n]*\bsecret\b|"
        r"\bvault\s+(?:kv\s+(?:put|delete|destroy)|write|login|token\s+create)\b|"
        r"\b(?:passwd|chpasswd|htpasswd)\b|\bkeytool\s+-(?:genkey|genkeypair|delete|importcert)\b|"
        r"(?:>>?\s*|(?:\btee\b(?:\s+-a)?|\bcp\b|\bmv\b|\binstall\b)\s+)[^;&|\n]*?"
        r"(?:\.ssh/|\.pem\b|\.key\b|\.env\b|credentials|\.netrc\b|\.npmrc\b|\.pypirc\b)", re.IGNORECASE),
    "bulk_third_party_messaging": re.compile(
        r"\b(?:bulk|mass|broadcast|blast|everyone|all\s+contacts)\b.{0,40}"
        r"\b(?:message|text|email|sms|dm)\b|"
        r"\b(?:message|text|email|sms|dm)\b.{0,40}\b(?:bulk|mass|broadcast|"
        r"blast|everyone|all\s+contacts)\b|"
        r"\b(?:mailx?|sendmail|mutt|msmtp|swaks)\b[^;&|\n]*\s(?:-b|--bcc)\s|"
        r"\bxargs\b[^;&|\n]*\b(?:mailx?|sendmail|mutt|msmtp|swaks)\b|"
        r"\bfor\b[\s\S]{0,200}?(?:\bdo\b|:\s*\n)[\s\S]{0,300}?"
        r"(?:\b(?:mailx?|sendmail|mutt|msmtp|swaks)\b|\.(?:sendmail|send_message|send_mail)\s*\()|"
        r"\b(?:mailmerge|send-bulk-templated-email|send-bulk-email|bulk_send)\b", re.IGNORECASE),
}
# Where a request moves money by reaching it (``mind.floor.money_hosts`` replaces this list).
MONEY_HOSTS = ("api.stripe.com", "api.paypal.com", "api-m.paypal.com", "api.braintreegateway.com",
               "connect.squareup.com", "api.wise.com", "api.transferwise.com", "api.checkout.com", "api.adyen.com",
               "api.mollie.com", "api.razorpay.com", "api.gocardless.com", "api.coinbase.com", "api.revolut.com")
# Tools whose effects cannot be bounded per call: what they run is matched by the recognizers above, and a
# command none of them names is unbounded. Every other ``browser_*`` tool is one too.
EXECUTOR_TOOLS = frozenset({
    "terminal", "process_manage", "process", "execute_code", "delegate_task", "computer_use", "browser_exec",
    "read_terminal", "close_terminal", "desktop_preview", "drive_preview", "annotate_preview", "read_window_below",
    "focus_pane", "setup_mcp", "gui_tour", "show_tip", "apply_layout", "open_preview", "close_preview", "read_preview",
})
# The argument that holds what such a tool runs, where the call has other arguments too: Python for the browser
# (``browser_exec``), JavaScript in the page (``browser_console``; ``browser_cdp``, whose ``params`` carry it), text
# for the standard input of a process on the host (``process_manage``), text typed on the desktop, where a terminal
# window runs each line as a command (``computer_use``, whose action ``type`` carries it). The recognizers read it
# as written besides the JSON of the call: in JSON a line end is the two characters ``\n``, which glue the first
# word of a later line to an ``n``, so a step that starts there would go unseen.
RUN_ARGUMENTS = {"browser_exec": "code", "browser_console": "expression", "browser_cdp": "params",
                 "process_manage": "data", "computer_use": "text"}
# Tools that reach nothing outside the worker's own workspace, the kanban board and the sidecar.
EFFECT_FREE_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "x_search", "vision_analyze", "video_analyze",
    "session_search", "skills_list", "skill_view", "todo_list", "todo", "tool_search", "tool_describe", "clarify",
    "image_generate", "text_to_speech", "browser_vault_list", "ha_list_entities", "ha_get_state",
    "ha_list_services", "kanban_show", "kanban_get", "kanban_list", "kanban_complete", "kanban_block",
    "kanban_request_review", "kanban_request_changes", "kanban_heartbeat", "kanban_comment", "kanban_create",
    "kanban_link", "kanban_unblock", "kanban_attach", "kanban_attach_url", "kanban_attachments", "feishu_doc_read",
    "feishu_drive_list_comments", "feishu_drive_list_comment_replies", "yb_query_group_info", "yb_query_group_members",
    "yb_search_sticker",
})
MESSAGE_TOOLS = frozenset({
    "send_message", "react_to_message", "discord", "discord_admin", "yb_send_dm", "yb_send_sticker",
    "feishu_drive_reply_comment", "feishu_drive_add_comment",
})
# Hermes' vault calls. Its older sources register one tool per action (``browser_vault_fill``, ...); its newer ones
# register ONE tool, ``browser_vault``, whose ``action`` argument says which. The one tool is read as the older
# tool of its action (``read_as``), so every vault name and rule of the product is the older tool's.
VAULT_TOOL = "browser_vault"
VAULT_ACTIONS = ("list", "unlock", "fill", "save_login", "enter_code")
# ``mind.floor.tools`` entries: the class the owner names (``none`` or a floor class) and the argument names.
DECLARATION_KEYS = ("class", "amount", "currency", "default_currency", "target")
# ``mind.actions``: the owner's dedicated tools, called through ``colonymind_action``. Each is a fixed command (a list
# of words run without a shell) whose words written ``{name}`` are its declared arguments, each checked before
# anything runs: a string against its ``enum`` or whole against its ``pattern`` (never starting with ``-``: an
# option is the owner's word, not the caller's), a number within ``min`` and ``max``. ``class`` is what running it
# does (``none`` or a floor class) and ``amount``, ``currency``, ``default_currency`` and ``target`` name its
# arguments as a declaration's do. A standing grant names the action: a call's effect carries the action's name.
ACTION_TOOL = "colonymind_action"
ACTION_KEYS = ("description", "command", "args", "timeout")
ACTION_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")
ARG_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
ARG_KEYS = ("type", "enum", "pattern", "min", "max", "default")
ACTION_TIMEOUT = (120.0, 3600.0)   # seconds: an action's default run and the longest it may be given
# The tools the effect table reads itself (and every ``browser_`` and ``colonymind_`` tool): a declaration or an
# action never takes one's name, so declaring the shell sets no command free.
RESERVED_TOOLS = EXECUTOR_TOOLS | EFFECT_FREE_TOOLS | MESSAGE_TOOLS | frozenset({
    "write_file", "patch", "cronjob_manage", "skill_manage", "memory", "outbox"})


def table_reads(tool: Any) -> bool:
    """A tool the effect table reads itself: its own reading stands over any declaration of it."""
    tool = str(tool or "")
    return tool in RESERVED_TOOLS or tool.startswith(("browser_", "colonymind_"))


def read_as(tool: Any, args: Any) -> str:
    """The name a call is read by: the tool's own, except that Hermes' one vault tool is read as the older tool of
    its action (``browser_vault`` with ``action: "fill"`` is ``browser_vault_fill``). The effect table, the guard,
    the deny list and a grant all read it so. The action is the exact word, as Hermes reads it: with a missing or
    unknown one the call stays the browser call it is. ``target`` (the managed browser or the in-app one) changes
    nothing: Hermes runs the same operation behind the same origin check for both."""
    tool = str(tool or "")
    action = args.get("action") if tool == VAULT_TOOL and isinstance(args, Mapping) else None
    return f"{VAULT_TOOL}_{action}" if isinstance(action, str) and action in VAULT_ACTIONS else tool


@dataclass(frozen=True)
class Effect:
    """What one tool call does outside the worker's own workspace: a floor class, ``unbounded`` (an executor call
    no recognizer names, or a tool the table does not know) or ``message``. ``call`` is the complete call in
    canonical form (every argument, and the directory a command runs in): a permit binds to it. ``problem`` says
    why the call cannot be read the way the owner declared it (a declaration with a problem, an action nobody
    declared, arguments its checks refuse): no grant and no task's approval covers such a call."""

    cls: str
    tool: str
    target: str = ""
    amount: float | None = None
    currency: str = ""
    recipients: tuple[str, ...] = ()
    bounded: bool = True
    call: str = ""
    problem: str = ""

    @property
    def floor(self) -> bool:
        return self.cls in FLOOR_PATTERNS

    @property
    def digest(self) -> str:
        """The same call has the same digest: an ask for it is formed once and a yes lets exactly it run."""
        body = json.dumps([self.cls, self.tool, self.call or self.target, self.amount, self.currency.upper(),
                           sorted(self.recipients)], ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        what = "an unbounded effect" if self.cls == "unbounded" else self.cls.replace("_", " ")
        target = " ".join(self.target.split())
        target = target if len(target) <= 160 else target[:159] + "..."
        amount = f" ({self.amount:g} {self.currency or 'in an unknown currency'})" if self.amount is not None else ""
        people = f" to {len(self.recipients)} people" if len(self.recipients) > 1 else ""
        return f"{what}: {self.tool} {target!r}{amount}{people}" if target else f"{what}: {self.tool}{amount}{people}"

    def as_dict(self) -> dict[str, Any]:
        return {"cls": self.cls, "tool": self.tool, "target": self.target, "amount": self.amount,
                "currency": self.currency, "recipients": list(self.recipients), "bounded": self.bounded,
                "call": self.call, "problem": self.problem, "digest": self.digest, "text": self.describe()}


def floor_class(text: str) -> str | None:
    """The floor class this command text shows, if any."""
    value = str(text or "")
    return next((name for name, pattern in FLOOR_PATTERNS.items() if pattern.search(value)), None)


floor_match = floor_class


def money_host(text: str, hosts: Any = MONEY_HOSTS) -> str | None:
    """The payment provider host a command or request reaches, if any."""
    value = str(text or "").lower()
    for host in hosts or ():
        host = str(host).strip().lower()
        if host and re.search(r"(?:^|[^\w.-]|\.)" + re.escape(host) + r"(?:$|[^\w.-]|\.(?![\w-]))", value):
            return host
    return None


def finite_amount(value: Any) -> float | None:
    """An amount the floor can count: a finite number of zero or more (a number or its plain written form);
    anything else (text around it, NaN, infinity, a negative, a boolean) is an unknown amount."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def declaration_problem(spec: Any) -> str:
    """What is wrong with one ``mind.floor.tools`` entry ('' when nothing is): the class is named explicitly,
    ``none`` or a floor class; the other keys are argument names (``default_currency`` a three-letter code)."""
    if not isinstance(spec, dict):
        return "a declaration is a mapping with a class"
    cls = spec.get("class")
    if not isinstance(cls, str) or (cls != "none" and cls not in FLOOR_PATTERNS):
        return f"class is none or one of {', '.join(FLOOR_PATTERNS)}"
    unknown = sorted(str(key) for key in spec if key not in DECLARATION_KEYS)
    if unknown:
        return f"unknown field(s): {', '.join(unknown)}"
    if any(key in spec and not isinstance(spec[key], str) for key in DECLARATION_KEYS[1:]):
        return "amount, currency, default_currency and target name arguments (text)"
    if spec.get("default_currency") and not re.fullmatch(r"[A-Za-z]{3}", str(spec["default_currency"])):
        return "default_currency is a three-letter code"
    return ""


def _arg_value(arg: dict[str, Any], value: Any) -> tuple[str, str]:
    """One checked argument as its command word, or ('', what it must be): a string of one line its ``enum`` lists
    or its ``pattern`` matches whole, never starting with ``-``; a whole or finite number within its bounds."""
    if arg.get("type") == "string":
        if not isinstance(value, str) or not value or value.startswith("-") or len(value) > 1024 or \
                any(ord(char) < 32 for char in value):
            return "", "is text of one line, not empty and not starting with '-'"
        if "enum" in arg and value not in arg["enum"]:
            return "", f"is one of {', '.join(arg['enum'])}"
        if "pattern" in arg and not re.fullmatch(arg["pattern"], value):
            return "", f"matches {arg['pattern']} whole"
        return value, ""
    whole = arg.get("type") == "integer"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or \
            (whole and value != int(value)):
        return "", f"is {'a whole' if whole else 'a finite'} number"
    if ("min" in arg and value < arg["min"]) or ("max" in arg and value > arg["max"]):
        return "", f"is between {arg.get('min', 'any')} and {arg.get('max', 'any')}"
    text = str(int(value)) if value == int(value) else repr(float(value))
    return (text, "") if "e" not in text else ("", "is a plain number")


def _arg_problem(arg: Any) -> str:
    """What is wrong with one declared argument ('' when nothing is)."""
    if not isinstance(arg, dict) or arg.get("type") not in ("string", "integer", "number"):
        return "type is string, integer or number"
    unknown = sorted(str(key) for key in arg if key not in ARG_KEYS)
    if unknown:
        return f"unknown field(s): {', '.join(unknown)}"
    if arg["type"] == "string":
        if "min" in arg or "max" in arg:
            return "min and max bound numbers"
        if "enum" not in arg and "pattern" not in arg:
            return "a string argument has an enum or a pattern its value must match whole"
        if "enum" in arg and (not isinstance(arg["enum"], list) or not arg["enum"] or not all(
                isinstance(item, str) and item and not item.startswith("-") for item in arg["enum"])):
            return "enum lists the values it may take (text, none starting with '-')"
        if "pattern" in arg:
            try:
                re.compile(arg["pattern"])
            except (re.error, TypeError):
                return "pattern is a regular expression"
    else:
        if "enum" in arg or "pattern" in arg:
            return "enum and pattern check strings"
        for key in ("min", "max"):
            if key in arg and (isinstance(arg[key], bool) or not isinstance(arg[key], (int, float))
                               or not math.isfinite(arg[key])):
                return f"{key} is a finite number"
    why = _arg_value(arg, arg["default"])[1] if "default" in arg else ""
    return f"default {why}" if why else ""


def action_problem(name: Any, spec: Any) -> str:
    """What is wrong with one ``mind.actions`` entry ('' when nothing is): a name no tool the table reads has, a
    declaration's class and argument names, a ``description``, a ``command`` of words whose ``{name}`` words are
    its declared arguments (never its first word, the program), checked arguments and a ``timeout`` of at most an
    hour."""
    name = str(name or "")
    if not ACTION_NAME.fullmatch(name) or table_reads(name):
        return "an action's name is a lowercase word no other tool has"
    if not isinstance(spec, dict):
        return "an action is a mapping: class, description, command and args"
    problem = declaration_problem({key: value for key, value in spec.items() if key not in ACTION_KEYS})
    if problem:
        return problem
    if not isinstance(spec.get("description"), str) or not spec["description"].strip():
        return "description says what the action does"
    args = spec.get("args") if spec.get("args") is not None else {}
    if not isinstance(args, dict) or not all(isinstance(key, str) and ARG_NAME.fullmatch(key) for key in args):
        return "args maps argument names (lowercase words) to their checks"
    for key, arg in args.items():
        problem = _arg_problem(arg)
        if problem:
            return f"args.{key}: {problem}"
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(word, str) and word for word in command):
        return "command is a list of words, run without a shell"
    for index, word in enumerate(command):
        if ("{" in word or "}" in word) and (index == 0 or word != "{" + word[1:-1] + "}" or word[1:-1] not in args):
            return f"a command word with braces is one declared argument written {{name}}, never the program: {word}"
    for key in ("amount", "currency", "target"):
        if spec.get(key) and spec[key] not in args:
            return f"{key} names one of the action's args"
    if spec.get("amount") and args[spec["amount"]]["type"] == "string":
        return "amount names a number argument"
    timeout = spec.get("timeout", ACTION_TIMEOUT[0])
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= ACTION_TIMEOUT[1]:
        return f"timeout is a number of seconds, at most {ACTION_TIMEOUT[1]:g}"
    return ""


def action_values(spec: dict[str, Any], given: Any) -> tuple[dict[str, str], str]:
    """A call's arguments for a declared action, checked: each declared argument given (or its default) and no
    other; ({}, why) when one is missing, unknown or refused by its check."""
    args = spec.get("args") or {}
    given = {} if given is None else given
    if not isinstance(given, dict):
        return {}, "args is a mapping of the action's arguments"
    unknown = sorted(str(key) for key in given if key not in args)
    if unknown:
        return {}, f"unknown argument(s) {', '.join(unknown)}; it takes {', '.join(args) or 'none'}"
    values: dict[str, str] = {}
    for key, arg in args.items():
        if key not in given and "default" not in arg:
            return {}, f"{key} is required"
        values[key], why = _arg_value(arg, given[key] if key in given else arg["default"])
        if why:
            return {}, f"{key} {why}"
    return values, ""


def action_argv(spec: dict[str, Any], values: dict[str, str]) -> list[str]:
    """The command a declared action runs: its words, each ``{name}`` word replaced by that checked argument."""
    return [values[word[1:-1]] if word.startswith("{") and word.endswith("}") else word for word in spec["command"]]


def canonical_call(tool: str, args: dict[str, Any], cwd: Any = None) -> str:
    """The complete call a permit binds to: every argument as given and, for a command, the directory it runs in
    (a ``workdir`` that resolves is that directory; one whose directory cannot be told stays in the call as
    written, so another relative ``workdir`` is another call)."""
    body: dict[str, Any] = {"tool": str(tool or ""), "args": dict(args)}
    if tool == "terminal":
        workdir = body["args"].pop("workdir", None)
        body["cwd"] = str(cwd) if cwd else None
        if not cwd and workdir:
            body["workdir"] = str(workdir)
    return json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)


def _texts(value: Any) -> list[str]:
    """Every text in an argument: the string itself, or the strings anywhere in a mapping's values or a list. Of a
    text that is JSON (``browser_cdp``'s ``params`` can arrive so: Hermes makes the object of them only after the
    hook has run) the texts inside the value it parses to are read as well; one that does not parse is read as it
    is."""
    if isinstance(value, str):
        try:
            return [value, *_texts(json.loads(value))] if value else []
        except (ValueError, RecursionError):
            return [value]
    if isinstance(value, Mapping):
        value = list(value.values())
    return [text for item in value for text in _texts(item)] if isinstance(value, (list, tuple)) else []


def executor_text(tool: str, args: dict[str, Any]) -> str:
    """What the recognizers read of an executor call: a command or a program as written; of any other call the
    JSON of its arguments and, on a new line each, the texts of the argument it runs (``RUN_ARGUMENTS``) as
    written. More text can only hold more."""
    if tool == "terminal":
        return str(args.get("command") or "")
    if tool == "execute_code":
        return str(args.get("code") or "")
    name = RUN_ARGUMENTS.get(tool)
    return "\n".join([json.dumps(args, ensure_ascii=False, sort_keys=True), *(_texts(args.get(name)) if name else [])])


def effect_of(tool: str, args: Any, cwd: Any = None, *, context: Any = None, declared: Any = None,
              actions: Any = None, money_hosts: Any = None) -> Effect | None:
    """The effect table: what one tool call does outside the worker's own workspace, or None.

    ``cwd`` is the directory a command runs in (None when it is not known), part of the call a permit binds to;
    ``context`` carries what only the plugin can read (``vault_kind`` of a vault item, ``own_job`` for a cron job
    this task created); ``declared`` is ``mind.floor.tools`` (the owner's classes for tools the table does not
    know), ``actions`` is ``mind.actions`` (the owner's dedicated tools, called through ``colonymind_action``) and
    ``money_hosts`` is ``mind.floor.money_hosts``. An amount is only ever a declared tool's or action's checked
    argument. What an executor runs is only ever held by what the recognizers name, never set free."""
    tool, args = str(tool or ""), args if isinstance(args, dict) else {}
    context = context if isinstance(context, dict) else {}
    declared = declared if isinstance(declared, dict) else {}
    actions = actions if isinstance(actions, dict) else {}
    hosts = MONEY_HOSTS if money_hosts is None else tuple(money_hosts)
    workdir = str(args.get("workdir") or "") if tool == "terminal" else ""
    here = (workdir if os.path.isabs(workdir) else os.path.join(str(cwd), workdir) if cwd else "") if workdir \
        else str(cwd or "")
    call = canonical_call(tool, args, here)
    tool = read_as(tool, args)      # Hermes' one vault tool is the older tool of its action; the call stays as given
    if tool == ACTION_TOOL:
        name = str(args.get("action") or "")
        spec = actions.get(name)
        problem = f"no action {name!r} is declared" if spec is None else action_problem(name, spec)
        values, why = ({}, "") if problem else action_values(spec, args.get("args"))
        if problem or why:
            return Effect("unbounded", tool, target=name, bounded=False, call=call,
                          problem=f"{name or 'the action'}: {problem or why}")
        if spec["class"] == "none":
            return None
        currency = str(values.get(spec.get("currency") or "") or spec.get("default_currency") or "").upper()
        return Effect(spec["class"], name, target=values.get(spec.get("target") or "", ""),
                      amount=finite_amount(values[spec["amount"]]) if spec.get("amount") else None,
                      currency=currency if re.fullmatch(r"[A-Z]{3}", currency) else "", call=call)
    if tool in declared and not table_reads(tool):
        spec = declared[tool]
        problem = declaration_problem(spec)
        if problem:
            return Effect("unbounded", tool, target=json.dumps(args, ensure_ascii=False, sort_keys=True)[:2000],
                          bounded=False, call=call, problem=f"the declaration of {tool} has a problem: {problem}")
        if spec["class"] == "none":
            return None
        currency = str(args.get(spec.get("currency") or "") or spec.get("default_currency") or "").strip().upper()
        return Effect(spec["class"], tool, target=str(args.get(spec.get("target") or "") or ""),
                      amount=finite_amount(args.get(spec["amount"])) if spec.get("amount") else None,
                      currency=currency if re.fullmatch(r"[A-Z]{3}", currency) else "", call=call)
    if tool in EFFECT_FREE_TOOLS or tool.startswith("colonymind_"):
        return None
    if tool in MESSAGE_TOOLS:
        target = str(args.get("contact_id") or args.get("target") or args.get("chat_id") or args.get("to") or "")
        return Effect("message", tool, target=target, recipients=(target,) if target else (), call=call)
    if tool in ("write_file", "patch"):
        return None     # confined to the workspace by the plugin before anything else (outside is refused)
    action = str(args.get("action") or "").strip().lower()
    if tool == "cronjob_manage":
        if action in ("list", "pause", "resume", "resnap", ""):
            return None
        job = str(args.get("job_id") or "")
        if action in ("remove", "delete"):
            return None if context.get("own_job") else \
                Effect("irreversible_deletion", tool, target=f"cron job {job}", call=call)
        prompt = str(args.get("prompt") or args.get("script") or "")
        return Effect("unbounded", tool, target=f"cron job {job} {action}: {prompt}".strip(), bounded=False, call=call)
    if tool in ("skill_manage", "memory"):
        # One call or a batch (``operations``): any removal in it deletes what the worker did not make.
        removals = ("delete", "remove", "remove_file") if tool == "skill_manage" else ("remove",)
        steps = [args] + [item for item in args.get("operations") or [] if isinstance(item, dict)]
        gone = [step for step in steps if str(step.get("action") or "").strip().lower() in removals]
        if not gone:
            return None
        what = [str(step.get("file_path") or step.get("name") or step.get("old_text") or args.get("name") or "")
                for step in gone]
        return Effect("irreversible_deletion", tool, target=f"{'skill' if tool == 'skill_manage' else 'memory'} "
                      + "; ".join(what), call=call)
    if tool == "browser_vault_fill":
        if str(context.get("vault_kind") or "") in ("login", "address"):
            return None
        return Effect("money_movement", tool, target=f"vault item {args.get('handle') or ''}".strip(), call=call)
    if tool == "browser_vault_save_login":
        return Effect("credential_change", tool, target=str(args.get("origin") or args.get("url") or ""), call=call)
    if tool in EXECUTOR_TOOLS or tool.startswith("browser_"):
        text = executor_text(tool, args)
        if money_host(text, hosts):
            return Effect("money_movement", tool, target=text, call=call)
        cls = floor_class(text)
        return Effect(cls, tool, target=text, call=call) if cls else \
            Effect("unbounded", tool, target=text, bounded=False, call=call)
    return Effect("unbounded", tool, target=json.dumps(args, ensure_ascii=False, sort_keys=True)[:2000],
                  bounded=False, call=call)
