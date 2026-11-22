"""Check "github": new activity on a list of repositories since the last look.

Reads each repository's event listing with GET requests only. Settings (``github`` in ``checks.json``):
``repositories`` (``owner/name`` entries), ``token_env`` (the name of the environment variable that
holds a read token; without one the requests are anonymous) and ``api_url``. The first look at a
repository only notes its newest event. The line for the scheduler holds a count, never a title.
"""

import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_base  # noqa: E402

TOKEN_ENV = "COLONYMIND_CHECK_GITHUB_TOKEN"
API_URL = "https://api.github.com"


def _item(repository, event):
    payload = event.get("payload") or {}
    about = payload.get("issue") or payload.get("pull_request") or payload.get("release") or {}
    return {"repository": repository, "type": event.get("type"), "action": payload.get("action"),
            "by": (event.get("actor") or {}).get("login"), "at": event.get("created_at"),
            "ref": payload.get("ref"), "number": about.get("number"),
            "title": about.get("title") or about.get("name"), "url": about.get("html_url")}


def look(since, settings, until=None):
    config = settings.get("github") or {}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "colonymind-check"}
    token = os.environ.get(config.get("token_env") or TOKEN_ENV, "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    api = (config.get("api_url") or API_URL).rstrip("/")
    since = since if isinstance(since, dict) else {}
    marker, found, problem = dict(since), [], ""
    for repository in config.get("repositories") or []:
        try:
            events = check_base.get_json(
                "%s/repos/%s/events?per_page=30" % (api, urllib.parse.quote(repository, safe="/")), headers)
            ids = [int(event["id"]) for event in events]
        except Exception as error:  # noqa: BLE001 - one unreadable repository does not hide the others
            problem = type(error).__name__.lower()
            continue
        last = since.get(repository)
        newest = max(ids + [last or 0])
        if last is not None:
            top = (until or {}).get(repository, newest)
            found += [_item(repository, event) for event in reversed(events) if last < int(event["id"]) <= top]
        marker[repository] = newest
    return marker, found, problem


if __name__ == "__main__":
    sys.exit(check_base.main("github", look))
