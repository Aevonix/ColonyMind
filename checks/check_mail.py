"""Check "mail": new mail in the agent's own mailbox since the last look.

Reads the engine's local route ``GET /v1/agent-mail/new``. The first look only notes where the mailbox
stands. The line for the scheduler holds a count, never a sender, a subject or mail text.
"""

import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_base  # noqa: E402

PAGE = 20  # most mail one run reports; the rest waits for the next run


def look(since, settings, until=None):
    url, headers = check_base.engine(settings)
    query = "" if since is None else "?" + urllib.parse.urlencode({"since": since, "limit": PAGE})
    page = check_base.get_json(url + "/v1/agent-mail/new" + query, headers, direct=True)
    messages = page["messages"]
    if until is not None:
        messages = [message for message in messages if message["sequence"] <= until]
    return page["marker"], messages


if __name__ == "__main__":
    sys.exit(check_base.main("mail", look))
