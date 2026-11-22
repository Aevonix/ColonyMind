"""Pages the agent keeps on each person and on itself.

``store``  the lines and their sources (``page_lines`` in the source ledger's database)
``night``  the nightly rewrite, one stage of the nightly pass
``turn``   what of a page a turn carries
``api``    ``/v1/mind/pages``: the owner reads every page, the agent sets its own lines

A person's page describes the person and holds the agent's boundaries with them. The agent's own page holds
what it is like and its likes and dislikes. The owner can read every page.
"""
