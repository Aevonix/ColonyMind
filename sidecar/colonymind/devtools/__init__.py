"""Development tools that ship in the package, so they run from an installed wheel.

- ``fake_model``: a scripted OpenAI-compatible model server (no keys, loopback only). The CI probes run it as
  ``python -m colonymind.devtools.fake_model PORT``; ``colonymind demo`` serves it on a thread.
- ``mind_loop_probe``: the mind loop's acceptance list against stock Hermes, driven by
  ``scripts/ci_mind_loop.sh`` as ``python -m colonymind.devtools.mind_loop_probe``.
"""
