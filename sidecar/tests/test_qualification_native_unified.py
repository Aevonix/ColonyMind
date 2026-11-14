"""Controlled native integration, distinct from real-model performance."""
import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading

import pytest

# The live transition runs of this pack exercised the plugin task platform
# (colonymind_task, task handoffs) that M1 removed; the pack definitions and
# their evaluators stay until the M2/M3 cleanup replaces them.

from colonymind.qualification.native import configuration, native_context
from colonymind.qualification.native_unified import CONSUMERS, EVALUATORS
from colonymind.qualification.native_unified_cases import CASES, cases
from colonymind.qualification.records import read
from colonymind.qualification.runner import evaluate


def test_pack_coverage_matches_its_scenarios_and_comparison():
    from colonymind.qualification.pack_registry import PACKS

    native = cases()
    comparison = cases(arm='base_hermes')
    assert len(native) == 3
    assert len(comparison) == 1
    assert len({case.inputs['scenario'] for case in native + comparison}) == 3
    assert PACKS['unified'].coverage == (
        'Three native shared-work scenarios and one base-Hermes comparison; four executions, three scenarios.')


