"""Engine admission uses all callers' metrics, with bounded stale readings."""

import asyncio
from math import isfinite
import re

import httpx

from colonymind.util.temporal import now_utc


METRICS = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
           "vllm:kv_cache_usage_perc": "kv"}


def parse_metrics(text):
    values = {key: [] for key in METRICS.values()}
    for line in text.splitlines():
        match = re.match(r"^([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$", line)
        if not match or match[1] not in METRICS:
            continue
        try:
            value = float(match[2])
        except ValueError:
            return None
        if not isfinite(value) or value < 0:
            return None
        values[METRICS[match[1]]].append(value)
    if not all(values.values()) or max(values["kv"]) > 1:
        return None
    # Assumption: gauges on distinct engine labels are summed; KV uses the
    # busiest engine, so a healthy shard cannot hide a saturated one.
    return {"running": sum(values["running"]), "waiting": sum(values["waiting"]), "kv": max(values["kv"])}


class EngineHealth:
    def __init__(self, config, clock=None):
        self.config = config
        self.clock = clock or now_utc
        self.readings = {}
        self.last_poll = None
        self._lock = asyncio.Lock()

    def record(self, family, *, running, waiting, kv):
        if not all(isfinite(x) and x >= 0 for x in (running, waiting, kv)) or kv > 1:
            self.invalidate(family)
            return
        self.readings[family] = ({"running": running, "waiting": waiting, "kv": kv}, self.clock())

    def invalidate(self, family):
        self.readings.pop(family, None)

    async def refresh(self):
        """Read configured endpoints at most once per poll interval.

        Endpoints are supplied by the host under health.metrics_urls. No socket
        is opened when none are configured. The request timeout is one poll
        interval, an assumption, and errors invalidate only the affected family.
        """
        endpoints = self.config.get("metrics_urls", {})
        if not endpoints:
            return
        async with self._lock:
            if self.last_poll and (self.clock() - self.last_poll).total_seconds() < self.config["poll_seconds"]:
                return
            self.last_poll = self.clock()
            async with httpx.AsyncClient(timeout=self.config["poll_seconds"]) as client:
                async def read(family, url):
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        parsed = parse_metrics(response.text)
                        if parsed is None:
                            raise ValueError("incomplete metrics")
                        self.record(family, **parsed)
                    except (httpx.HTTPError, ValueError):
                        self.invalidate(family)
                await asyncio.gather(*(read(family, url) for family, url in endpoints.items()))

    def admit(self, family, priority, occupied):
        reading = self.readings.get(family)
        fresh = reading and (self.clock() - reading[1]).total_seconds() <= self.config["poll_seconds"]
        if not fresh:
            return None if priority == "owner" and occupied < self.config["unreadable_owner_cap"] else "metrics_unreadable"
        metrics = reading[0]
        limit = self.config["running_limits"].get(family)
        if (metrics["waiting"] > self.config["max_waiting"]
                or metrics["kv"] >= self.config["max_kv_fraction"]
                or limit is None or metrics["running"] > limit):
            return "engine_busy"
        return None
