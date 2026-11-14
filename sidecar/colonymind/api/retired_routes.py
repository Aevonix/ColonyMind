"""Refusal-only compatibility for callers deployed before a route's retirement.

These paths have no router, request model, or mutation handler.
"""
SOURCE_FORGET = '/v1/host/memory/sources/forget'
RETIRED_ROUTE_CODES = {SOURCE_FORGET: 'memory_forget_moved'}
