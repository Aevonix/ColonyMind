"""Validation for the standalone watch configuration."""
import math
import re
from urllib.parse import urlsplit

DEPENDENCY_FAILURE_PASSES = 2
NOTICE_TIMEOUT_S = 30


def validate_watch(value):
    from colonymind.config import ConfigError, _parse_bool

    if not isinstance(value, dict):
        raise ConfigError('watch must be a mapping')
    value.setdefault('dependency_failure_passes', DEPENDENCY_FAILURE_PASSES)
    value.setdefault('notice_timeout_s', NOTICE_TIMEOUT_S)
    for name in ('enabled', 'tmux'):
        value[name] = _parse_bool(value[name], field_name='watch.' + name)
    for name in ('stuck_minutes', 'repeat_failures', 'repeat_window_hours', 'health_timeout_s',
                 'dependency_failure_passes', 'notice_timeout_s'):
        try:
            number = float(value[name])
            if isinstance(value[name], bool) or not math.isfinite(number) or number <= 0:
                raise ValueError
            if name in {'repeat_failures', 'dependency_failure_passes'} and not number.is_integer():
                raise ValueError
            value[name] = int(number) if name in {'repeat_failures', 'dependency_failure_passes'} else number
        except (TypeError, ValueError):
            raise ConfigError('watch.' + name + ' must be positive') from None
    if not isinstance(value['hermes_home'], str) or not value['hermes_home'].strip():
        raise ConfigError('watch.hermes_home must be a directory')
    command = value['notice_command']
    if command is None:
        value['notice_command'] = []
    elif not isinstance(command, list) or any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command):
        raise ConfigError('watch.notice_command must be an argv list')

    def url(text):
        if not isinstance(text, str):
            return False
        try:
            parsed = urlsplit(text)
            return parsed.scheme in {'http', 'https'} and bool(parsed.hostname) and not parsed.username and not parsed.password
        except (TypeError, ValueError):
            return False

    downloads = value['downloads']
    if not isinstance(downloads, dict) or not isinstance(downloads.get('url'), str):
        raise ConfigError('watch.downloads.url must be a URL or empty')
    if downloads['url'] and not url(downloads['url']):
        raise ConfigError('watch.downloads.url must be an HTTP URL without credentials')
    endpoints = value['endpoints']
    if not isinstance(endpoints, list):
        raise ConfigError('watch.endpoints must be a list')
    names = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict) or not isinstance(endpoint.get('name'), str) or not re.fullmatch(
                r'[A-Za-z0-9_. -]{1,64}', endpoint['name']):
            raise ConfigError('watch endpoint name must be a short plain label')
        if endpoint['name'] in names:
            raise ConfigError('watch endpoint names must be unique')
        names.add(endpoint['name'])
        if not url(endpoint.get('url')):
            raise ConfigError('watch endpoint URL must be HTTP without credentials')
        status = endpoint.get('expect_status', 200)
        if type(status) is not int or not 100 <= status <= 599:
            raise ConfigError('watch endpoint expect_status must be an HTTP status')
        endpoint['expect_status'] = status
    return value
