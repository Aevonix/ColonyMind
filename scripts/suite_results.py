"""Record which tests failed from pytest's own test reports, for scripts/suite_env.sh.

The runner loads this plugin in both phases and passes --suite-results. At the end of the session that file
holds one JSON object whose "failed" member maps the complete node id of every test with a failed setup, call
or teardown (relative to the directory pytest runs in) to the JUnit classname and name pytest gives that test,
whose "uncollected" member lists the node ids of the modules and other collectors that could not be
collected, and whose "ended_early" member says whether a call of pytest.exit ended the session.
"""
import json

import pytest
from _pytest.junitxml import bin_xml_escape, mangle_test_address


def pytest_addoption(parser):
    parser.addoption('--suite-results', metavar='PATH',
                     help='write the ids of failed tests and of collection errors to PATH as JSON')


def pytest_configure(config):
    if config.getoption('suite_results'):
        config.pluginmanager.register(SuiteResults(config))


class SuiteResults:
    def __init__(self, config):
        self.config = config
        self.failed = {}
        self.uncollected = []
        self.ended_early = False

    def pytest_runtest_logreport(self, report):
        if report.failed:
            # The same names the JUnit report gives this test, so each failure there can be matched to an id.
            names = mangle_test_address(report.nodeid)
            classnames = names[:-1]
            if self.config.option.junitprefix:
                classnames.insert(0, self.config.option.junitprefix)
            junit = ['.'.join(classnames), bin_xml_escape(names[-1])]
            self.failed[self.config.cwd_relative_nodeid(report.nodeid)] = junit

    def pytest_collectreport(self, report):
        if report.failed:
            self.uncollected.append(self.config.cwd_relative_nodeid(report.nodeid))

    def pytest_keyboard_interrupt(self, excinfo):
        # Only pytest.exit ends a session with a status its caller chooses; every other interruption gives status 2.
        self.ended_early = isinstance(excinfo.value, pytest.exit.Exception)

    def pytest_sessionfinish(self):
        with open(self.config.getoption('suite_results'), 'w', encoding='utf-8') as stream:
            json.dump({'failed': self.failed, 'uncollected': self.uncollected, 'ended_early': self.ended_early},
                      stream)
