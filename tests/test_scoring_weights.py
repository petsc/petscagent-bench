"""Tests that every copy of the scoring weights agrees with the config.

The weights decide every score the benchmark reports. They are written once in
config/green_agent_config.yaml, then mirrored by hand into the dashboard loader
so the page can show the weighting without parsing YAML at build time. A mirror
that drifts is invisible: the numbers stay plausible and nothing errors.
"""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

CONFIG = ROOT / "config" / "green_agent_config.yaml"


def config_weights():
    return yaml.safe_load(CONFIG.read_text())["scoring"]["weights"]


class ScoringWeightsTest(unittest.TestCase):
    def test_the_configured_weights_sum_to_one(self):
        total = sum(config_weights().values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_the_aggregator_uses_the_configured_weights(self):
        from src.metrics.aggregation import MetricsAggregator

        aggregator = MetricsAggregator(yaml.safe_load(CONFIG.read_text()))
        self.assertEqual(aggregator.CATEGORY_WEIGHTS, config_weights())

    def test_the_dashboard_mirror_matches_the_configured_weights(self):
        from load_results import CATEGORY_WEIGHTS

        self.assertEqual(CATEGORY_WEIGHTS, config_weights())

    def test_every_weighted_category_has_at_least_one_evaluator(self):
        """A category nothing feeds scores zero and silently drags the composite."""
        from src.metrics.aggregation import MetricsAggregator

        mapped = set(MetricsAggregator.EVALUATOR_CATEGORY_MAP.values())
        self.assertEqual(set(config_weights()), mapped)


class ConfigIsRequiredTest(unittest.TestCase):
    """The harness must refuse to score rather than guess at the weights."""

    def test_a_missing_config_raises(self):
        from src.green_agent.server import load_green_agent_config

        with self.assertRaises(FileNotFoundError):
            load_green_agent_config("config/no_such_config.yaml")

    def test_a_config_that_is_not_a_mapping_raises(self):
        import tempfile

        from src.green_agent.server import load_green_agent_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("")
            with self.assertRaises(ValueError):
                load_green_agent_config(str(path))


if __name__ == "__main__":
    unittest.main()
