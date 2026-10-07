"""Tests on the scoring weights, which decide every score the benchmark reports."""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "green_agent_config.yaml"


def config_weights():
    return yaml.safe_load(CONFIG.read_text())["scoring"]["weights"]


class ScoringWeightsTest(unittest.TestCase):
    def test_the_configured_weights_sum_to_one(self):
        total = sum(config_weights().values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_every_weighted_category_has_at_least_one_evaluator(self):
        """A category nothing feeds scores zero and silently drags the composite."""
        from src.metrics.aggregation import MetricsAggregator

        mapped = set(MetricsAggregator.EVALUATOR_CATEGORY_MAP.values())
        self.assertEqual(set(config_weights()), mapped)


if __name__ == "__main__":
    unittest.main()
