"""The green agent must refuse to score rather than guess at the weights.

A missing or malformed config once fell back to weights built into the code.
The scores that followed were plausible and carried no sign that the file the
run was supposed to be using had never been read.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ConfigIsRequiredTest(unittest.TestCase):
    def test_a_missing_config_raises(self):
        from src.green_agent.server import load_green_agent_config

        with self.assertRaises(FileNotFoundError):
            load_green_agent_config("config/no_such_config.yaml")

    def test_a_config_that_is_not_a_mapping_raises(self):
        from src.green_agent.server import load_green_agent_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("")
            with self.assertRaises(ValueError):
                load_green_agent_config(str(path))


if __name__ == "__main__":
    unittest.main()
