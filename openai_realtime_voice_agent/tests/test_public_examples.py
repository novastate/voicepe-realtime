import re
import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPOSITORY_ROOT / "examples" / "openclaw-bridge" / "bridge.mjs"


class TestPublicExamples(unittest.TestCase):
    def test_private_imessage_routes_are_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "examples/openclaw-bridge/.imessage-routes.json"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bridge_loads_imessage_routes_from_private_configuration(self):
        source = BRIDGE.read_text(encoding="utf-8")
        self.assertIn('readOptional(".imessage-routes.json")', source)
        self.assertIn("process.env.IMESSAGE_ROUTES", source)
        self.assertNotIn("new Map([", source)
        self.assertIsNone(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", source))
        self.assertIsNone(re.search(r"(?<!\w)\+?\d{10,15}(?!\w)", source))


if __name__ == "__main__":
    unittest.main()
