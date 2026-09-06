import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import check_root_layout
from check_root_layout import (
    DEFAULT_HOME,
    DIRECTORY_HOME,
    HOME_BY_SUFFIX,
    REPOSITORY_ALLOWLIST,
    root_layout_problems,
)


GATE = Path(check_root_layout.__file__).resolve()


class RootLayoutGateProcessTest(unittest.TestCase):
    """The gate as CI runs it: a python process over the script, started somewhere."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repository = Path(directory.name)
        (self.repository / "scripts").mkdir()
        shutil.copy(GATE, self.repository / "scripts" / GATE.name)
        (self.repository / "README.md").write_text("a repository the gate is happy with\n")
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=self.repository,
            check=True,
            capture_output=True,
        )

    def _run_gate(self, working_directory: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.repository / "scripts" / GATE.name)],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_gate_passes_on_an_allowlisted_tree(self) -> None:
        completed = self._run_gate(self.repository)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("passed", completed.stdout)

    def test_gate_names_an_uncommitted_stray_wherever_it_is_started_from(self) -> None:
        (self.repository / "NOTES.md").write_text("a stray at the root\n")

        for working_directory in (".", "scripts"):
            with self.subTest(working_directory=working_directory):
                completed = self._run_gate(self.repository / working_directory)

                self.assertEqual(completed.returncode, 1)
                self.assertIn("NOTES.md", completed.stderr)

    def test_gate_accepts_a_scratch_file_gitignore_names(self) -> None:
        (self.repository / ".gitignore").write_text("/*.png\n")
        (self.repository / "Eingefügtes Bild.png").write_bytes(b"pasted scratch")

        completed = self._run_gate(self.repository)

        self.assertEqual(completed.returncode, 0, completed.stderr)


class RootLayoutProblemsTest(unittest.TestCase):
    def test_the_layout_this_repository_carries_passes(self) -> None:
        listing = [
            ".gitignore",
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "sonar-project.properties",
            "uv.lock",
            ".github/workflows/ci.yml",
            "deploy/agent-telegram.service",
            "scripts/bot_start.sh",
            "scripts/check_root_layout.py",
            "src/claudebot/telegram_bot.py",
            "tests/test_check_root_layout.py",
        ]

        self.assertEqual(root_layout_problems(listing, REPOSITORY_ALLOWLIST), ())

    def test_the_entry_documents_and_the_licence_are_legitimate_root_files(self) -> None:
        listing = ["README.md", "AGENTS.md", "CLAUDE.md", "HEART.md", "LICENSE"]

        self.assertEqual(root_layout_problems(listing, REPOSITORY_ALLOWLIST), ())

    def test_stray_root_file_carries_the_sentence_where_it_belongs(self) -> None:
        strays = [
            ("telegram_bot.py", HOME_BY_SUFFIX[".py"]),
            ("bot_start.sh", HOME_BY_SUFFIX[".sh"]),
            ("agent-telegram.service", HOME_BY_SUFFIX[".service"]),
            ("NOTES.md", HOME_BY_SUFFIX[".md"]),
            ("requirements.txt", DEFAULT_HOME),
        ]

        for stray, home in strays:
            with self.subTest(stray=stray):
                problems = root_layout_problems(["README.md", stray], REPOSITORY_ALLOWLIST)

                self.assertEqual(problems, (f"{stray}: {home}",))

    def test_stray_root_directory_carries_the_sentence_where_it_belongs(self) -> None:
        problems = root_layout_problems(["README.md", "tooling/helper.py"], REPOSITORY_ALLOWLIST)

        self.assertEqual(problems, (f"tooling/: {DIRECTORY_HOME}",))


if __name__ == "__main__":
    unittest.main()
