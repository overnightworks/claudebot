import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import check_root_layout
from check_root_layout import (
    DEFAULT_HOME,
    DIRECTORY_HOME,
    HOME_BY_SUFFIX,
    REPOSITORY_ALLOWLIST,
    main,
    repository_listing,
    root_layout_problems,
)


GATE = Path(check_root_layout.__file__).resolve()


class TemporaryRepositoryTestCase(unittest.TestCase):
    """A throwaway git repository carrying a copy of the gate, allowlisted as it stands."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repository = Path(directory.name)
        (self.repository / "scripts").mkdir()
        shutil.copy(GATE, self.repository / "scripts" / GATE.name)
        (self.repository / "README.md").write_text("a repository the gate is happy with\n")
        self._git("init", "--quiet")

    def _git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            capture_output=True,
        )


class RootLayoutGateProcessTest(TemporaryRepositoryTestCase):
    """The gate as CI runs it: a python process over the script, started somewhere."""

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

class RootLayoutGateInProcessTest(TemporaryRepositoryTestCase):
    """The same gate called in this process, so its own coverage is measured."""

    def _judge_the_temporary_repository(self) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(check_root_layout, "REPOSITORY_ROOT", self.repository):
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = main()
        return exit_code, out.getvalue(), err.getvalue()

    def test_the_gate_passes_on_an_allowlisted_tree(self) -> None:
        exit_code, stdout, stderr = self._judge_the_temporary_repository()

        self.assertEqual(exit_code, 0, stderr)
        self.assertIn("passed", stdout)

    def test_the_gate_fails_naming_every_stray_and_where_it_belongs(self) -> None:
        (self.repository / "NOTES.md").write_text("a stray at the root\n")
        (self.repository / "tooling").mkdir()
        (self.repository / "tooling" / "helper.py").write_text("a stray directory\n")

        exit_code, _, stderr = self._judge_the_temporary_repository()

        self.assertEqual(exit_code, 1)
        self.assertIn(f"NOTES.md: {HOME_BY_SUFFIX['.md']}", stderr)
        self.assertIn(f"tooling/: {DIRECTORY_HOME}", stderr)

    def test_the_listing_holds_tracked_and_unignored_untracked_paths(self) -> None:
        (self.repository / ".gitignore").write_text("/*.png\n")
        (self.repository / "Eingefügtes Bild.png").write_bytes(b"pasted scratch")
        (self.repository / "NOTES.md").write_text("a stray at the root\n")
        self._git("add", "README.md")

        self.assertEqual(
            set(repository_listing(self.repository)),
            {"README.md", ".gitignore", "NOTES.md", f"scripts/{GATE.name}"},
        )


class PastedRootImageTest(unittest.TestCase):
    """This repository's own answer to a pasted scratch image, which .gitignore owns."""

    def _repository_excludes(self, name: str) -> bool:
        """Whether this repository's exclude rules hide `name`, as the gate's listing does.

        `--no-index` asks the exclude machinery behind `git ls-files --others
        --exclude-standard` about a path that need not exist, so the answer
        costs the working tree nothing.
        """
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", name],
            cwd=check_root_layout.REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode > 1:
            raise RuntimeError(f"git check-ignore failed: {completed.stderr}")
        return completed.returncode == 0

    def test_this_repository_treats_a_pasted_root_image_as_scratch(self) -> None:
        for image in ("pasted.png", "pasted.jpg", "pasted.jpeg"):
            with self.subTest(image=image):
                self.assertTrue(self._repository_excludes(image))


class RootLayoutProblemsTest(unittest.TestCase):
    def test_the_layout_this_repository_carries_passes(self) -> None:
        listing = [
            ".gitignore",
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "sonar-project.properties",
            "uv.lock",
            ".aco/board.toml",
            "quality/config.toml",
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
