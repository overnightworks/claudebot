import os
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_bridge import Provider, SessionRecord
from telegram_bot import BotConfig, BotRuntime, _find_audio_files, _resolve_binary


SESSION_ID = "019fe190-8663-7583-8697-a910fb1154e8"


class LoggingBoundaryTests(unittest.TestCase):
    def test_http_transport_cannot_log_token_bearing_urls_at_info(self) -> None:
        self.assertGreater(logging.getLogger("httpx").getEffectiveLevel(), logging.INFO)
        self.assertGreater(logging.getLogger("httpcore").getEffectiveLevel(), logging.INFO)


class RuntimeHandoffTests(unittest.TestCase):
    def test_first_start_binds_latest_matching_root_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            transcript = root / "rollout.jsonl"
            transcript.write_text("", encoding="utf-8")
            record = SessionRecord(
                provider=Provider.CODEX,
                session_id=SESSION_ID,
                working_directory=project,
                modified_at=1,
                transcript_path=transcript,
            )
            config = BotConfig(
                telegram_token="test-token",
                allowed_user_ids=frozenset({1}),
                project_directory=project,
                claude_binary="/bin/true",
                codex_binary="/bin/true",
                claude_model="opus",
                codex_model=None,
                timeout_seconds=10,
                state_path=root / "state.json",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "AGENT_PROVIDER": "codex",
                        "CODEX_SESSION_ID": "",
                        "CLAUDE_SESSION_ID": "",
                    },
                ),
                patch("telegram_bot.discover_sessions", return_value=[record]),
            ):
                runtime = BotRuntime(config)
            self.assertEqual(runtime.session_id(), SESSION_ID)


class BinaryResolutionTests(unittest.TestCase):
    def test_stale_configured_path_falls_back_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "codex"
            binary.write_text("", encoding="utf-8")
            binary.chmod(0o755)
            with (
                patch.dict(os.environ, {"TEST_BIN": "/missing/codex"}),
                patch("telegram_bot.shutil.which", return_value=str(binary)),
            ):
                self.assertEqual(_resolve_binary("TEST_BIN", "codex"), str(binary))

    def test_stable_symlink_is_not_resolved_to_versioned_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "codex-0.146.0"
            target.write_text("", encoding="utf-8")
            target.chmod(0o755)
            stable = root / "codex"
            stable.symlink_to(target)
            with (
                patch.dict(os.environ, {"TEST_BIN": str(stable)}),
                patch("telegram_bot.shutil.which", return_value=None),
            ):
                self.assertEqual(_resolve_binary("TEST_BIN", "codex"), str(stable))


class AudioBoundaryTests(unittest.TestCase):
    def test_only_project_audio_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            inside = project / "inside.mp3"
            inside.write_bytes(b"audio")
            outside = root / "outside.mp3"
            outside.write_bytes(b"audio")
            found = _find_audio_files(
                f"{inside} {outside}",
                project,
            )
            self.assertEqual(found, [inside])


if __name__ == "__main__":
    unittest.main()
