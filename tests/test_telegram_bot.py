import asyncio
import os
import logging
import tempfile
import timeit
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_bridge import AgentReply, BridgeError, ControlIntent, Provider, SessionRecord
from telegram_bot import (
    BotConfig,
    BotRuntime,
    MAX_REPLY_CHUNK,
    _authorized,
    _cancel_active_task,
    _find_audio_files,
    _handle_control_intent,
    _keep_typing,
    _positive_integer,
    _resolve_binary,
    _run_agent_task,
    _run_systemctl,
    _send_reply,
    _start_claude_remote_control,
    _switch_reply,
    _takeover_reply,
    cmd_claude_remote,
    cmd_claude_remote_stop,
    cmd_new,
    cmd_player,
    cmd_provider,
    cmd_session,
    cmd_sessions,
    cmd_start,
    cmd_status,
    cmd_stop,
    cmd_takeover,
    handle_message,
    load_config,
    _runtime,
)


SESSION_ID = "019fe190-8663-7583-8697-a910fb1154e8"
OTHER_SESSION_ID = "019fe643-8a24-7f90-ac01-8fc5242d91ed"
ALLOWED_USER_ID = 1
OTHER_USER_ID = 2


class LoggingBoundaryTests(unittest.TestCase):
    def test_http_transport_cannot_log_token_bearing_urls_at_info(self) -> None:
        self.assertGreater(logging.getLogger("httpx").getEffectiveLevel(), logging.INFO)
        self.assertGreater(logging.getLogger("httpcore").getEffectiveLevel(), logging.INFO)


def _record(
    provider: Provider,
    project: Path,
    root: Path,
    session_id: str = SESSION_ID,
    modified_at: float = 1,
) -> SessionRecord:
    transcript = root / f"{session_id}.jsonl"
    if not transcript.exists():
        transcript.write_text("", encoding="utf-8")
    return SessionRecord(
        provider=provider,
        session_id=session_id,
        working_directory=project,
        modified_at=modified_at,
        transcript_path=transcript,
    )


def _config(project: Path, root: Path, **overrides) -> BotConfig:
    defaults = dict(
        telegram_token="test-token",
        allowed_user_ids=frozenset({ALLOWED_USER_ID}),
        project_directory=project,
        claude_binary="/bin/true",
        codex_binary="/bin/true",
        claude_model="opus",
        codex_model=None,
        timeout_seconds=10,
        state_path=root / "state.json",
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


class RuntimeHandoffTests(unittest.TestCase):
    def test_first_start_binds_latest_matching_root_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            record = _record(Provider.CODEX, project, root)
            config = _config(project, root)
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

    def test_no_configured_ids_and_no_local_session_leaves_state_unwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            config = _config(project, root)
            with (
                patch.dict(
                    os.environ,
                    {"AGENT_PROVIDER": "", "CODEX_SESSION_ID": "", "CLAUDE_SESSION_ID": ""},
                ),
                patch("telegram_bot.discover_sessions", return_value=[]),
            ):
                runtime = BotRuntime(config)
            self.assertIsNone(runtime.session_id())
            self.assertFalse(config.state_path.exists())

    def test_configured_provider_without_session_id_falls_back_to_local_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            record = _record(Provider.CLAUDE, project, root, session_id=OTHER_SESSION_ID)
            config = _config(project, root)
            with (
                patch.dict(
                    os.environ,
                    {
                        "AGENT_PROVIDER": "claude",
                        "CODEX_SESSION_ID": "",
                        "CLAUDE_SESSION_ID": "",
                    },
                ),
                patch(
                    "telegram_bot.discover_sessions",
                    side_effect=lambda provider, _project: (
                        [record] if provider is Provider.CLAUDE else []
                    ),
                ),
            ):
                runtime = BotRuntime(config)
            self.assertEqual(runtime.provider, Provider.CLAUDE)
            self.assertEqual(runtime.session_id(), OTHER_SESSION_ID)
            self.assertTrue(config.state_path.exists())

    def test_preexisting_state_is_not_overwritten_by_configured_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            config = _config(project, root)
            with (
                patch.dict(os.environ, {"CODEX_SESSION_ID": "", "CLAUDE_SESSION_ID": ""}),
                patch("telegram_bot.discover_sessions", return_value=[]),
            ):
                BotRuntime(config).store.save(
                    {"provider": "codex", "sessions": {"codex": SESSION_ID}}
                )
            with (
                patch.dict(
                    os.environ,
                    {"CODEX_SESSION_ID": OTHER_SESSION_ID, "CLAUDE_SESSION_ID": ""},
                ),
                patch("telegram_bot.discover_sessions", return_value=[]),
            ):
                runtime = BotRuntime(config)
            self.assertEqual(runtime.session_id(Provider.CODEX), SESSION_ID)

    def test_configured_session_id_seeds_an_empty_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            config = _config(project, root)
            with (
                patch.dict(
                    os.environ,
                    {"CODEX_SESSION_ID": SESSION_ID, "CLAUDE_SESSION_ID": ""},
                ),
                patch("telegram_bot.discover_sessions", return_value=[]),
            ):
                runtime = BotRuntime(config)
            self.assertEqual(runtime.session_id(Provider.CODEX), SESSION_ID)
            self.assertTrue(config.state_path.exists())


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

    def test_no_candidate_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BIN", None)
            with patch("telegram_bot.shutil.which", return_value=None):
                with self.assertRaises(BridgeError):
                    _resolve_binary("TEST_BIN", "totally-nonexistent-claudebot-binary-xyz")

    def test_duplicate_candidate_path_is_only_checked_once(self) -> None:
        missing = "/definitely/missing/claudebot-test-binary"
        with (
            patch.dict(os.environ, {"TEST_BIN": missing}),
            patch("telegram_bot.shutil.which", return_value=missing),
        ):
            with self.assertRaises(BridgeError):
                _resolve_binary("TEST_BIN", "totally-nonexistent-claudebot-binary-xyz")


class PositiveIntegerTests(unittest.TestCase):
    def test_missing_environment_variable_uses_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISSING_TIMEOUT", None)
            self.assertEqual(_positive_integer("MISSING_TIMEOUT", 42), 42)

    def test_non_integer_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"BAD_TIMEOUT": "soon"}):
            with self.assertRaises(BridgeError):
                _positive_integer("BAD_TIMEOUT", 1)

    def test_non_positive_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"BAD_TIMEOUT": "0"}):
            with self.assertRaises(BridgeError):
                _positive_integer("BAD_TIMEOUT", 1)


class LoadConfigTests(unittest.TestCase):
    def _base_env(self, root: Path) -> dict[str, str]:
        claude_binary = root / "claude"
        codex_binary = root / "codex"
        for binary in (claude_binary, codex_binary):
            binary.write_text("", encoding="utf-8")
            binary.chmod(0o755)
        return {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_ALLOWED_USER_IDS": "1,2",
            "AGENT_PROJECT_DIR": str(root),
            "CLAUDE_BIN": str(claude_binary),
            "CODEX_BIN": str(codex_binary),
        }

    def test_missing_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            del env["TELEGRAM_BOT_TOKEN"]
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(BridgeError):
                    load_config()

    def test_missing_allowed_ids_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            del env["TELEGRAM_ALLOWED_USER_IDS"]
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(BridgeError):
                    load_config()

    def test_blank_allowed_ids_after_parsing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            env["TELEGRAM_ALLOWED_USER_IDS"] = " , , "
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(BridgeError):
                    load_config()

    def test_non_numeric_allowed_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            env["TELEGRAM_ALLOWED_USER_IDS"] = "abc"
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(BridgeError):
                    load_config()

    def test_missing_project_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            env["AGENT_PROJECT_DIR"] = str(Path(directory) / "does-not-exist")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(BridgeError):
                    load_config()

    def test_project_directory_source_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            songmaker_dir = root / "songmaker"
            songmaker_dir.mkdir()
            env = self._base_env(root)
            del env["AGENT_PROJECT_DIR"]
            env["SONGMAKER_DIR"] = str(songmaker_dir)
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
            self.assertEqual(config.project_directory, songmaker_dir.resolve())

    def test_neither_project_variable_falls_back_to_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            del env["AGENT_PROJECT_DIR"]
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
            self.assertEqual(config.project_directory, Path.cwd().resolve())

    def test_successful_config_carries_codex_model_and_allowed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            env["CODEX_MODEL"] = "o1"
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
            self.assertEqual(config.allowed_user_ids, frozenset({1, 2}))
            self.assertEqual(config.codex_model, "o1")
            self.assertEqual(config.claude_model, "opus")

    def test_blank_codex_model_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._base_env(Path(directory))
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
            self.assertIsNone(config.codex_model)


# Quadrupling path-shaped text costs a linear scan four times as much and a
# backtracking one sixteen; the headroom sits between the two so the assertion
# reads the growth curve rather than this machine's current load. The floor
# keeps scheduler noise from turning a sub-millisecond pair into a failure.
_PATHOLOGICAL_SEGMENTS = 500
_GROWTH_FACTOR = 4
_LINEAR_GROWTH_HEADROOM = 8.0
_MEASUREMENT_FLOOR_SECONDS = 0.05
# The bound the pattern once applied to a whole path, truncating anything
# longer to its tail; the fixtures below stay above it on purpose.
_LENGTH_THAT_USED_TO_TRUNCATE = 255


def _seconds_to_scan(segments: int) -> float:
    payload = ("/segment.name" * segments) + "!"
    return min(
        timeit.repeat(
            lambda: _find_audio_files(payload, Path("/tmp")), repeat=3, number=1
        )
    )


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

    def test_relative_path_is_resolved_against_the_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            (project / "sub").mkdir(parents=True)
            audio = project / "sub" / "clip.wav"
            audio.write_bytes(b"audio")
            found = _find_audio_files("Hör dir sub/clip.wav an", project)
            self.assertEqual(found, [audio])

    def test_missing_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            found = _find_audio_files("ghost/nope.mp3", project)
            self.assertEqual(found, [])

    def test_oversized_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            audio = project / "big.mp3"
            audio.write_bytes(b"x" * 10)
            with patch("telegram_bot.MAX_AUDIO_BYTES", 1):
                found = _find_audio_files(str(audio), project)
            self.assertEqual(found, [])

    def test_a_deep_path_is_matched_in_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            deep = project.joinpath(*["d" * 8 for _ in range(30)])
            deep.mkdir(parents=True)
            audio = deep / "clip.mp3"
            audio.write_bytes(b"audio")
            self.assertGreater(len(str(audio)), _LENGTH_THAT_USED_TO_TRUNCATE)
            found = _find_audio_files(f"fertig: {audio}", project)
            self.assertEqual(found, [audio])

    def test_a_long_filename_is_matched_in_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir(parents=True)
            audio = project / ("n" * 251 + ".mp3")
            audio.write_bytes(b"audio")
            self.assertGreater(len(str(audio)), _LENGTH_THAT_USED_TO_TRUNCATE)
            found = _find_audio_files(f"fertig: {audio}", project)
            self.assertEqual(found, [audio])

    def test_scanning_pathological_input_grows_linearly_with_its_length(self) -> None:
        short = _seconds_to_scan(_PATHOLOGICAL_SEGMENTS)
        long = _seconds_to_scan(_PATHOLOGICAL_SEGMENTS * _GROWTH_FACTOR)
        self.assertLessEqual(
            long,
            short * _LINEAR_GROWTH_HEADROOM + _MEASUREMENT_FLOOR_SECONDS,
        )


class BotRuntimeMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.config = _config(self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[]):
            self.runtime = BotRuntime(self.config)

    def test_switch_provider_binds_latest_local_session(self) -> None:
        record = _record(Provider.CLAUDE, self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            bound = self.runtime.switch_provider(Provider.CLAUDE)
        self.assertEqual(bound, record)
        self.assertEqual(self.runtime.session_id(Provider.CLAUDE), SESSION_ID)

    def test_switch_provider_without_local_session_still_persists_choice(self) -> None:
        with patch("telegram_bot.discover_sessions", return_value=[]):
            bound = self.runtime.switch_provider(Provider.CLAUDE)
        self.assertIsNone(bound)
        self.assertEqual(self.runtime.provider, Provider.CLAUDE)
        self.assertTrue(self.config.state_path.exists())

    def test_takeover_binds_latest_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            bound = self.runtime.takeover()
        self.assertEqual(bound, record)
        self.assertEqual(self.runtime.session_id(), SESSION_ID)

    def test_takeover_without_local_session_raises(self) -> None:
        with patch("telegram_bot.discover_sessions", return_value=[]):
            with self.assertRaises(BridgeError):
                self.runtime.takeover()

    def test_clear_session_removes_bound_id(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            self.runtime.takeover()
        self.runtime.clear_session()
        self.assertIsNone(self.runtime.session_id())

    def test_bind_session_id_requires_a_known_session(self) -> None:
        with patch("telegram_bot.discover_sessions", return_value=[]):
            with self.assertRaises(BridgeError):
                self.runtime.bind_session_id(SESSION_ID)

    def test_bind_session_id_binds_a_known_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            bound = self.runtime.bind_session_id(SESSION_ID)
        self.assertEqual(bound, record)

    def test_bound_record_is_none_without_a_bound_session(self) -> None:
        self.assertIsNone(self.runtime.bound_record())

    def test_bound_record_finds_the_matching_local_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            self.runtime.takeover()
            self.assertEqual(self.runtime.bound_record(), record)

    def test_session_is_locally_active_delegates_to_agent_bridge(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with patch("telegram_bot.session_is_active", return_value=True) as mocked:
            self.assertTrue(self.runtime.session_is_locally_active(record))
        mocked.assert_called_once()

    def test_ask_refuses_a_locally_open_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        with (
            patch("telegram_bot.discover_sessions", return_value=[record]),
            patch("telegram_bot.session_is_active", return_value=True),
        ):
            self.runtime.takeover()
            with self.assertRaises(BridgeError):
                asyncio.run(self.runtime.ask("hi"))

    def test_ask_updates_the_bound_session_id(self) -> None:
        reply = AgentReply(text="hallo", session_id=OTHER_SESSION_ID, delivered_during_run=False)
        with (
            patch("telegram_bot.discover_sessions", return_value=[]),
            patch.object(self.runtime.bridge, "ask", AsyncMock(return_value=reply)),
        ):
            result = asyncio.run(self.runtime.ask("hi"))
        self.assertEqual(result, reply)
        self.assertEqual(self.runtime.session_id(), OTHER_SESSION_ID)


class FakeUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class FakeChat:
    def __init__(self) -> None:
        self.actions: list[str] = []

    async def send_action(self, action: str) -> None:
        self.actions.append(action)


class FakeMessage:
    def __init__(self, text: str | None = None, user_id: int = ALLOWED_USER_ID) -> None:
        self.text = text
        self.from_user = FakeUser(user_id)
        self.chat = FakeChat()
        self.replies: list[str] = []
        self.audio: list[dict[str, object]] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)

    async def reply_audio(self, *, audio: object, title: str, filename: str) -> None:
        self.audio.append({"audio": audio, "title": title, "filename": filename})


_NO_OVERRIDE = object()


class FakeUpdate:
    def __init__(self, message: FakeMessage, effective_user: object = _NO_OVERRIDE) -> None:
        self.effective_message = message
        self.effective_user = (
            message.from_user if effective_user is _NO_OVERRIDE else effective_user
        )


class FakeContext:
    def __init__(self, args: list[str] | None = None) -> None:
        self.args = args or []


class HandlerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.config = _config(self.project, self.root)
        with patch("telegram_bot.discover_sessions", return_value=[]):
            self.runtime = BotRuntime(self.config)
        patcher = patch("telegram_bot.runtime", self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)


class RuntimeAccessorTests(unittest.TestCase):
    def test_uninitialized_runtime_raises(self) -> None:
        with patch("telegram_bot.runtime", None):
            with self.assertRaises(RuntimeError):
                _runtime()


class AuthorizationTests(HandlerTestCase):
    async def test_missing_user_is_not_authorized(self) -> None:
        update = FakeUpdate(FakeMessage(), effective_user=None)
        self.assertFalse(_authorized(update))

    async def test_unlisted_user_is_not_authorized(self) -> None:
        update = FakeUpdate(FakeMessage(user_id=OTHER_USER_ID))
        self.assertFalse(_authorized(update))

    async def test_listed_user_is_authorized(self) -> None:
        update = FakeUpdate(FakeMessage(user_id=ALLOWED_USER_ID))
        self.assertTrue(_authorized(update))


class CmdStartTests(HandlerTestCase):
    async def test_unauthorized_user_is_refused(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_start(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Nicht autorisiert."])

    async def test_authorized_user_gets_the_welcome_text(self) -> None:
        message = FakeMessage()
        await cmd_start(FakeUpdate(message), FakeContext())
        self.assertEqual(len(message.replies), 1)
        self.assertIn("Bereit", message.replies[0])


class CmdProviderTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_provider(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_without_arguments_reports_the_active_provider(self) -> None:
        message = FakeMessage()
        await cmd_provider(FakeUpdate(message), FakeContext())
        self.assertIn("codex", message.replies[0])

    async def test_switching_to_a_provider_without_a_local_session(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await cmd_provider(FakeUpdate(message), FakeContext(["claude"]))
        self.assertIn("Claude", message.replies[0])

    async def test_switching_to_a_provider_binds_the_latest_session(self) -> None:
        record = _record(Provider.CLAUDE, self.project, self.root)
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            await cmd_provider(FakeUpdate(message), FakeContext(["claude"]))
        self.assertIn(record.compact_label(), message.replies[0])

    async def test_unknown_provider_name_reports_an_error(self) -> None:
        message = FakeMessage()
        await cmd_provider(FakeUpdate(message), FakeContext(["gemini"]))
        self.assertTrue(message.replies[0].startswith("⚠️"))


class CmdTakeoverTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_takeover(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_takeover_with_a_local_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        with (
            patch("telegram_bot.discover_sessions", return_value=[record]),
            patch("telegram_bot.session_is_active", return_value=False),
        ):
            await cmd_takeover(FakeUpdate(message), FakeContext())
        self.assertIn(record.compact_label(), message.replies[0])

    async def test_takeover_without_a_local_session_reports_the_error(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await cmd_takeover(FakeUpdate(message), FakeContext())
        self.assertTrue(message.replies[0].startswith("⚠️"))


class CmdNewTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_new(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_without_a_running_task(self) -> None:
        message = FakeMessage()
        await cmd_new(FakeUpdate(message), FakeContext())
        self.assertFalse(message.replies[0].startswith("Laufender"))

    async def test_stops_a_running_task_first(self) -> None:
        message = FakeMessage()
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[message.from_user.id] = task
        await cmd_new(FakeUpdate(message), FakeContext())
        self.assertTrue(message.replies[0].startswith("Laufender Auftrag gestoppt."))


class CmdSessionsTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_sessions(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_no_local_sessions(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await cmd_sessions(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Keine passende lokale Session gefunden."])

    async def test_lists_local_sessions(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            await cmd_sessions(FakeUpdate(message), FakeContext())
        self.assertIn(record.compact_label(), message.replies[0])
        self.assertEqual(
            self.runtime.session_choices[message.from_user.id], [record]
        )


class CmdSessionTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_session(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_without_arguments_prompts_for_usage(self) -> None:
        message = FakeMessage()
        await cmd_session(FakeUpdate(message), FakeContext())
        self.assertIn("/sessions", message.replies[0])

    async def test_selecting_by_number_from_the_last_listing(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        self.runtime.session_choices[message.from_user.id] = [record]
        await cmd_session(FakeUpdate(message), FakeContext(["1"]))
        self.assertIn(record.compact_label(), message.replies[0])

    async def test_selecting_an_out_of_range_number_reports_the_error(self) -> None:
        message = FakeMessage()
        self.runtime.session_choices[message.from_user.id] = []
        await cmd_session(FakeUpdate(message), FakeContext(["1"]))
        self.assertTrue(message.replies[0].startswith("⚠️"))

    async def test_selecting_by_uuid(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[record]):
            await cmd_session(FakeUpdate(message), FakeContext([SESSION_ID]))
        self.assertIn(record.compact_label(), message.replies[0])

    async def test_selecting_an_unknown_uuid_reports_the_error(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await cmd_session(FakeUpdate(message), FakeContext([SESSION_ID]))
        self.assertTrue(message.replies[0].startswith("⚠️"))


class CmdStatusTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_status(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_reports_ready_without_a_session(self) -> None:
        message = FakeMessage()
        await cmd_status(FakeUpdate(message), FakeContext())
        self.assertIn("Session: neu", message.replies[0])
        self.assertIn("bereit", message.replies[0])

    async def test_reports_working_while_a_task_is_active(self) -> None:
        message = FakeMessage()
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[message.from_user.id] = task
        await cmd_status(FakeUpdate(message), FakeContext())
        self.assertIn("arbeitet per Telegram", message.replies[0])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_reports_locally_open_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        with (
            patch("telegram_bot.discover_sessions", return_value=[record]),
            patch("telegram_bot.session_is_active", return_value=True),
        ):
            self.runtime.takeover()
            await cmd_status(FakeUpdate(message), FakeContext())
        self.assertIn("lokal geöffnet", message.replies[0])
        self.assertIn(SESSION_ID[:8], message.replies[0])


class CmdStopTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_stop(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_without_a_running_task(self) -> None:
        message = FakeMessage()
        await cmd_stop(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Kein laufender Auftrag."])

    async def test_stops_a_running_task(self) -> None:
        message = FakeMessage()
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[message.from_user.id] = task
        await cmd_stop(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Laufender Auftrag ist gestoppt."])
        self.assertNotIn(message.from_user.id, self.runtime.active_tasks)


class CmdClaudeRemoteTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_claude_remote(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_success_reports_the_remote_control_url(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot._start_claude_remote_control", AsyncMock()):
            await cmd_claude_remote(FakeUpdate(message), FakeContext())
        self.assertIn("claude.ai/code", message.replies[0])

    async def test_failure_reports_the_bridge_error(self) -> None:
        message = FakeMessage()
        failure = AsyncMock(side_effect=BridgeError("nicht aktiv"))
        with patch("telegram_bot._start_claude_remote_control", failure):
            await cmd_claude_remote(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["⚠️ nicht aktiv"])

    async def test_unauthorized_stop_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_claude_remote_stop(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_stop_reports_success(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot._run_systemctl", AsyncMock()) as mocked:
            await cmd_claude_remote_stop(FakeUpdate(message), FakeContext())
        mocked.assert_awaited_once_with("stop", "claude-remote-control.service")
        self.assertEqual(message.replies, ["Claude Remote Control ist gestoppt."])


class CmdPlayerTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(user_id=OTHER_USER_ID)
        await cmd_player(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_missing_environment_file(self) -> None:
        message = FakeMessage()
        await cmd_player(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Kein Player für dieses Projekt gefunden."])

    async def test_reports_the_configured_url(self) -> None:
        (self.project / ".server.env").write_text(
            "\n".join(
                [
                    "# a comment line",
                    "no-equals-sign-here",
                    "SONGMAKER_URL=https://example.test/player",
                ]
            ),
            encoding="utf-8",
        )
        message = FakeMessage()
        await cmd_player(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["https://example.test/player"])

    async def test_reports_missing_url_when_not_configured(self) -> None:
        (self.project / ".server.env").write_text("OTHER_KEY=1", encoding="utf-8")
        message = FakeMessage()
        await cmd_player(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Keine SONGMAKER_URL gefunden."])


class HandleMessageTests(HandlerTestCase):
    async def test_unauthorized_user_gets_no_reply(self) -> None:
        message = FakeMessage(text="hallo", user_id=OTHER_USER_ID)
        await handle_message(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_empty_text_is_ignored(self) -> None:
        message = FakeMessage(text=None)
        await handle_message(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, [])

    async def test_overlong_text_is_refused(self) -> None:
        message = FakeMessage(text="x" * 16_001)
        await handle_message(FakeUpdate(message), FakeContext())
        self.assertEqual(message.replies, ["Die Nachricht ist für den Bot zu lang."])

    async def test_control_intent_is_delegated(self) -> None:
        message = FakeMessage(text="Wechsel zu Claude")
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await handle_message(FakeUpdate(message), FakeContext())
        self.assertIn("Claude", message.replies[0])

    async def test_busy_user_is_told_to_wait(self) -> None:
        message = FakeMessage(text="Wie geht es dir?")
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[message.from_user.id] = task
        await handle_message(FakeUpdate(message), FakeContext())
        self.assertIn("Ich arbeite noch", message.replies[0])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_locally_open_session_refuses_the_message(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage(text="Wie geht es dir?")
        with (
            patch("telegram_bot.discover_sessions", return_value=[record]),
            patch("telegram_bot.session_is_active", return_value=True),
        ):
            self.runtime.takeover()
            await handle_message(FakeUpdate(message), FakeContext())
        self.assertIn("lokalen Terminal", message.replies[0])

    async def test_ordinary_message_starts_a_background_agent_task(self) -> None:
        message = FakeMessage(text="Wie geht es dir?")
        reply = AgentReply(text="Gut, danke!", session_id=SESSION_ID, delivered_during_run=False)
        with (
            patch("telegram_bot.discover_sessions", return_value=[]),
            patch.object(self.runtime, "ask", AsyncMock(return_value=reply)),
        ):
            await handle_message(FakeUpdate(message), FakeContext())
            task = self.runtime.active_tasks[message.from_user.id]
            await task
        self.assertIn("Empfangen", message.replies[0])
        self.assertIn("Gut, danke!", message.replies[-1])
        self.assertNotIn(message.from_user.id, self.runtime.active_tasks)


class ControlIntentHandlingTests(HandlerTestCase):
    async def test_switch_action_binds_the_provider(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await _handle_control_intent(message, ControlIntent("switch", Provider.CLAUDE))
        self.assertIn("Claude", message.replies[0])

    async def test_takeover_action_binds_the_latest_session(self) -> None:
        record = _record(Provider.CODEX, self.project, self.root)
        message = FakeMessage()
        with (
            patch("telegram_bot.discover_sessions", return_value=[record]),
            patch("telegram_bot.session_is_active", return_value=False),
        ):
            await _handle_control_intent(message, ControlIntent("takeover"))
        self.assertIn(record.compact_label(), message.replies[0])

    async def test_new_action_reports_a_running_task_was_stopped(self) -> None:
        message = FakeMessage()
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[message.from_user.id] = task
        await _handle_control_intent(message, ControlIntent("new"))
        self.assertTrue(message.replies[0].startswith("Laufender Auftrag gestoppt."))

    async def test_claude_remote_action_reports_success(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot._start_claude_remote_control", AsyncMock()):
            await _handle_control_intent(message, ControlIntent("claude_remote"))
        self.assertIn("claude.ai/code", message.replies[0])

    async def test_bridge_error_is_reported(self) -> None:
        message = FakeMessage()
        with patch("telegram_bot.discover_sessions", return_value=[]):
            await _handle_control_intent(message, ControlIntent("takeover"))
        self.assertTrue(message.replies[0].startswith("⚠️"))

    async def test_unrecognized_action_is_a_no_op(self) -> None:
        message = FakeMessage()
        await _handle_control_intent(message, ControlIntent("unknown"))
        self.assertEqual(message.replies, [])


class CancelActiveTaskTests(HandlerTestCase):
    async def test_no_task_returns_false(self) -> None:
        self.assertFalse(await _cancel_active_task(ALLOWED_USER_ID))

    async def test_already_done_task_returns_false(self) -> None:
        async def noop() -> None:
            return None

        task = asyncio.create_task(noop())
        await task
        self.runtime.active_tasks[ALLOWED_USER_ID] = task
        self.assertFalse(await _cancel_active_task(ALLOWED_USER_ID))
        self.assertNotIn(ALLOWED_USER_ID, self.runtime.active_tasks)

    async def test_running_task_is_cancelled(self) -> None:
        task = asyncio.create_task(asyncio.sleep(30))
        self.runtime.active_tasks[ALLOWED_USER_ID] = task
        self.assertTrue(await _cancel_active_task(ALLOWED_USER_ID))
        self.assertTrue(task.cancelled())
        self.assertNotIn(ALLOWED_USER_ID, self.runtime.active_tasks)


class AgentTaskLifecycleTests(HandlerTestCase):
    async def test_reply_not_delivered_during_run_is_sent_at_the_end(self) -> None:
        message = FakeMessage()
        reply = AgentReply(text="fertig", session_id=SESSION_ID, delivered_during_run=False)

        async def slow_ask(text: str, reply_observer=None) -> AgentReply:
            # Yield control at least once so the typing-indicator task, started
            # just before this call, gets a turn before the run finishes.
            await asyncio.sleep(0)
            return reply

        with patch.object(self.runtime, "ask", slow_ask):
            await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertEqual(message.replies, ["fertig"])
        self.assertIn("typing", message.chat.actions)
        self.assertNotIn(ALLOWED_USER_ID, self.runtime.active_tasks)

    async def test_reply_observer_delivers_a_reply_mid_run(self) -> None:
        message = FakeMessage()
        reply = AgentReply(text="final", session_id=SESSION_ID, delivered_during_run=True)

        async def ask_with_mid_run_delivery(text: str, reply_observer=None) -> AgentReply:
            await reply_observer("zwischenstand")
            return reply

        with patch.object(self.runtime, "ask", ask_with_mid_run_delivery):
            await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertEqual(message.replies, ["zwischenstand"])

    async def test_reply_already_delivered_during_run_is_not_repeated(self) -> None:
        message = FakeMessage()
        reply = AgentReply(text="fertig", session_id=SESSION_ID, delivered_during_run=True)
        with patch.object(self.runtime, "ask", AsyncMock(return_value=reply)):
            await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertEqual(message.replies, [])

    async def test_cancellation_propagates_after_cleanup(self) -> None:
        message = FakeMessage()
        with patch.object(
            self.runtime, "ask", AsyncMock(side_effect=asyncio.CancelledError())
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertNotIn(ALLOWED_USER_ID, self.runtime.active_tasks)

    async def test_bridge_error_is_reported_and_logged(self) -> None:
        message = FakeMessage()
        with patch.object(
            self.runtime, "ask", AsyncMock(side_effect=BridgeError("abgelehnt"))
        ):
            await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertEqual(message.replies, ["⚠️ abgelehnt"])

    async def test_unexpected_failure_reports_a_generic_error(self) -> None:
        message = FakeMessage()
        with patch.object(self.runtime, "ask", AsyncMock(side_effect=RuntimeError("boom"))):
            await _run_agent_task(ALLOWED_USER_ID, message, "hi")
        self.assertEqual(message.replies, ["⚠️ Unerwarteter interner Bot-Fehler."])

    async def test_keep_typing_stops_by_cancellation_after_the_indicator(self) -> None:
        chat = FakeChat()
        real_send_action = chat.send_action
        calls = 0

        async def send_action_then_cancel(action: str) -> None:
            nonlocal calls
            calls += 1
            await real_send_action(action)
            raise asyncio.CancelledError()

        chat.send_action = send_action_then_cancel
        with self.assertRaises(asyncio.CancelledError):
            await _keep_typing(chat)
        self.assertEqual(calls, 1)
        self.assertEqual(chat.actions, ["typing"])


class SystemctlTests(HandlerTestCase):
    async def test_run_systemctl_success(self) -> None:
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.returncode = 0
        with patch(
            "telegram_bot.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
        ):
            await _run_systemctl("start", "claude-remote-control.service")

    async def test_run_systemctl_failure_raises_with_a_bounded_message(self) -> None:
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b"unit not found " * 40))
        process.returncode = 1
        with patch(
            "telegram_bot.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
        ):
            with self.assertRaises(BridgeError) as caught:
                await _run_systemctl("start", "claude-remote-control.service")
        self.assertLessEqual(len(str(caught.exception)), 350)

    async def test_start_claude_remote_control_succeeds_on_first_probe(self) -> None:
        probe = AsyncMock()
        probe.wait = AsyncMock(return_value=0)
        with (
            patch("telegram_bot._run_systemctl", AsyncMock()),
            patch(
                "telegram_bot.asyncio.create_subprocess_exec", AsyncMock(return_value=probe)
            ),
        ):
            await _start_claude_remote_control()

    async def test_start_claude_remote_control_gives_up_after_all_probes(self) -> None:
        probe = AsyncMock()
        probe.wait = AsyncMock(return_value=1)
        with (
            patch("telegram_bot._run_systemctl", AsyncMock()),
            patch(
                "telegram_bot.asyncio.create_subprocess_exec", AsyncMock(return_value=probe)
            ),
            patch("telegram_bot.asyncio.sleep", AsyncMock()),
        ):
            with self.assertRaises(BridgeError):
                await _start_claude_remote_control()


class SendReplyTests(HandlerTestCase):
    async def test_long_reply_is_split_into_chunks(self) -> None:
        message = FakeMessage()
        long_reply = "a" * (MAX_REPLY_CHUNK + 5)
        await _send_reply(message, long_reply)
        self.assertEqual(len(message.replies), 2)

    async def test_reply_with_audio_uploads_it(self) -> None:
        audio_path = self.project / "clip.mp3"
        audio_path.write_bytes(b"sound")
        message = FakeMessage()
        await _send_reply(message, f"hier: {audio_path}")
        self.assertEqual(message.audio[0]["title"], "clip")
        self.assertEqual(message.audio[0]["filename"], "clip.mp3")
        self.assertEqual(message.audio[0]["audio"], b"sound")
        self.assertIn("upload_document", message.chat.actions)


class ReplyFormattingTests(unittest.TestCase):
    def test_switch_reply_without_a_bound_session(self) -> None:
        text = _switch_reply(Provider.CODEX, None)
        self.assertIn("Codex", text)
        self.assertIn("startet neu", text)

    def test_switch_reply_with_a_bound_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            record = _record(Provider.CODEX, project, root)
            text = _switch_reply(Provider.CODEX, record)
        self.assertIn(record.compact_label(), text)

    def test_takeover_reply_when_locally_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            record = _record(Provider.CODEX, project, root)
            text = _takeover_reply(record, locally_open=True)
        self.assertIn("noch lokal geöffnet", text)

    def test_takeover_reply_when_not_locally_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            record = _record(Provider.CODEX, project, root)
            text = _takeover_reply(record, locally_open=False)
        self.assertIn("setzt sie fort", text)


if __name__ == "__main__":
    unittest.main()
