import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from agent_bridge import (
    AgentBridge,
    BridgeError,
    Provider,
    SessionRecord,
    StateStore,
    build_claude_command,
    build_codex_command,
    discover_sessions,
    parse_control_intent,
    session_is_active,
    validate_session_id,
    _bounded_error,
    _codex_agent_message,
    _codex_thread_id,
    _codex_transcript_agent_message,
    _codex_turn_is_completed,
    _communicate_streaming,
    _kill_process_group,
    _process_group_id,
    _read_bounded_stream,
    _same_path,
    _send_and_close_stdin,
    _watch_codex_transcript,
)


SESSION_ID = "019fe190-8663-7583-8697-a910fb1154e8"
OTHER_SESSION_ID = "019fe643-8a24-7f90-ac01-8fc5242d91ed"


def _write_executable(directory: Path, name: str, body: str) -> str:
    """Create a standalone fake CLI binary; the real subprocess boundary faked at the edge."""
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return str(path)


class ControlIntentTests(unittest.TestCase):
    def test_natural_provider_switches_are_recognized(self) -> None:
        self.assertEqual(parse_control_intent("Wechsel zu Codex").provider, Provider.CODEX)
        self.assertEqual(parse_control_intent("geh bitte zu Claude").provider, Provider.CLAUDE)

    def test_takeover_and_new_session_are_recognized(self) -> None:
        self.assertEqual(parse_control_intent("Übernimm die Session").action, "takeover")
        self.assertEqual(parse_control_intent("starte eine neue Sitzung").action, "new")

    def test_ordinary_chat_is_not_intercepted(self) -> None:
        self.assertIsNone(parse_control_intent("Was macht Claude im Atelier besser?"))

    def test_claude_remote_request_is_recognized(self) -> None:
        self.assertEqual(
            parse_control_intent("Starte Claude Remote fürs Handy").action,
            "claude_remote",
        )


class DiscoveryTests(unittest.TestCase):
    def test_codex_discovers_only_root_sessions_in_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            self._write_codex_session(
                root / "rollout-root.jsonl", SESSION_ID, project, "cli"
            )
            self._write_codex_session(
                root / "rollout-subagent.jsonl",
                OTHER_SESSION_ID,
                project,
                {"subagent": {"thread_spawn": {"depth": 1}}},
            )
            records = discover_sessions(
                Provider.CODEX, project, codex_root=root
            )
            self.assertEqual([record.session_id for record in records], [SESSION_ID])

    def test_claude_discovers_root_session_and_title(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            session_file = root / f"{SESSION_ID}.jsonl"
            session_file.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "queue-operation", "sessionId": SESSION_ID}),
                        json.dumps(
                            {
                                "type": "user",
                                "sessionId": SESSION_ID,
                                "cwd": str(project),
                            }
                        ),
                        json.dumps(
                            {
                                "type": "ai-title",
                                "sessionId": SESSION_ID,
                                "aiTitle": "Atelier observer",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            records = discover_sessions(
                Provider.CLAUDE, project, claude_root=root
            )
            self.assertEqual(records[0].session_id, SESSION_ID)
            self.assertEqual(records[0].title, "Atelier observer")

    def test_open_transcript_marks_session_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            transcript = root / "rollout-root.jsonl"
            self._write_codex_session(transcript, SESSION_ID, project, "cli")
            record = discover_sessions(
                Provider.CODEX, project, codex_root=root
            )[0]
            descriptor_directory = root / "proc" / "42" / "fd"
            descriptor_directory.mkdir(parents=True)
            (descriptor_directory / "7").symlink_to(transcript)
            self.assertTrue(session_is_active(record, root / "proc"))

    def test_codex_resume_process_marks_session_active_without_open_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            transcript = root / "rollout-root.jsonl"
            self._write_codex_session(transcript, SESSION_ID, project, "cli")
            record = discover_sessions(
                Provider.CODEX, project, codex_root=root
            )[0]
            process_directory = root / "proc" / "42"
            (process_directory / "fd").mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(
                b"node\0/usr/bin/codex\0resume\0" + SESSION_ID.encode() + b"\0"
            )
            self.assertTrue(session_is_active(record, root / "proc"))

    def test_unrelated_process_mentioning_session_is_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            transcript = root / "rollout-root.jsonl"
            self._write_codex_session(transcript, SESSION_ID, project, "cli")
            record = discover_sessions(
                Provider.CODEX, project, codex_root=root
            )[0]
            process_directory = root / "proc" / "42"
            (process_directory / "fd").mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(
                b"rg\0" + SESSION_ID.encode() + b"\0"
            )
            self.assertFalse(session_is_active(record, root / "proc"))

    def test_bridge_owned_process_group_is_not_local_session_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            transcript = root / "rollout-root.jsonl"
            self._write_codex_session(transcript, SESSION_ID, project, "cli")
            record = discover_sessions(Provider.CODEX, project, codex_root=root)[0]
            process_directory = root / "proc" / "42"
            (process_directory / "fd").mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(
                b"node\0/usr/bin/codex\0exec\0resume\0"
                + SESSION_ID.encode()
                + b"\0"
            )
            (process_directory / "stat").write_text(
                "42 (codex) S 1 42 42 0 0 0\n", encoding="utf-8"
            )
            self.assertFalse(
                session_is_active(
                    record,
                    root / "proc",
                    ignored_process_groups=frozenset({42}),
                )
            )

    @staticmethod
    def _write_codex_session(
        path: Path, session_id: str, project: Path, source: object
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "cwd": str(project),
                        "source": source,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )


class CommandTests(unittest.TestCase):
    def test_codex_resume_uses_stdin_and_keeps_sandbox_bypass_absent(self) -> None:
        command = build_codex_command(
            "/usr/bin/codex",
            Path("/tmp/output"),
            session_id=SESSION_ID,
            model="gpt-5.6-sol",
        )
        self.assertEqual(command[-1], "-")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("secret prompt", command)

    def test_new_codex_session_is_workspace_scoped(self) -> None:
        command = build_codex_command(
            "/usr/bin/codex", Path("/tmp/output"), session_id=None, model=None
        )
        self.assertIn("workspace-write", command)

    def test_claude_does_not_skip_permissions(self) -> None:
        command = build_claude_command(
            "/usr/bin/claude",
            session_id=SESSION_ID,
            model="opus",
            system_prompt="mobile",
        )
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertIn("acceptEdits", command)


class CodexStreamingTests(unittest.IsolatedAsyncioTestCase):
    def test_only_completed_agent_messages_are_replies(self) -> None:
        self.assertEqual(
            _codex_agent_message(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": " Fertig. "},
                    }
                )
            ),
            "Fertig.",
        )
        self.assertIsNone(
            _codex_agent_message(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "command_execution", "text": "intern"},
                    }
                )
            )
        )
        self.assertTrue(
            _codex_turn_is_completed(json.dumps({"type": "turn.completed"}).encode())
        )
        self.assertFalse(
            _codex_turn_is_completed(json.dumps({"type": "turn.started"}).encode())
        )

    def test_transcript_messages_distinguish_commentary_from_final(self) -> None:
        commentary = json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "phase": "commentary",
                    "message": " Zwischenstand ",
                },
            }
        ).encode()
        final = json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "phase": "final_answer",
                    "message": " Fertig ",
                },
            }
        ).encode()
        self.assertEqual(
            _codex_transcript_agent_message(commentary),
            ("Zwischenstand", False),
        )
        self.assertEqual(
            _codex_transcript_agent_message(final),
            ("Fertig", True),
        )

    async def test_reply_is_observed_before_long_running_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observed: list[str] = []
            delivered = asyncio.Event()

            async def observer(reply: str) -> None:
                observed.append(reply)
                delivered.set()

            bridge = AgentBridge(
                working_directory=Path(directory),
                claude_binary="/bin/true",
                codex_binary="/bin/true",
                claude_model="opus",
                codex_model=None,
                timeout_seconds=30,
                system_prompt="mobile",
            )
            event = json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Live reply"},
                }
            )
            command = [
                sys.executable,
                "-c",
                f"import time; print({event!r}, flush=True); time.sleep(30)",
            ]
            task = asyncio.create_task(
                bridge._run(
                    command,
                    "message",
                    codex_reply_observer=observer,
                )
            )
            await asyncio.wait_for(delivered.wait(), timeout=2)
            self.assertEqual(observed, ["Live reply"])
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_completed_codex_turn_stops_persistent_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observed: list[str] = []

            async def observer(reply: str) -> None:
                observed.append(reply)

            bridge = AgentBridge(
                working_directory=Path(directory),
                claude_binary="/bin/true",
                codex_binary="/bin/true",
                claude_model="opus",
                codex_model=None,
                timeout_seconds=30,
                system_prompt="mobile",
            )
            reply_event = json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Turn reply"},
                }
            )
            completed_event = json.dumps({"type": "turn.completed"})
            command = [
                sys.executable,
                "-c",
                (
                    f"import time; print({reply_event!r}, flush=True); "
                    f"print({completed_event!r}, flush=True); time.sleep(30)"
                ),
            ]
            stdout, _ = await asyncio.wait_for(
                bridge._run(
                    command,
                    "message",
                    codex_reply_observer=observer,
                    stop_after_codex_turn=True,
                ),
                timeout=2,
            )
            self.assertIn('"type": "turn.completed"', stdout)
            self.assertEqual(observed, ["Turn reply"])
            self.assertEqual(bridge.active_process_groups, frozenset())

    async def test_transcript_final_stops_goal_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "session.jsonl"
            transcript.write_text("", encoding="utf-8")
            observed: list[str] = []

            async def observer(reply: str) -> None:
                observed.append(reply)

            bridge = AgentBridge(
                working_directory=root,
                claude_binary="/bin/true",
                codex_binary="/bin/true",
                claude_model="opus",
                codex_model=None,
                timeout_seconds=30,
                system_prompt="mobile",
            )
            events = [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "phase": "commentary",
                        "message": "Working",
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "phase": "final_answer",
                        "message": "Done",
                    },
                },
            ]
            script = (
                "import json,sys,time; "
                "stream=open(sys.argv[1], 'a', encoding='utf-8'); "
                f"events={events!r}; "
                "[stream.write(json.dumps(event)+'\\n') for event in events]; "
                "stream.flush(); time.sleep(30)"
            )
            stdout, _ = await asyncio.wait_for(
                bridge._run(
                    [sys.executable, "-c", script, str(transcript)],
                    "message",
                    codex_reply_observer=observer,
                    stop_after_codex_turn=True,
                    codex_transcript_path=transcript,
                    codex_transcript_offset=0,
                ),
                timeout=2,
            )
            self.assertEqual(stdout, "")
            self.assertEqual(observed, ["Working", "Done"])
            self.assertEqual(bridge.active_process_groups, frozenset())


class StateStoreTests(unittest.TestCase):
    def test_state_is_written_with_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = StateStore(path)
            state = {"provider": "codex", "sessions": {"codex": SESSION_ID}}
            store.save(state)
            self.assertEqual(store.load(), state)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_missing_state_file_defaults_to_codex_with_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "absent.json")
            self.assertEqual(store.load(), {"provider": "codex", "sessions": {}})

    def test_unreadable_state_file_raises_bridge_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(BridgeError):
                StateStore(path).load()

    def test_state_with_unknown_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({"provider": "gpt", "sessions": {}}), encoding="utf-8")
            with self.assertRaises(BridgeError):
                StateStore(path).load()

    def test_state_with_non_dict_sessions_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(
                json.dumps({"provider": "codex", "sessions": ["not", "a", "dict"]}),
                encoding="utf-8",
            )
            with self.assertRaises(BridgeError):
                StateStore(path).load()


class PureHelperTests(unittest.TestCase):
    def test_compact_label_appends_title_only_when_present(self) -> None:
        record = SessionRecord(
            provider=Provider.CLAUDE,
            session_id=SESSION_ID,
            working_directory=Path("/tmp/atelier"),
            modified_at=0,
            transcript_path=Path("/tmp/atelier/transcript.jsonl"),
        )
        self.assertNotIn("—", record.compact_label())
        titled = SessionRecord(**{**record.__dict__, "title": "Atelier observer"})
        self.assertIn("Atelier observer", titled.compact_label())

    def test_invalid_session_id_is_rejected(self) -> None:
        with self.assertRaises(BridgeError):
            validate_session_id("not-a-uuid")

    def test_bounded_error_passes_through_short_detail(self) -> None:
        self.assertEqual(_bounded_error("Fehler", "Detail"), "Fehler: Detail")

    def test_bounded_error_omits_colon_for_empty_detail(self) -> None:
        self.assertEqual(_bounded_error("Fehler", "   "), "Fehler")

    def test_bounded_error_truncates_long_detail(self) -> None:
        detail = "x" * 400
        bounded = _bounded_error("Fehler", detail)
        self.assertTrue(bounded.endswith("…"))
        self.assertEqual(len(bounded), len("Fehler: ") + 301)

    def test_kill_process_group_ignores_already_gone_process(self) -> None:
        # No process group is ever assigned this id on a running system.
        _kill_process_group(2**30)

    def test_codex_thread_id_finds_the_started_event_and_skips_malformed_lines(self) -> None:
        output = "\n".join(
            [
                "not json",
                json.dumps({"type": "turn.started"}),
                json.dumps({"type": "thread.started", "thread_id": OTHER_SESSION_ID}),
            ]
        )
        self.assertEqual(_codex_thread_id(output), OTHER_SESSION_ID)

    def test_codex_thread_id_is_none_without_a_started_event(self) -> None:
        self.assertIsNone(_codex_thread_id(json.dumps({"type": "turn.started"})))

    def test_codex_resume_without_model_omits_model_flag(self) -> None:
        command = build_codex_command(
            "/usr/bin/codex", Path("/tmp/output"), session_id=SESSION_ID, model=None
        )
        self.assertNotIn("--model", command)

    def test_new_codex_session_with_model_includes_model_flag(self) -> None:
        command = build_codex_command(
            "/usr/bin/codex", Path("/tmp/output"), session_id=None, model="gpt-5.6-sol"
        )
        self.assertIn("--model", command)

    def test_new_claude_session_omits_resume_flag(self) -> None:
        command = build_claude_command(
            "/usr/bin/claude", session_id=None, model="opus", system_prompt="mobile"
        )
        self.assertNotIn("--resume", command)

    def test_codex_transcript_message_rejects_non_agent_events(self) -> None:
        self.assertIsNone(_codex_transcript_agent_message(b"not json"))
        self.assertIsNone(
            _codex_transcript_agent_message(json.dumps({"type": "turn.started"}).encode())
        )
        self.assertIsNone(
            _codex_transcript_agent_message(
                json.dumps({"type": "event_msg", "payload": "not-a-dict"}).encode()
            )
        )
        self.assertIsNone(
            _codex_transcript_agent_message(
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "command_execution"}}
                ).encode()
            )
        )
        self.assertIsNone(
            _codex_transcript_agent_message(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "phase": "unknown"},
                    }
                ).encode()
            )
        )
        self.assertIsNone(
            _codex_transcript_agent_message(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "agent_message",
                            "phase": "final_answer",
                            "message": "   ",
                        },
                    }
                ).encode()
            )
        )

    def test_codex_turn_completed_check_rejects_invalid_json(self) -> None:
        self.assertFalse(_codex_turn_is_completed(b"not json"))

    def test_codex_agent_message_rejects_malformed_or_empty_payloads(self) -> None:
        self.assertIsNone(_codex_agent_message("not json"))
        self.assertIsNone(
            _codex_agent_message(
                json.dumps({"type": "item.completed", "item": "not-a-dict"})
            )
        )
        self.assertIsNone(
            _codex_agent_message(
                json.dumps(
                    {"type": "item.completed", "item": {"type": "agent_message", "text": "  "}}
                )
            )
        )


class DiscoveryEdgeCaseTests(unittest.TestCase):
    def test_nonexistent_codex_root_yields_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = discover_sessions(
                Provider.CODEX, Path(directory), codex_root=Path(directory) / "absent"
            )
            self.assertEqual(records, [])

    def test_nonexistent_claude_root_yields_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = discover_sessions(
                Provider.CLAUDE, Path(directory), claude_root=Path(directory) / "absent"
            )
            self.assertEqual(records, [])

    def test_malformed_codex_rollout_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rollout-broken.jsonl").write_text("not json\n", encoding="utf-8")
            records = discover_sessions(Provider.CODEX, root, codex_root=root)
            self.assertEqual(records, [])

    def test_claude_subagent_transcripts_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            subagents_dir = root / "subagents"
            subagents_dir.mkdir()
            (subagents_dir / f"{SESSION_ID}.jsonl").write_text(
                json.dumps({"sessionId": SESSION_ID, "cwd": str(project)}) + "\n",
                encoding="utf-8",
            )
            records = discover_sessions(Provider.CLAUDE, project, claude_root=root)
            self.assertEqual(records, [])

    def test_malformed_claude_transcript_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{SESSION_ID}.jsonl").write_text("not json\n", encoding="utf-8")
            records = discover_sessions(Provider.CLAUDE, root, claude_root=root)
            self.assertEqual(records, [])

    def test_claude_transcript_without_a_session_id_line_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (root / f"{SESSION_ID}.jsonl").write_text(
                json.dumps({"type": "queue-operation", "cwd": str(project)}) + "\n",
                encoding="utf-8",
            )
            records = discover_sessions(Provider.CLAUDE, project, claude_root=root)
            self.assertEqual(records, [])

    def test_claude_transcript_scan_stops_after_one_hundred_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            lines = ["{}" for _ in range(150)]
            lines.append(json.dumps({"sessionId": SESSION_ID, "cwd": str(project)}))
            (root / f"{SESSION_ID}.jsonl").write_text("\n".join(lines), encoding="utf-8")
            records = discover_sessions(Provider.CLAUDE, project, claude_root=root)
            self.assertEqual(records, [])

    def test_claude_transcript_without_title_still_matches_by_end_of_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (root / f"{SESSION_ID}.jsonl").write_text(
                json.dumps({"sessionId": SESSION_ID, "cwd": str(project)}) + "\n",
                encoding="utf-8",
            )
            records = discover_sessions(Provider.CLAUDE, project, claude_root=root)
            self.assertEqual(records[0].session_id, SESSION_ID)
            self.assertIsNone(records[0].title)

    def test_a_broken_symlink_loop_does_not_match_as_the_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            looping = root / "loop"
            looping.symlink_to(looping)
            self.assertFalse(_same_path(looping, root))


class SessionActiveEdgeCaseTests(unittest.TestCase):
    def _record(self, root: Path) -> SessionRecord:
        project = root / "project"
        project.mkdir(exist_ok=True)
        transcript = root / "rollout-root.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": SESSION_ID, "cwd": str(project), "source": "cli"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return discover_sessions(Provider.CODEX, project, codex_root=root)[0]

    def test_nonexistent_proc_root_is_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            self.assertFalse(session_is_active(record, root / "absent-proc"))

    def test_non_numeric_proc_entries_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            (root / "proc" / "self").mkdir(parents=True)
            self.assertFalse(session_is_active(record, root / "proc"))

    def test_process_without_the_session_id_in_its_arguments_is_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            process_directory = root / "proc" / "42"
            (process_directory / "fd").mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(b"bash\0-c\0true\0")
            self.assertFalse(session_is_active(record, root / "proc"))

    def test_unreadable_fd_directory_is_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            process_directory = root / "proc" / "42"
            process_directory.mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(b"bash\0")
            self.assertFalse(session_is_active(record, root / "proc"))

    def test_unrelated_open_descriptor_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            other_file = root / "unrelated.txt"
            other_file.write_text("x", encoding="utf-8")
            process_directory = root / "proc" / "42"
            (process_directory / "fd").mkdir(parents=True)
            (process_directory / "cmdline").write_bytes(b"bash\0")
            (process_directory / "fd" / "3").symlink_to(other_file)
            self.assertFalse(session_is_active(record, root / "proc"))

    def test_stat_without_a_command_terminator_yields_no_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process_directory = Path(directory)
            (process_directory / "stat").write_text("42 codex S 1 42 42", encoding="utf-8")
            self.assertIsNone(_process_group_id(process_directory))

    def test_stat_with_too_few_fields_yields_no_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process_directory = Path(directory)
            (process_directory / "stat").write_text("42 (codex) S 1", encoding="utf-8")
            self.assertIsNone(_process_group_id(process_directory))


class StreamingHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdin_write_survives_a_broken_pipe(self) -> None:
        class _AlreadyClosedStdin:
            def write(self, data: bytes) -> None:
                pass

            async def drain(self) -> None:
                raise BrokenPipeError()

            def close(self) -> None:
                pass

            async def wait_closed(self) -> None:
                pass

        await _send_and_close_stdin(_AlreadyClosedStdin(), "message")

    async def test_communicate_streaming_requires_all_three_pipes(self) -> None:
        class _IncompleteProcess:
            stdin = None
            stdout = object()
            stderr = object()

        with self.assertRaises(BridgeError):
            await _communicate_streaming(
                _IncompleteProcess(),
                "message",
                codex_reply_observer=None,
                stop_after_codex_turn=False,
                codex_transcript_path=None,
                codex_transcript_offset=None,
            )

    async def test_bounded_stream_raises_once_the_limit_is_exceeded(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"x" * 20 + b"\n")
        reader.feed_eof()
        with self.assertRaises(BridgeError):
            await _read_bounded_stream(
                reader,
                maximum_bytes=10,
                codex_reply_observer=None,
                codex_turn_completed=None,
            )


class TranscriptWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_transcript_file_raises_bridge_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BridgeError):
                await _watch_codex_transcript(
                    Path(directory) / "absent.jsonl",
                    start_offset=0,
                    reply_observer=None,
                    turn_completed=asyncio.Event(),
                )

    async def test_final_message_without_an_observer_still_completes_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "session.jsonl"
            final_event = json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "phase": "final_answer",
                        "message": "Done",
                    },
                }
            )
            transcript.write_text(final_event + "\n", encoding="utf-8")
            turn_completed = asyncio.Event()
            await _watch_codex_transcript(
                transcript, start_offset=0, reply_observer=None, turn_completed=turn_completed
            )
            self.assertTrue(turn_completed.is_set())

    async def test_irrelevant_and_partial_lines_are_skipped_until_the_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "session.jsonl"
            transcript.write_text("", encoding="utf-8")
            observed: list[str] = []

            async def observer(reply: str) -> None:
                observed.append(reply)

            turn_completed = asyncio.Event()
            task = asyncio.create_task(
                _watch_codex_transcript(
                    transcript,
                    start_offset=0,
                    reply_observer=observer,
                    turn_completed=turn_completed,
                )
            )
            await asyncio.sleep(0.25)
            irrelevant_event = json.dumps({"type": "turn.started"})
            final_event = json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "phase": "final_answer",
                        "message": "Done",
                    },
                }
            )
            with transcript.open("a", encoding="utf-8") as stream:
                stream.write(irrelevant_event + "\n")
                stream.write(final_event[: len(final_event) // 2])
                stream.flush()
            await asyncio.sleep(0.25)
            with transcript.open("a", encoding="utf-8") as stream:
                stream.write(final_event[len(final_event) // 2 :])
                stream.write("\n")
            await asyncio.wait_for(task, timeout=2)
            self.assertEqual(observed, ["Done"])
            self.assertTrue(turn_completed.is_set())

    async def test_turn_completed_externally_exits_the_watch_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "session.jsonl"
            transcript.write_text("", encoding="utf-8")
            turn_completed = asyncio.Event()
            task = asyncio.create_task(
                _watch_codex_transcript(
                    transcript,
                    start_offset=0,
                    reply_observer=None,
                    turn_completed=turn_completed,
                )
            )
            await asyncio.sleep(0.05)
            turn_completed.set()
            await asyncio.wait_for(task, timeout=1)

    async def test_unreadable_transcript_stops_a_still_running_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            working_directory = Path(directory)
            transcript = working_directory / "session.jsonl"
            transcript.write_text("", encoding="utf-8")
            os.chmod(transcript, 0o000)
            try:
                bridge = AgentBridge(
                    working_directory=working_directory,
                    claude_binary="/bin/true",
                    codex_binary=_write_executable(
                        working_directory,
                        "fake-codex",
                        "import sys, time\nsys.stdin.read()\ntime.sleep(5)\n",
                    ),
                    claude_model="opus",
                    codex_model=None,
                    timeout_seconds=5,
                    system_prompt="mobile",
                )
                with self.assertRaises(BridgeError):
                    await bridge.ask(
                        Provider.CODEX,
                        "message",
                        None,
                        codex_transcript_path=transcript,
                    )
            finally:
                os.chmod(transcript, 0o600)


CODEX_NEW_SESSION_BODY = """
import sys
args = sys.argv[1:]
output_path = args[args.index("--output-last-message") + 1]
sys.stdin.read()
print('{"type": "thread.started", "thread_id": "%s"}')
with open(output_path, "w", encoding="utf-8") as handle:
    handle.write("Hallo von Codex")
""" % OTHER_SESSION_ID

CODEX_RESUME_BODY = """
import sys
args = sys.argv[1:]
output_path = args[args.index("--output-last-message") + 1]
sys.stdin.read()
with open(output_path, "w", encoding="utf-8") as handle:
    handle.write("Antwort auf Wiederaufnahme")
"""

CODEX_NO_REPLY_BODY = """
import sys
sys.stdin.read()
"""

CODEX_MISSING_SESSION_BODY = """
import sys
args = sys.argv[1:]
output_path = args[args.index("--output-last-message") + 1]
sys.stdin.read()
with open(output_path, "w", encoding="utf-8") as handle:
    handle.write("Antwort ohne Session")
"""

CODEX_EMITS_THEN_EXITS_BODY = """
import sys
sys.stdin.read()
print('{"type": "item.completed", "item": {"type": "agent_message", "text": "Live"}}')
"""

CODEX_DUPLICATE_LIVE_REPLY_BODY = """
import sys
args = sys.argv[1:]
output_path = args[args.index("--output-last-message") + 1]
sys.stdin.read()
print('{"type": "item.completed", "item": {"type": "agent_message", "text": "Live"}}')
print('{"type": "item.completed", "item": {"type": "agent_message", "text": "Live"}}')
with open(output_path, "w", encoding="utf-8") as handle:
    handle.write("Live")
"""

CODEX_LIVE_REPLY_WITHOUT_OBSERVER_BODY = """
import sys
args = sys.argv[1:]
output_path = args[args.index("--output-last-message") + 1]
sys.stdin.read()
print('{"type": "item.completed", "item": {"type": "agent_message", "text": "Live"}}')
with open(output_path, "w", encoding="utf-8") as handle:
    handle.write("Live")
"""

CLAUDE_SUCCESS_BODY = """
import sys, json
sys.stdin.read()
print(json.dumps({"result": "Hallo!", "session_id": "%s", "is_error": False}))
""" % SESSION_ID

CLAUDE_INVALID_JSON_BODY = """
import sys
sys.stdin.read()
print("not json")
"""

CLAUDE_ERROR_BODY = """
import sys, json
sys.stdin.read()
print(json.dumps({"is_error": True, "result": "boom"}))
"""

CLAUDE_EMPTY_REPLY_BODY = """
import sys, json
sys.stdin.read()
print(json.dumps({"result": "   ", "session_id": "%s"}))
""" % SESSION_ID

CLAUDE_MISSING_SESSION_BODY = """
import sys, json
sys.stdin.read()
print(json.dumps({"result": "Hallo"}))
"""

CLAUDE_NONZERO_EXIT_BODY = """
import sys
sys.stdin.read()
sys.stderr.write("boom")
sys.exit(3)
"""


class AskEndToEndTests(unittest.IsolatedAsyncioTestCase):
    def _bridge(self, directory: Path, *, claude_body: str = "", codex_body: str = "") -> AgentBridge:
        claude_binary = (
            _write_executable(directory, "fake-claude", claude_body)
            if claude_body
            else "/bin/true"
        )
        codex_binary = (
            _write_executable(directory, "fake-codex", codex_body)
            if codex_body
            else "/bin/true"
        )
        return AgentBridge(
            working_directory=directory,
            claude_binary=claude_binary,
            codex_binary=codex_binary,
            claude_model="opus",
            codex_model=None,
            timeout_seconds=5,
            system_prompt="mobile",
        )

    async def test_ask_claude_returns_the_reply_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_SUCCESS_BODY)
            reply = await bridge.ask(Provider.CLAUDE, "Hallo", None)
            self.assertEqual(reply.text, "Hallo!")
            self.assertEqual(reply.session_id, SESSION_ID)

    async def test_ask_claude_rejects_invalid_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_INVALID_JSON_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)

    async def test_ask_claude_surfaces_a_reported_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_ERROR_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)

    async def test_ask_claude_rejects_an_empty_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_EMPTY_REPLY_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)

    async def test_ask_claude_rejects_a_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_MISSING_SESSION_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)

    async def test_ask_claude_reports_a_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), claude_body=CLAUDE_NONZERO_EXIT_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)

    async def test_ask_codex_starts_a_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_NEW_SESSION_BODY)
            reply = await bridge.ask(Provider.CODEX, "Hallo", None)
            self.assertEqual(reply.text, "Hallo von Codex")
            self.assertEqual(reply.session_id, OTHER_SESSION_ID)

    async def test_ask_codex_resumes_a_known_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_RESUME_BODY)
            reply = await bridge.ask(Provider.CODEX, "Hallo", SESSION_ID)
            self.assertEqual(reply.text, "Antwort auf Wiederaufnahme")
            self.assertEqual(reply.session_id, SESSION_ID)

    async def test_ask_codex_rejects_an_empty_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_NO_REPLY_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CODEX, "Hallo", SESSION_ID)

    async def test_ask_codex_rejects_a_missing_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_MISSING_SESSION_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CODEX, "Hallo", None)

    async def test_ask_codex_rejects_an_unreadable_transcript_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_RESUME_BODY)
            with self.assertRaises(BridgeError):
                await bridge.ask(
                    Provider.CODEX,
                    "Hallo",
                    SESSION_ID,
                    codex_transcript_path=Path(directory) / "absent.jsonl",
                )

    async def test_ask_propagates_a_failing_reply_observer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_EMITS_THEN_EXITS_BODY)

            async def failing_observer(reply: str) -> None:
                raise ValueError("observer exploded")

            with self.assertRaises(ValueError):
                await bridge.ask(
                    Provider.CODEX, "Hallo", SESSION_ID, reply_observer=failing_observer
                )

    async def test_ask_codex_delivers_a_live_reply_exactly_once_when_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(Path(directory), codex_body=CODEX_DUPLICATE_LIVE_REPLY_BODY)
            observed: list[str] = []

            async def observer(reply: str) -> None:
                observed.append(reply)

            reply = await bridge.ask(
                Provider.CODEX, "Hallo", SESSION_ID, reply_observer=observer
            )
            self.assertEqual(observed, ["Live"])
            self.assertTrue(reply.delivered_during_run)

    async def test_ask_codex_without_a_reply_observer_still_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(
                Path(directory), codex_body=CODEX_LIVE_REPLY_WITHOUT_OBSERVER_BODY
            )
            reply = await bridge.ask(Provider.CODEX, "Hallo", SESSION_ID)
            self.assertEqual(reply.text, "Live")
            self.assertFalse(reply.delivered_during_run)

    async def test_ask_times_out_a_slow_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._bridge(
                Path(directory),
                claude_body="import sys, time\nsys.stdin.read()\ntime.sleep(5)\n",
            )
            bridge.timeout_seconds = 0.05
            with self.assertRaises(BridgeError):
                await bridge.ask(Provider.CLAUDE, "Hallo", None)


if __name__ == "__main__":
    unittest.main()
