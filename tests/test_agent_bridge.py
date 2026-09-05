import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from agent_bridge import (
    AgentBridge,
    Provider,
    StateStore,
    build_claude_command,
    build_codex_command,
    discover_sessions,
    parse_control_intent,
    session_is_active,
    _codex_agent_message,
    _codex_transcript_agent_message,
    _codex_turn_is_completed,
)


SESSION_ID = "019fe190-8663-7583-8697-a910fb1154e8"
OTHER_SESSION_ID = "019fe643-8a24-7f90-ac01-8fc5242d91ed"


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


if __name__ == "__main__":
    unittest.main()
