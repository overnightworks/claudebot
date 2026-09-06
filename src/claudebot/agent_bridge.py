"""Provider-neutral session discovery and CLI execution for the Telegram bridge."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import AbstractSet, Awaitable, Callable, Iterable
from uuid import UUID


class Provider(StrEnum):
    CLAUDE = "claude"
    CODEX = "codex"


@dataclass(frozen=True)
class SessionRecord:
    provider: Provider
    session_id: str
    working_directory: Path
    modified_at: float
    transcript_path: Path
    title: str | None = None

    def compact_label(self) -> str:
        timestamp = datetime.fromtimestamp(self.modified_at).strftime("%d.%m. %H:%M")
        title = f" — {self.title}" if self.title else ""
        return (
            f"{timestamp} · {self.working_directory.name} · "
            f"{self.session_id[:8]}…{title}"
        )


@dataclass(frozen=True)
class AgentReply:
    text: str
    session_id: str
    delivered_during_run: bool = False


@dataclass(frozen=True)
class ControlIntent:
    action: str
    provider: Provider | None = None


class BridgeError(RuntimeError):
    """A bounded error suitable for returning to the operator."""


def parse_control_intent(text: str) -> ControlIntent | None:
    """Recognize only unambiguous session-control sentences."""
    normalized = " ".join(text.casefold().strip().split())
    switch_words = r"(?:wechsel|wechsle|switch|geh|gehe)"
    if re.search(rf"\b{switch_words}\b.*\bcodex\b", normalized):
        return ControlIntent("switch", Provider.CODEX)
    if re.search(rf"\b{switch_words}\b.*\bclaude\b", normalized):
        return ControlIntent("switch", Provider.CLAUDE)
    if re.search(
        r"\b(?:übernimm|übernehme|uebernimm|uebernehme|resume|fortsetzen)\b"
        r".*\b(?:session|sitzung)\b",
        normalized,
    ):
        return ControlIntent("takeover")
    if re.search(
        r"\b(?:neue|neuer|new)\b.*\b(?:session|sitzung)\b", normalized
    ):
        return ControlIntent("new")
    if re.search(
        r"\b(?:verbinde|verbind|starte|start)\b.*\bclaude\b.*\b(?:remote|browser|handy)\b"
        r"|\bclaude\b.*\bremote(?: control)?\b.*\b(?:starte|start|verbinde)\b",
        normalized,
    ):
        return ControlIntent("claude_remote")
    return None


def validate_session_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise BridgeError("Die Session-ID ist keine gültige UUID.") from exc


def discover_sessions(
    provider: Provider,
    working_directory: Path,
    *,
    codex_root: Path | None = None,
    claude_root: Path | None = None,
) -> list[SessionRecord]:
    if provider is Provider.CODEX:
        root = codex_root or Path.home() / ".codex" / "sessions"
        records = _discover_codex_sessions(root)
    else:
        root = claude_root or Path.home() / ".claude" / "projects"
        records = _discover_claude_sessions(root)

    expected = _resolved_path(working_directory)
    if expected is None:
        return []
    matching = [
        record
        for record in records
        if _same_path(record.working_directory, expected)
    ]
    return sorted(matching, key=lambda record: record.modified_at, reverse=True)


def _resolved_path(path: Path) -> Path | None:
    """Return the resolved path, or None when the system cannot resolve it."""
    try:
        return path.resolve()
    # Resolution fails for a vanished working directory, and Python 3.12
    # additionally reports a symlink loop as RuntimeError where later versions
    # resolve it lexically.
    except (OSError, RuntimeError):
        return None


def _same_path(left: Path, right: Path) -> bool:
    resolved_left = _resolved_path(left)
    # A path that cannot be resolved is not the path we mean.
    return resolved_left is not None and resolved_left == _resolved_path(right)


def _discover_codex_sessions(root: Path) -> Iterable[SessionRecord]:
    if not root.is_dir():
        return []
    records: list[SessionRecord] = []
    for path in root.rglob("rollout-*.jsonl"):
        try:
            with path.open(encoding="utf-8") as stream:
                first_line = stream.readline()
            event = json.loads(first_line)
            payload = event.get("payload", {})
            source = payload.get("source")
            if isinstance(source, dict) and "subagent" in source:
                continue
            session_id = validate_session_id(str(payload["id"]))
            working_directory = Path(payload["cwd"])
            records.append(
                SessionRecord(
                    provider=Provider.CODEX,
                    session_id=session_id,
                    working_directory=working_directory,
                    modified_at=path.stat().st_mtime,
                    transcript_path=path,
                )
            )
        except (BridgeError, KeyError, OSError, TypeError, json.JSONDecodeError):
            continue
    return records


def _discover_claude_sessions(root: Path) -> Iterable[SessionRecord]:
    if not root.is_dir():
        return []
    records: list[SessionRecord] = []
    for path in root.rglob("*.jsonl"):
        if "subagents" in path.parts:
            continue
        record = _parse_claude_session_file(path)
        if record is not None:
            records.append(record)
    return records


def _parse_claude_session_file(path: Path) -> SessionRecord | None:
    try:
        with path.open(encoding="utf-8") as stream:
            session_id, working_directory, title = _read_claude_session_fields(stream)
        if not session_id or not working_directory:
            return None
        modified_at = path.stat().st_mtime
    except (BridgeError, OSError, TypeError, json.JSONDecodeError):
        return None
    return SessionRecord(
        provider=Provider.CLAUDE,
        session_id=session_id,
        working_directory=working_directory,
        modified_at=modified_at,
        transcript_path=path,
        title=title,
    )


def _read_claude_session_fields(
    stream: Iterable[str],
) -> tuple[str | None, Path | None, str | None]:
    """Scan the leading transcript lines for the session's identity and title."""
    session_id: str | None = None
    working_directory: Path | None = None
    title: str | None = None
    for line_number, line in enumerate(stream):
        if line_number >= 100:
            break
        event = json.loads(line)
        raw_session_id = event.get("sessionId")
        if raw_session_id:
            session_id = validate_session_id(str(raw_session_id))
        if event.get("cwd"):
            working_directory = Path(event["cwd"])
        if event.get("type") == "ai-title" and event.get("aiTitle"):
            title = str(event["aiTitle"])
        if session_id and working_directory and title:
            break
    return session_id, working_directory, title


def session_is_active(
    record: SessionRecord,
    proc_root: Path = Path("/proc"),
    *,
    ignored_process_groups: AbstractSet[int] = frozenset(),
) -> bool:
    """Return whether a live provider process owns the session.

    Codex does not keep its rollout file descriptor open for the whole terminal
    lifetime, so descriptor ownership alone misses a still-running ``resume``.
    The exact session id in a Codex process argv is the second independent proof.
    """
    expected = _resolved_path(record.transcript_path)
    try:
        # Listing eagerly keeps a missing or unreadable proc root inside this
        # guard: Python 3.12's iterdir() only fails once it is iterated.
        process_directories = list(proc_root.iterdir())
    except OSError:
        return False
    return any(
        _process_owns_session(process_directory, record, expected, ignored_process_groups)
        for process_directory in process_directories
        if process_directory.name.isdigit()
    )


def _process_owns_session(
    process_directory: Path,
    record: SessionRecord,
    expected_transcript: Path | None,
    ignored_process_groups: AbstractSet[int],
) -> bool:
    if _process_group_id(process_directory) in ignored_process_groups:
        return False
    if _process_runs_session(process_directory, record):
        return True
    # A transcript path that cannot be resolved leaves the argv proof standing
    # rather than reporting the session as inactive.
    return expected_transcript is not None and _process_has_transcript_open(
        process_directory, expected_transcript
    )


def _process_has_transcript_open(process_directory: Path, expected: Path) -> bool:
    descriptor_directory = process_directory / "fd"
    try:
        # Listing eagerly keeps the failing directory inside this guard, as in
        # session_is_active; a single descriptor that cannot be resolved is
        # merely not the transcript, so the remaining descriptors still count.
        descriptors = list(descriptor_directory.iterdir())
    except OSError:
        return False
    return any(_resolved_path(descriptor) == expected for descriptor in descriptors)


def _process_group_id(process_directory: Path) -> int | None:
    """Read Linux proc stat's process-group field without trusting the command name."""
    try:
        stat = (process_directory / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields_after_command = stat[command_end + 1 :].split()
    try:
        # Fields after ``comm`` begin with state, ppid, then pgrp.
        return int(fields_after_command[2])
    except (IndexError, ValueError):
        return None


def _process_runs_session(
    process_directory: Path, record: SessionRecord
) -> bool:
    try:
        arguments = tuple(
            value
            for value in (process_directory / "cmdline").read_bytes().split(b"\0")
            if value
        )
    except OSError:
        return False
    session_id = record.session_id.encode("ascii")
    if session_id not in arguments:
        return False
    return any(Path(os.fsdecode(argument)).name == record.provider.value for argument in arguments)


class StateStore:
    """Small durable owner for the selected provider and provider sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {"provider": Provider.CODEX.value, "sessions": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeError("Der gespeicherte Bot-Zustand ist nicht lesbar.") from exc
        provider = data.get("provider")
        sessions = data.get("sessions")
        if provider not in {item.value for item in Provider} or not isinstance(
            sessions, dict
        ):
            raise BridgeError("Der gespeicherte Bot-Zustand ist ungültig.")
        return {"provider": provider, "sessions": sessions}

    def save(self, state: dict[str, object]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", text=True
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True)
                stream.write("\n")
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)


class AgentBridge:
    def __init__(
        self,
        *,
        working_directory: Path,
        claude_binary: str,
        codex_binary: str,
        claude_model: str,
        codex_model: str | None,
        timeout_seconds: int,
        system_prompt: str,
    ) -> None:
        self.working_directory = working_directory.resolve()
        self.claude_binary = claude_binary
        self.codex_binary = codex_binary
        self.claude_model = claude_model
        self.codex_model = codex_model
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        self._active_process_groups: set[int] = set()

    @property
    def active_process_groups(self) -> frozenset[int]:
        """Process groups launched by this bridge, never local terminal ownership."""
        return frozenset(self._active_process_groups)

    async def ask(
        self,
        provider: Provider,
        message: str,
        session_id: str | None,
        *,
        reply_observer: Callable[[str], Awaitable[None]] | None = None,
        codex_transcript_path: Path | None = None,
    ) -> AgentReply:
        if provider is Provider.CODEX:
            return await self._ask_codex(
                message,
                session_id,
                reply_observer=reply_observer,
                transcript_path=codex_transcript_path,
            )
        return await self._ask_claude(message, session_id)

    async def _ask_codex(
        self,
        message: str,
        session_id: str | None,
        *,
        reply_observer: Callable[[str], Awaitable[None]] | None,
        transcript_path: Path | None,
    ) -> AgentReply:
        output_descriptor, output_name = tempfile.mkstemp(prefix="codex-reply-")
        os.close(output_descriptor)
        output_path = Path(output_name)
        os.chmod(output_path, 0o600)
        try:
            command = build_codex_command(
                self.codex_binary,
                output_path,
                session_id=session_id,
                model=self.codex_model,
            )
            delivered_during_run = False
            observed_reply: str | None = None
            delivered_replies: set[str] = set()

            async def observe_reply(reply: str) -> None:
                nonlocal delivered_during_run, observed_reply
                if reply in delivered_replies:
                    return
                delivered_replies.add(reply)
                observed_reply = reply
                if reply_observer is not None:
                    await reply_observer(reply)
                    delivered_during_run = True

            transcript_offset: int | None = None
            if transcript_path is not None:
                try:
                    transcript_offset = transcript_path.stat().st_size
                except OSError as exc:
                    raise BridgeError(
                        "Das Codex-Sessionprotokoll ist nicht lesbar."
                    ) from exc

            stdout, stderr = await self._run(
                command,
                message,
                codex_reply_observer=observe_reply,
                stop_after_codex_turn=True,
                codex_transcript_path=transcript_path,
                codex_transcript_offset=transcript_offset,
            )
            reply = output_path.read_text(encoding="utf-8").strip() or observed_reply or ""
            if not reply:
                raise BridgeError(_bounded_error("Codex lieferte keine Antwort", stderr))
            returned_session = session_id or _codex_thread_id(stdout)
            if not returned_session:
                raise BridgeError("Codex lieferte keine Session-ID.")
            return AgentReply(
                reply,
                validate_session_id(returned_session),
                delivered_during_run,
            )
        finally:
            output_path.unlink(missing_ok=True)

    async def _ask_claude(
        self, message: str, session_id: str | None
    ) -> AgentReply:
        command = build_claude_command(
            self.claude_binary,
            session_id=session_id,
            model=self.claude_model,
            system_prompt=self.system_prompt,
        )
        stdout, stderr = await self._run(command, message)
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise BridgeError(_bounded_error("Claude lieferte ungültige Daten", stderr)) from exc
        if data.get("is_error"):
            raise BridgeError(_bounded_error("Claude meldete einen Fehler", str(data.get("result", ""))))
        reply = str(data.get("result", "")).strip()
        returned_session = data.get("session_id") or session_id
        if not reply or not returned_session:
            raise BridgeError("Claude lieferte keine vollständige Antwort.")
        return AgentReply(reply, validate_session_id(str(returned_session)))

    async def _run(
        self,
        command: list[str],
        message: str,
        *,
        codex_reply_observer: Callable[[str], Awaitable[None]] | None = None,
        stop_after_codex_turn: bool = False,
        codex_transcript_path: Path | None = None,
        codex_transcript_offset: int | None = None,
    ) -> tuple[str, str]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_directory,
            start_new_session=True,
            limit=2_000_001,
        )
        self._active_process_groups.add(process.pid)
        try:
            stdout, stderr, stopped_after_turn = await asyncio.wait_for(
                _communicate_streaming(
                    process,
                    message,
                    codex_reply_observer=codex_reply_observer,
                    stop_after_codex_turn=stop_after_codex_turn,
                    codex_transcript_path=codex_transcript_path,
                    codex_transcript_offset=codex_transcript_offset,
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            _kill_process_group(process.pid)
            await process.wait()
            raise BridgeError("Der Agent hat das Zeitlimit überschritten.") from exc
        except asyncio.CancelledError:
            _kill_process_group(process.pid)
            await process.wait()
            raise
        except Exception:
            _kill_process_group(process.pid)
            await process.wait()
            raise
        finally:
            self._active_process_groups.discard(process.pid)
        decoded_stdout = stdout.decode("utf-8", errors="replace")
        decoded_stderr = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0 and not stopped_after_turn:
            raise BridgeError(
                _bounded_error(
                    f"Der Agent endete mit Code {process.returncode}",
                    decoded_stderr or decoded_stdout,
                )
            )
        return decoded_stdout, decoded_stderr


@dataclass
class _StreamingTasks:
    """The concurrent tasks one agent invocation juggles while it streams output."""

    stdout: asyncio.Task[bytes]
    stderr: asyncio.Task[bytes]
    process: asyncio.Task[int]
    completion: asyncio.Task[bool]
    transcript: asyncio.Task[None] | None

    def all(self) -> tuple[asyncio.Task, ...]:
        return tuple(
            task
            for task in (self.stdout, self.stderr, self.transcript, self.process, self.completion)
            if task is not None
        )


async def _communicate_streaming(
    process: asyncio.subprocess.Process,
    message: str,
    *,
    codex_reply_observer: Callable[[str], Awaitable[None]] | None,
    stop_after_codex_turn: bool,
    codex_transcript_path: Path | None,
    codex_transcript_offset: int | None,
) -> tuple[bytes, bytes, bool]:
    await _send_and_close_stdin(process.stdin, message)
    codex_turn_completed = asyncio.Event()
    tasks = _start_streaming_tasks(
        process,
        codex_reply_observer=codex_reply_observer,
        stop_after_codex_turn=stop_after_codex_turn,
        codex_transcript_path=codex_transcript_path,
        codex_transcript_offset=codex_transcript_offset,
        codex_turn_completed=codex_turn_completed,
    )
    try:
        stopped_after_turn = await _wait_for_turn_or_process_exit(
            process, tasks, codex_turn_completed
        )
        return await tasks.stdout, await tasks.stderr, stopped_after_turn
    finally:
        await _cancel_remaining_tasks(tasks.all())


async def _send_and_close_stdin(stdin: asyncio.StreamWriter, message: str) -> None:
    stdin.write(message.encode("utf-8"))
    try:
        await stdin.drain()
        stdin.close()
        await stdin.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass


def _start_streaming_tasks(
    process: asyncio.subprocess.Process,
    *,
    codex_reply_observer: Callable[[str], Awaitable[None]] | None,
    stop_after_codex_turn: bool,
    codex_transcript_path: Path | None,
    codex_transcript_offset: int | None,
    codex_turn_completed: asyncio.Event,
) -> _StreamingTasks:
    stdout_task = asyncio.create_task(
        _read_bounded_stream(
            process.stdout,
            maximum_bytes=2_000_000,
            codex_reply_observer=codex_reply_observer,
            codex_turn_completed=(
                codex_turn_completed if stop_after_codex_turn else None
            ),
        )
    )
    stderr_task = asyncio.create_task(
        _read_bounded_stream(
            process.stderr,
            maximum_bytes=200_000,
            codex_reply_observer=None,
            codex_turn_completed=None,
        )
    )
    transcript_task: asyncio.Task[None] | None = None
    if codex_transcript_path is not None and codex_transcript_offset is not None:
        transcript_task = asyncio.create_task(
            _watch_codex_transcript(
                codex_transcript_path,
                start_offset=codex_transcript_offset,
                reply_observer=codex_reply_observer,
                turn_completed=codex_turn_completed,
            )
        )
    return _StreamingTasks(
        stdout=stdout_task,
        stderr=stderr_task,
        process=asyncio.create_task(process.wait()),
        completion=asyncio.create_task(codex_turn_completed.wait()),
        transcript=transcript_task,
    )


async def _wait_for_turn_or_process_exit(
    process: asyncio.subprocess.Process,
    tasks: _StreamingTasks,
    codex_turn_completed: asyncio.Event,
) -> bool:
    completion_sources: set[asyncio.Task[object]] = {tasks.process, tasks.completion}
    if tasks.transcript is not None:
        completion_sources.add(tasks.transcript)
    await asyncio.wait(completion_sources, return_when=asyncio.FIRST_COMPLETED)
    if (
        tasks.transcript is not None
        and tasks.transcript.done()
        and not codex_turn_completed.is_set()
    ):
        await tasks.transcript
    stopped_after_turn = tasks.completion.done() and tasks.completion.result()
    if stopped_after_turn and process.returncode is None:
        _kill_process_group(process.pid)
    await tasks.process
    return stopped_after_turn


async def _cancel_remaining_tasks(tasks: tuple[asyncio.Task, ...]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _watch_codex_transcript(
    transcript_path: Path,
    *,
    start_offset: int,
    reply_observer: Callable[[str], Awaitable[None]] | None,
    turn_completed: asyncio.Event,
) -> None:
    """Deliver one resumed Codex turn even when a persistent goal auto-continues.

    Only the initial ``open`` runs off the event loop thread: the polling reads
    that follow already yield to other tasks via ``asyncio.sleep`` whenever no
    full line is available, so they do not need their own thread hop.
    """
    try:
        stream = await asyncio.to_thread(transcript_path.open, "rb")
        try:
            stream.seek(start_offset)
            while not turn_completed.is_set():
                line_offset = stream.tell()
                line = stream.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                if not line.endswith(b"\n"):
                    stream.seek(line_offset)
                    await asyncio.sleep(0.1)
                    continue
                reply = _codex_transcript_agent_message(line)
                if reply is None:
                    continue
                text, is_final = reply
                if reply_observer is not None:
                    await reply_observer(text)
                if is_final:
                    turn_completed.set()
                    return
        finally:
            stream.close()
    except OSError as exc:
        raise BridgeError("Das Codex-Sessionprotokoll wurde unlesbar.") from exc


def _codex_transcript_agent_message(line: bytes) -> tuple[str, bool] | None:
    try:
        event = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    payload = event.get("payload")
    if event.get("type") != "event_msg" or not isinstance(payload, dict):
        return None
    if payload.get("type") != "agent_message":
        return None
    phase = payload.get("phase")
    if phase not in {"commentary", "final_answer"}:
        return None
    text = payload.get("message")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip(), phase == "final_answer"


async def _read_bounded_stream(
    stream: asyncio.StreamReader,
    *,
    maximum_bytes: int,
    codex_reply_observer: Callable[[str], Awaitable[None]] | None,
    codex_turn_completed: asyncio.Event | None,
) -> bytes:
    output = bytearray()
    while line := await stream.readline():
        output.extend(line)
        if len(output) > maximum_bytes:
            raise BridgeError("Die Agent-Ausgabe war unerwartet groß.")
        if codex_reply_observer is not None:
            reply = _codex_agent_message(line.decode("utf-8", errors="replace"))
            if reply is not None:
                await codex_reply_observer(reply)
        if codex_turn_completed is not None and _codex_turn_is_completed(line):
            codex_turn_completed.set()
            break
    return bytes(output)


def _codex_turn_is_completed(line: bytes) -> bool:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    return event.get("type") == "turn.completed"


def _codex_agent_message(line: str) -> str | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    item = event.get("item")
    if event.get("type") != "item.completed" or not isinstance(item, dict):
        return None
    if item.get("type") != "agent_message":
        return None
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def build_codex_command(
    binary: str,
    output_path: Path,
    *,
    session_id: str | None,
    model: str | None,
) -> list[str]:
    if session_id:
        command = [
            binary,
            "exec",
            "resume",
            "--json",
            "--output-last-message",
            str(output_path),
        ]
        if model:
            command.extend(["--model", model])
        command.extend([validate_session_id(session_id), "-"])
        return command

    command = [
        binary,
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--output-last-message",
        str(output_path),
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")
    return command


def build_claude_command(
    binary: str,
    *,
    session_id: str | None,
    model: str,
    system_prompt: str,
) -> list[str]:
    command = [
        binary,
        "--print",
        "--model",
        model,
        "--output-format",
        "json",
        "--append-system-prompt",
        system_prompt,
        "--permission-mode",
        "acceptEdits",
    ]
    if session_id:
        command.extend(["--resume", validate_session_id(session_id)])
    return command


def _codex_thread_id(output: str) -> str | None:
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _kill_process_group(process_id: int) -> None:
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _bounded_error(prefix: str, detail: str) -> str:
    compact = " ".join(detail.split())
    if len(compact) > 300:
        compact = f"{compact[:300]}…"
    return f"{prefix}: {compact}" if compact else prefix
