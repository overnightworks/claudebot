#!/usr/bin/env python3
"""Private Telegram bridge for local Claude Code and Codex CLI sessions."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent_bridge import (
    AgentBridge,
    BridgeError,
    ControlIntent,
    Provider,
    SessionRecord,
    StateStore,
    discover_sessions,
    parse_control_intent,
    session_is_active,
    validate_session_id,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent-telegram")
# Telegram's Bot API token is embedded in request URLs. Keep transport loggers
# below INFO so credentials never enter the durable bot log.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MAX_MESSAGE_LENGTH = 16_000
MAX_REPLY_CHUNK = 4_000
MAX_AUDIO_BYTES = 50 * 1024 * 1024

SYSTEM_PROMPT = (
    "You are continuing a local coding-agent conversation with Felix via Telegram. "
    "Keep status answers concise enough for a phone. Preserve the session's existing "
    "goal, repository guidance, provider tools, and authority boundaries. Never claim "
    "that a session or provider switched; the Telegram bridge owns that operation."
)


@dataclass(frozen=True)
class BotConfig:
    telegram_token: str
    allowed_user_ids: frozenset[int]
    project_directory: Path
    claude_binary: str
    codex_binary: str
    claude_model: str
    codex_model: str | None
    timeout_seconds: int
    state_path: Path


def load_config() -> BotConfig:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise BridgeError("TELEGRAM_BOT_TOKEN fehlt.")

    allowed_raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    if not allowed_raw:
        raise BridgeError(
            "TELEGRAM_ALLOWED_USER_IDS muss vor dem Start gesetzt sein; "
            "automatische Erstregistrierung ist absichtlich deaktiviert."
        )
    try:
        allowed_user_ids = frozenset(
            int(value.strip()) for value in allowed_raw.split(",") if value.strip()
        )
    except ValueError as exc:
        raise BridgeError("TELEGRAM_ALLOWED_USER_IDS enthält keine gültigen IDs.") from exc
    if not allowed_user_ids:
        raise BridgeError("TELEGRAM_ALLOWED_USER_IDS ist leer.")

    project_raw = os.environ.get("AGENT_PROJECT_DIR") or os.environ.get(
        "SONGMAKER_DIR"
    )
    project_directory = Path(project_raw or Path.cwd()).expanduser().resolve()
    if not project_directory.is_dir():
        raise BridgeError(f"Projektverzeichnis existiert nicht: {project_directory}")

    claude_binary = _resolve_binary("CLAUDE_BIN", "claude")
    codex_binary = _resolve_binary("CODEX_BIN", "codex")
    state_path = Path(
        os.environ.get(
            "AGENT_BOT_STATE_PATH",
            Path.home() / ".local" / "state" / "agent-telegram" / "state.json",
        )
    ).expanduser()
    timeout_seconds = _positive_integer("AGENT_TIMEOUT", 1_800)
    codex_model = os.environ.get("CODEX_MODEL", "").strip() or None

    return BotConfig(
        telegram_token=token,
        allowed_user_ids=allowed_user_ids,
        project_directory=project_directory,
        claude_binary=claude_binary,
        codex_binary=codex_binary,
        claude_model=os.environ.get("CLAUDE_MODEL", "opus"),
        codex_model=codex_model,
        timeout_seconds=timeout_seconds,
        state_path=state_path,
    )


def _resolve_binary(environment_name: str, command: str) -> str:
    configured = os.environ.get(environment_name, "").strip()
    candidates = [
        configured,
        shutil.which(command) or "",
        str(Path.home() / ".local" / "bin" / command),
        str(Path("/usr/local/bin") / command),
        str(Path("/usr/bin") / command),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().absolute()
        stable_path = str(path)
        if stable_path in seen:
            continue
        seen.add(stable_path)
        if path.is_file() and os.access(path, os.X_OK):
            if configured and stable_path != str(Path(configured).expanduser().absolute()):
                log.warning(
                    "Ignoring stale %s path; using %s.", environment_name, stable_path
                )
            return stable_path
    raise BridgeError(f"{command} CLI wurde nicht gefunden ({environment_name}).")


def _positive_integer(environment_name: str, default: int) -> int:
    raw = os.environ.get(environment_name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise BridgeError(f"{environment_name} muss eine Zahl sein.") from exc
    if value <= 0:
        raise BridgeError(f"{environment_name} muss größer als null sein.")
    return value


class BotRuntime:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.store = StateStore(config.state_path)
        self.state = self.store.load()
        self.bridge = AgentBridge(
            working_directory=config.project_directory,
            claude_binary=config.claude_binary,
            codex_binary=config.codex_binary,
            claude_model=config.claude_model,
            codex_model=config.codex_model,
            timeout_seconds=config.timeout_seconds,
            system_prompt=SYSTEM_PROMPT,
        )
        self.active_tasks: dict[int, asyncio.Task[None]] = {}
        self.session_choices: dict[int, list[SessionRecord]] = {}
        self._seed_configured_sessions()

    @property
    def provider(self) -> Provider:
        return Provider(str(self.state["provider"]))

    def session_id(self, provider: Provider | None = None) -> str | None:
        sessions = self.state["sessions"]
        assert isinstance(sessions, dict)
        value = sessions.get((provider or self.provider).value)
        return str(value) if value else None

    def _seed_configured_sessions(self) -> None:
        sessions = self.state["sessions"]
        assert isinstance(sessions, dict)
        changed = False
        for provider, environment_name in (
            (Provider.CODEX, "CODEX_SESSION_ID"),
            (Provider.CLAUDE, "CLAUDE_SESSION_ID"),
        ):
            configured = os.environ.get(environment_name, "").strip()
            if configured and provider.value not in sessions:
                sessions[provider.value] = validate_session_id(configured)
                changed = True
        default_provider = os.environ.get("AGENT_PROVIDER", "").strip().casefold()
        if default_provider in {provider.value for provider in Provider}:
            self.state["provider"] = default_provider
            changed = True
        active_provider = Provider(str(self.state["provider"]))
        if active_provider.value not in sessions:
            latest = next(iter(self.sessions(active_provider)), None)
            if latest:
                sessions[active_provider.value] = latest.session_id
                changed = True
        if changed:
            self.store.save(self.state)

    def sessions(self, provider: Provider | None = None) -> list[SessionRecord]:
        return discover_sessions(
            provider or self.provider, self.config.project_directory
        )

    def switch_provider(self, provider: Provider) -> SessionRecord | None:
        self.state["provider"] = provider.value
        latest = next(iter(self.sessions(provider)), None)
        if latest:
            self._bind(latest)
        else:
            self.store.save(self.state)
        return latest

    def takeover(self) -> SessionRecord:
        latest = next(iter(self.sessions()), None)
        if not latest:
            raise BridgeError(
                f"Keine lokale {self.provider.value}-Session für "
                f"{self.config.project_directory} gefunden."
            )
        self._bind(latest)
        return latest

    def clear_session(self) -> None:
        sessions = self.state["sessions"]
        assert isinstance(sessions, dict)
        sessions.pop(self.provider.value, None)
        self.store.save(self.state)

    def bind_session_id(self, session_id: str) -> SessionRecord:
        validated = validate_session_id(session_id)
        match = next(
            (record for record in self.sessions() if record.session_id == validated),
            None,
        )
        if not match:
            raise BridgeError(
                "Diese Session wurde für den aktiven Provider und das Projekt "
                "nicht gefunden."
            )
        self._bind(match)
        return match

    def _bind(self, record: SessionRecord) -> None:
        sessions = self.state["sessions"]
        assert isinstance(sessions, dict)
        sessions[record.provider.value] = record.session_id
        self.store.save(self.state)

    def bound_record(self) -> SessionRecord | None:
        session_id = self.session_id()
        if not session_id:
            return None
        return next(
            (record for record in self.sessions() if record.session_id == session_id),
            None,
        )

    def session_is_locally_active(self, record: SessionRecord) -> bool:
        return session_is_active(
            record,
            ignored_process_groups=self.bridge.active_process_groups,
        )

    async def ask(self, text: str, reply_observer=None):
        record = self.bound_record()
        if record and self.session_is_locally_active(record):
            raise BridgeError(
                "Diese Session ist noch im lokalen Terminal geöffnet. Beende dort "
                "Codex/Claude; die nächste Telegram-Nachricht setzt exakt hier fort."
            )
        reply = await self.bridge.ask(
            self.provider,
            text,
            self.session_id(),
            reply_observer=reply_observer,
            codex_transcript_path=(
                record.transcript_path
                if record and self.provider is Provider.CODEX
                else None
            ),
        )
        sessions = self.state["sessions"]
        assert isinstance(sessions, dict)
        sessions[self.provider.value] = reply.session_id
        self.store.save(self.state)
        return reply


runtime: BotRuntime | None = None


def _runtime() -> BotRuntime:
    if runtime is None:
        raise RuntimeError("Bot runtime is not initialized")
    return runtime


def _authorized(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id in _runtime().config.allowed_user_ids)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        await update.effective_message.reply_text("Nicht autorisiert.")
        return
    await update.effective_message.reply_text(
        "Bereit. Schreib normal oder sag: ‚Wechsel zu Codex‘, "
        "‚Wechsel zu Claude‘ oder ‚Übernimm die Session‘."
    )


async def cmd_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not context.args:
        await update.effective_message.reply_text(
            f"Aktiver Provider: {_runtime().provider.value}"
        )
        return
    try:
        provider = Provider(context.args[0].casefold())
        await _cancel_active_task(update.effective_user.id)
        record = _runtime().switch_provider(provider)
        await update.effective_message.reply_text(_switch_reply(provider, record))
    except (BridgeError, ValueError) as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")


async def cmd_takeover(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    try:
        await _cancel_active_task(update.effective_user.id)
        record = _runtime().takeover()
        await update.effective_message.reply_text(
            _takeover_reply(record, _runtime().session_is_locally_active(record))
        )
    except BridgeError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    stopped = await _cancel_active_task(update.effective_user.id)
    _runtime().clear_session()
    prefix = "Laufender Auftrag gestoppt. " if stopped else ""
    await update.effective_message.reply_text(
        f"{prefix}Die nächste Nachricht startet eine neue "
        f"{_runtime().provider.value}-Session."
    )


async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    records = _runtime().sessions()[:8]
    _runtime().session_choices[update.effective_user.id] = records
    if not records:
        await update.effective_message.reply_text("Keine passende lokale Session gefunden.")
        return
    lines = [f"{index}. {record.compact_label()}" for index, record in enumerate(records, 1)]
    await update.effective_message.reply_text(
        "\n".join(lines) + "\n\nWechsel mit /session <Nummer>."
    )


async def cmd_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not context.args:
        await update.effective_message.reply_text(
            "Nutze /sessions oder /session <UUID>."
        )
        return
    value = context.args[0]
    try:
        await _cancel_active_task(update.effective_user.id)
        if value.isdigit():
            records = _runtime().session_choices.get(update.effective_user.id, [])
            index = int(value) - 1
            if index < 0 or index >= len(records):
                raise BridgeError("Diese Sitzungsnummer ist nicht in der letzten Liste.")
            record = records[index]
            _runtime()._bind(record)
        else:
            record = _runtime().bind_session_id(value)
        await update.effective_message.reply_text(
            _takeover_reply(record, _runtime().session_is_locally_active(record))
        )
    except BridgeError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    provider = _runtime().provider
    session_id = _runtime().session_id()
    active = update.effective_user.id in _runtime().active_tasks
    record = _runtime().bound_record()
    locally_open = bool(record and _runtime().session_is_locally_active(record))
    session_label = f"{session_id[:8]}…" if session_id else "neu"
    if active:
        agent_status = "arbeitet per Telegram"
    elif locally_open:
        agent_status = "lokal geöffnet — CLI vor Telegram-Weiterarbeit beenden"
    else:
        agent_status = "bereit"
    await update.effective_message.reply_text(
        f"Provider: {provider.value}\nSession: {session_label}\n"
        f"Agent: {agent_status}"
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    task = _runtime().active_tasks.get(update.effective_user.id)
    if not task:
        await update.effective_message.reply_text("Kein laufender Auftrag.")
        return
    await _cancel_active_task(update.effective_user.id)
    await update.effective_message.reply_text("Laufender Auftrag ist gestoppt.")


async def cmd_claude_remote(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    del context
    if not _authorized(update):
        return
    try:
        await _start_claude_remote_control()
        await update.effective_message.reply_text(
            "Claude Remote Control ist online. Öffne https://claude.ai/code "
            "mit demselben Claude-Konto und wähle „Atelier Observer“."
        )
    except BridgeError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")


async def cmd_claude_remote_stop(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    del context
    if not _authorized(update):
        return
    await _run_systemctl("stop", "claude-remote-control.service")
    await update.effective_message.reply_text("Claude Remote Control ist gestoppt.")


async def cmd_player(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    environment_file = _runtime().config.project_directory / ".server.env"
    if not environment_file.exists():
        await update.effective_message.reply_text("Kein Player für dieses Projekt gefunden.")
        return
    values: dict[str, str] = {}
    for line in environment_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    url = values.get("SONGMAKER_URL")
    await update.effective_message.reply_text(url or "Keine SONGMAKER_URL gefunden.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not _authorized(update):
        return
    message = update.effective_message
    text = message.text or ""
    if not text:
        return
    if len(text) > MAX_MESSAGE_LENGTH:
        await message.reply_text("Die Nachricht ist für den Bot zu lang.")
        return

    intent = parse_control_intent(text)
    if intent:
        await _handle_control_intent(message, intent)
        return

    user_id = update.effective_user.id
    existing = _runtime().active_tasks.get(user_id)
    if existing and not existing.done():
        await message.reply_text(
            "Ich arbeite noch. Mit /stop abbrechen oder danach erneut senden."
        )
        return
    record = _runtime().bound_record()
    if record and _runtime().session_is_locally_active(record):
        await message.reply_text(
            "Diese Session ist noch im lokalen Terminal geöffnet. Beende dort "
            "Codex/Claude vollständig und sende die Nachricht danach erneut."
        )
        return
    provider_label = _runtime().provider.value.capitalize()
    await message.reply_text(
        f"✅ Empfangen. {provider_label} arbeitet jetzt; "
        "mit /status prüfen oder mit /stop abbrechen."
    )
    task = asyncio.create_task(_run_agent_task(user_id, message, text))
    _runtime().active_tasks[user_id] = task


async def _handle_control_intent(message, intent: ControlIntent) -> None:
    try:
        user_id = message.from_user.id
        if intent.action == "switch" and intent.provider:
            await _cancel_active_task(user_id)
            record = _runtime().switch_provider(intent.provider)
            await message.reply_text(_switch_reply(intent.provider, record))
        elif intent.action == "takeover":
            await _cancel_active_task(user_id)
            record = _runtime().takeover()
            await message.reply_text(
                _takeover_reply(record, _runtime().session_is_locally_active(record))
            )
        elif intent.action == "new":
            stopped = await _cancel_active_task(user_id)
            _runtime().clear_session()
            prefix = "Laufender Auftrag gestoppt. " if stopped else ""
            await message.reply_text(
                f"{prefix}Die nächste Nachricht startet eine neue "
                f"{_runtime().provider.value}-Session."
            )
        elif intent.action == "claude_remote":
            await _start_claude_remote_control()
            await message.reply_text(
                "Claude Remote Control ist online. Öffne https://claude.ai/code "
                "und wähle „Atelier Observer“."
            )
    except BridgeError as exc:
        await message.reply_text(f"⚠️ {exc}")


async def _cancel_active_task(user_id: int) -> bool:
    """Stop one bridge-owned turn and wait until its process group is gone."""
    task = _runtime().active_tasks.get(user_id)
    if task is None or task.done():
        _runtime().active_tasks.pop(user_id, None)
        return False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    _runtime().active_tasks.pop(user_id, None)
    return True


async def _run_agent_task(user_id: int, message, text: str) -> None:
    typing_task = asyncio.create_task(_keep_typing(message.chat))
    try:
        async def deliver_reply(reply: str) -> None:
            await _send_reply(message, reply)

        reply = await _runtime().ask(text, deliver_reply)
        if not reply.delivered_during_run:
            await _send_reply(message, reply.text)
    except asyncio.CancelledError:
        pass
    except BridgeError as exc:
        log.warning("Agent request refused: %s", exc)
        await message.reply_text(f"⚠️ {exc}")
    except Exception:
        log.exception("Unexpected agent bridge failure")
        await message.reply_text("⚠️ Unerwarteter interner Bot-Fehler.")
    finally:
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)
        _runtime().active_tasks.pop(user_id, None)


async def _keep_typing(chat) -> None:
    try:
        while True:
            await chat.send_action("typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def _start_claude_remote_control() -> None:
    await _run_systemctl("start", "claude-remote-control.service")
    for _ in range(30):
        process = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "is-active",
            "--quiet",
            "claude-remote-control.service",
        )
        if await process.wait() == 0:
            return
        await asyncio.sleep(0.2)
    raise BridgeError("Claude Remote Control wurde nicht aktiv.")


async def _run_systemctl(action: str, unit: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        action,
        unit,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    if process.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace")
        compact = " ".join(detail.split())[:300]
        raise BridgeError(f"Claude Remote Control konnte nicht starten: {compact}")


async def _send_reply(message, reply: str) -> None:
    for start in range(0, len(reply), MAX_REPLY_CHUNK):
        await message.reply_text(reply[start : start + MAX_REPLY_CHUNK])
    for audio_path in _find_audio_files(reply, _runtime().config.project_directory):
        await message.chat.send_action("upload_document")
        with audio_path.open("rb") as stream:
            await message.reply_audio(
                audio=stream,
                title=audio_path.stem,
                filename=audio_path.name,
            )


def _find_audio_files(text: str, project_directory: Path) -> list[Path]:
    matches = re.findall(
        r"(?:/[\w._-]+)+\.(?:mp3|wav)|(?:[\w._-]+/)*[\w._-]+\.(?:mp3|wav)",
        text,
    )
    project = project_directory.resolve()
    found: list[Path] = []
    for raw_path in matches:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = project / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(project)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.stat().st_size <= MAX_AUDIO_BYTES:
            found.append(resolved)
    return found


def _switch_reply(provider: Provider, record: SessionRecord | None) -> str:
    if not record:
        return f"{provider.value.capitalize()} ist aktiv; die nächste Nachricht startet neu."
    return (
        f"{provider.value.capitalize()} ist aktiv. Passende letzte Session gebunden: "
        f"{record.compact_label()}."
    )


def _takeover_reply(record: SessionRecord, locally_open: bool) -> str:
    suffix = (
        " Sie ist noch lokal geöffnet; beende dort zuerst die CLI."
        if locally_open
        else " Die nächste Nachricht setzt sie fort."
    )
    return f"Session übernommen: {record.compact_label()}.{suffix}"


def main() -> None:
    global runtime
    try:
        config = load_config()
        runtime = BotRuntime(config)
    except BridgeError as exc:
        raise SystemExit(str(exc)) from exc

    log.info(
        "Starting private agent bridge: provider=%s project=%s",
        runtime.provider.value,
        config.project_directory,
    )
    application = ApplicationBuilder().token(config.telegram_token).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("provider", cmd_provider))
    application.add_handler(CommandHandler("takeover", cmd_takeover))
    application.add_handler(CommandHandler("new", cmd_new))
    application.add_handler(CommandHandler("clear", cmd_new))
    application.add_handler(CommandHandler("sessions", cmd_sessions))
    application.add_handler(CommandHandler("session", cmd_session))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("claude_remote", cmd_claude_remote))
    application.add_handler(
        CommandHandler("claude_remote_stop", cmd_claude_remote_stop)
    )
    application.add_handler(CommandHandler("player", cmd_player))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
