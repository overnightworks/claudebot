from __future__ import annotations

import base64
import http.server
import json
import socket
import threading
import urllib.error
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import sonar_findings_gate as gate


PROJECT_KEY = "overnightworks_demo"
STUB_CREDENTIAL = "stub-credential"
DEFAULT_ENVIRONMENT = {
    gate.EVENT_NAME_VARIABLE: "push",
    gate.BRANCH_NAME_VARIABLE: "main",
    gate.CREDENTIAL_VARIABLE: STUB_CREDENTIAL,
}
MEASURES_ENDPOINT = gate.MEASURES_ENDPOINT
ISSUES_ENDPOINT = gate.FINDINGS_ENDPOINT
# A handler that never answers releases at teardown; the bound only keeps a
# failing test from hanging the suite.
UNANSWERED_REQUEST_LIMIT_SECONDS = 30
PATCHED_REQUEST_TIMEOUT_SECONDS = 0.25


Responder = Callable[[http.server.BaseHTTPRequestHandler], None]


@dataclass(frozen=True)
class RecordedRequest:
    endpoint: str
    query: dict[str, list[str]]
    raw_query: str
    authorization: str | None


def json_body(payload: object) -> Responder:
    def respond(handler: http.server.BaseHTTPRequestHandler) -> None:
        write_body(handler, 200, json.dumps(payload), "application/json")

    return respond


def text_body(text: str, status: int = 200) -> Responder:
    def respond(handler: http.server.BaseHTTPRequestHandler) -> None:
        write_body(handler, status, text, "text/html")

    return respond


def redirect_to(location: str) -> Responder:
    def respond(handler: http.server.BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", location)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    return respond


def silence(released: threading.Event) -> Responder:
    def respond(_: http.server.BaseHTTPRequestHandler) -> None:
        released.wait(UNANSWERED_REQUEST_LIMIT_SECONDS)

    return respond


def write_body(
    handler: http.server.BaseHTTPRequestHandler, status: int, body: str, content_type: str
) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


class RequestRecordingHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - the name http.server dispatches to
        self.server.stub.answer(self)

    def log_message(self, *args: object) -> None:
        """Keep the stand-in silent; the recorded requests are the evidence."""


@dataclass
class SonarCloudStub:
    """A loopback stand-in for the SonarCloud endpoints the gate reads."""

    responses: dict[str, list[Responder]] = field(default_factory=dict)
    requests: list[RecordedRequest] = field(default_factory=list)
    released: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RequestRecordingHandler)
        self._server.daemon_threads = True
        self._server.stub = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def api_root(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/api"

    def answers(self, endpoint: str, *responders: Responder) -> None:
        self.responses[endpoint] = list(responders)

    def never_answers(self, endpoint: str) -> None:
        self.answers(endpoint, silence(self.released))

    def endpoints_requested(self) -> list[str]:
        return [request.endpoint for request in self.requests]

    def last_request_to(self, endpoint: str) -> RecordedRequest:
        return [request for request in self.requests if request.endpoint == endpoint][-1]

    def answer(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        endpoint = parsed.path.removeprefix("/api/")
        self.requests.append(
            RecordedRequest(
                endpoint=endpoint,
                query=parse_qs(parsed.query),
                raw_query=parsed.query,
                authorization=handler.headers.get("Authorization"),
            )
        )
        self.next_responder(endpoint)(handler)

    def next_responder(self, endpoint: str) -> Responder:
        queued = self.responses.get(endpoint)
        if not queued:
            return text_body(f"no response queued for {endpoint}", status=500)
        return queued.pop(0) if len(queued) > 1 else queued[0]

    def close(self) -> None:
        self.released.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


def issue_payload(
    *,
    rule: str = "python:S1481",
    file_path: str = "scripts/example.py",
    line: int | None = 12,
    message: str = "Remove this unused local variable.",
) -> dict[str, object]:
    payload = {"rule": rule, "component": f"{PROJECT_KEY}:{file_path}", "message": message}
    if line is not None:
        payload["line"] = line
    return payload


def issues_page(*issues: dict[str, object], total: int | None = None) -> Responder:
    return json_body({"total": len(issues) if total is None else total, "issues": list(issues)})


@pytest.fixture
def sonar_cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SonarCloudStub]:
    properties_path = tmp_path / "sonar-project.properties"
    properties_path.write_text(
        f"sonar.organization=overnightworks\n{gate.PROJECT_KEY_PROPERTY}={PROJECT_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "SONAR_PROPERTIES_PATH", properties_path)
    stub = SonarCloudStub()
    monkeypatch.setattr(gate, "SONAR_API_ROOT", stub.api_root)
    stub.answers(MEASURES_ENDPOINT, json_body({"component": {"key": PROJECT_KEY}}))
    stub.answers(ISSUES_ENDPOINT, issues_page())
    try:
        yield stub
    finally:
        stub.close()


def run_gate(monkeypatch: pytest.MonkeyPatch, **environment: str) -> int:
    for name, value in {**DEFAULT_ENVIRONMENT, **environment}.items():
        monkeypatch.setenv(name, value)
    return gate.main()


def test_reports_a_scope_that_resolves_and_holds_no_findings_as_clean(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run_gate(monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"SonarCloud clean for '{PROJECT_KEY}' (branch=main): 0 open finding(s)\n"
    assert captured.err == ""


def test_lists_every_open_finding_and_fails_the_build(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sonar_cloud.answers(
        ISSUES_ENDPOINT,
        issues_page(
            issue_payload(),
            issue_payload(rule="python:S5754", file_path="scripts/other.py", line=None, message="Specify an exception class."),
        ),
    )

    exit_code = run_gate(monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.splitlines() == [
        "python:S1481 scripts/example.py:12 Remove this unused local variable.",
        "python:S5754 scripts/other.py:? Specify an exception class.",
    ]
    assert captured.err.strip() == "2 open SonarCloud finding(s) in scope branch=main"


def test_reads_every_page_of_a_paginated_result(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sonar_cloud.answers(
        ISSUES_ENDPOINT,
        issues_page(issue_payload(file_path="scripts/first.py"), total=2),
        issues_page(issue_payload(file_path="scripts/second.py"), total=2),
    )

    exit_code = run_gate(monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert [line.split()[1] for line in captured.out.splitlines()] == [
        "scripts/first.py:12",
        "scripts/second.py:12",
    ]
    assert [
        request.query["p"] for request in sonar_cloud.requests if request.endpoint == ISSUES_ENDPOINT
    ] == [["1"], ["2"]]


def test_stops_when_a_page_reports_more_findings_than_it_returns(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sonar_cloud.answers(
        ISSUES_ENDPOINT,
        issues_page(issue_payload(), total=7),
        issues_page(total=7),
    )

    exit_code = run_gate(monkeypatch)

    assert exit_code == 1
    assert capsys.readouterr().err.strip() == "7 open SonarCloud finding(s) in scope branch=main"


def test_reports_findings_an_answer_lists_while_claiming_a_total_of_zero(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sonar_cloud.answers(ISSUES_ENDPOINT, issues_page(issue_payload(), total=0))

    exit_code = run_gate(monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.splitlines() == [
        "python:S1481 scripts/example.py:12 Remove this unused local variable."
    ]
    assert captured.err.strip() == "1 open SonarCloud finding(s) in scope branch=main"


def test_stops_when_a_finding_names_no_file_within_the_project(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sonar_cloud.answers(
        ISSUES_ENDPOINT,
        issues_page({"rule": "python:S1481", "component": "no-project-key", "message": "Unused."}),
    )

    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch)

    assert "'no-project-key'" in str(failure.value)
    assert "SonarCloud clean" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(TimeoutError(), True, id="read_ran_out_of_time"),
        pytest.param(urllib.error.URLError(TimeoutError()), True, id="connect_ran_out_of_time"),
        pytest.param(urllib.error.URLError(ConnectionRefusedError()), False, id="connection_refused"),
        pytest.param(OSError("broken pipe"), False, id="other_transport_failure"),
    ],
)
def test_recognises_which_transport_failures_ran_out_of_time(
    error: OSError, expected: bool
) -> None:
    assert gate.timed_out(error) is expected


def test_stops_at_the_page_limit_when_the_service_keeps_claiming_more(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gate, "MAXIMUM_FINDINGS_PAGES", 2)
    sonar_cloud.answers(ISSUES_ENDPOINT, issues_page(issue_payload(), total=9_999))

    exit_code = run_gate(monkeypatch)

    assert exit_code == 1
    assert sonar_cloud.endpoints_requested().count(ISSUES_ENDPOINT) == 2
    assert capsys.readouterr().err.strip() == "9999 open SonarCloud finding(s) in scope branch=main"


@pytest.mark.parametrize(
    ("environment", "expected_scope"),
    [
        pytest.param({gate.BRANCH_NAME_VARIABLE: "release/2%off & more"}, "branch=release/2%off & more", id="branch"),
        pytest.param(
            {gate.EVENT_NAME_VARIABLE: "pull_request", gate.PULL_REQUEST_NUMBER_VARIABLE: "41"},
            "pullRequest=41",
            id="pull_request",
        ),
    ],
)
def test_asks_about_the_scope_of_the_run_with_its_value_encoded(
    sonar_cloud: SonarCloudStub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment: dict[str, str],
    expected_scope: str,
) -> None:
    exit_code = run_gate(monkeypatch, **environment)

    parameter_name, _, value = expected_scope.partition("=")
    assert exit_code == 0
    assert capsys.readouterr().out.strip().endswith(f"({expected_scope}): 0 open finding(s)")
    for endpoint in (MEASURES_ENDPOINT, ISSUES_ENDPOINT):
        assert sonar_cloud.last_request_to(endpoint).query[parameter_name] == [value]


def test_sends_the_credential_in_a_header_and_never_in_the_query(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_gate(monkeypatch)
    capsys.readouterr()

    for request in sonar_cloud.requests:
        scheme, _, encoded = request.authorization.partition(" ")
        assert scheme == "Basic"
        assert base64.b64decode(encoded).decode() == f"{STUB_CREDENTIAL}:"
        assert STUB_CREDENTIAL not in request.raw_query


def test_stops_when_the_scope_does_not_resolve(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    sonar_cloud.answers(MEASURES_ENDPOINT, text_body("not found", status=404))

    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch)

    assert f"SonarCloud did not resolve {PROJECT_KEY} in scope branch=main (HTTP 404)" in str(failure.value)
    assert "Its findings were never read." in str(failure.value)
    assert sonar_cloud.endpoints_requested() == [MEASURES_ENDPOINT]


def test_stops_before_asking_when_the_credential_is_empty(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch, **{gate.CREDENTIAL_VARIABLE: ""})

    assert "SONAR_TOKEN is empty or unset" in str(failure.value)
    assert sonar_cloud.requests == []


def test_stops_when_the_answer_is_not_json(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    sonar_cloud.answers(MEASURES_ENDPOINT, text_body("<html>proxy error</html>"))

    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch)

    assert (
        f"SonarCloud answered the {MEASURES_ENDPOINT} request for {PROJECT_KEY} in scope "
        "branch=main with a body that is not JSON." in str(failure.value)
    )


def test_refuses_a_redirect_instead_of_forwarding_the_credential(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    sonar_cloud.answers(MEASURES_ENDPOINT, redirect_to(f"{sonar_cloud.api_root}/elsewhere"))

    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch)

    assert "(HTTP 302)" in str(failure.value)
    assert sonar_cloud.endpoints_requested() == [MEASURES_ENDPOINT]


def test_stops_when_sonarcloud_accepts_the_call_and_never_answers(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "REQUEST_TIMEOUT_SECONDS", PATCHED_REQUEST_TIMEOUT_SECONDS)
    sonar_cloud.never_answers(MEASURES_ENDPOINT)

    with pytest.raises(SystemExit) as failure:
        run_gate(monkeypatch)

    assert (
        f"SonarCloud did not answer within {PATCHED_REQUEST_TIMEOUT_SECONDS}s: the request for "
        f"{PROJECT_KEY} in scope branch=main timed out." in str(failure.value)
    )
    assert "re-run the job before investigating SonarCloud." in str(failure.value)


def test_fails_loudly_when_sonarcloud_cannot_be_reached(
    sonar_cloud: SonarCloudStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "SONAR_API_ROOT", f"http://127.0.0.1:{unused_port()}/api")

    with pytest.raises(OSError, match="Connection refused"):
        run_gate(monkeypatch)


def unused_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_reads_the_project_key_from_the_properties_file_of_this_repository() -> None:
    assert gate.read_project_key(gate.SONAR_PROPERTIES_PATH)


def test_stops_when_the_properties_file_names_no_project(tmp_path: Path) -> None:
    properties_path = tmp_path / "sonar-project.properties"
    properties_path.write_text("sonar.organization=overnightworks\n", encoding="utf-8")

    with pytest.raises(SystemExit) as failure:
        gate.read_project_key(properties_path)

    assert f"does not define {gate.PROJECT_KEY_PROPERTY}." in str(failure.value)
