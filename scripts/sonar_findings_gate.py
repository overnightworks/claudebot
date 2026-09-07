#!/usr/bin/env python3
"""Fail the build when SonarCloud holds open findings for the analysed scope."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SONAR_PROPERTIES_PATH = REPOSITORY_ROOT / "sonar-project.properties"
PROJECT_KEY_PROPERTY = "sonar.projectKey"

SONAR_API_ROOT = "https://sonarcloud.io/api"
MEASURES_ENDPOINT = "measures/component"
FINDINGS_ENDPOINT = "issues/search"
# A SonarCloud that accepts the connection and then stops answering must fail
# the job. This bounds one call; the workflow step's own ceiling bounds the
# paging above it.
REQUEST_TIMEOUT_SECONDS = 30
FINDINGS_PAGE_SIZE = 100
# `issues/search` refuses a page past its ten-thousandth issue, so the paging
# is bounded by that limit and never by the total an answer claims.
FINDINGS_SEARCH_LIMIT = 10_000
MAXIMUM_FINDINGS_PAGES = FINDINGS_SEARCH_LIMIT // FINDINGS_PAGE_SIZE
# `issueStatuses`, never the legacy `statuses`, whose vocabulary drops an issue
# a person accepts or marks false positive in the SonarCloud interface.
OPEN_ISSUE_STATUSES = "OPEN,CONFIRMED,ACCEPTED,FALSE_POSITIVE"
UNKNOWN_LINE = "?"

PULL_REQUEST_EVENT = "pull_request"
EVENT_NAME_VARIABLE = "GITHUB_EVENT_NAME"
BRANCH_NAME_VARIABLE = "GITHUB_REF_NAME"
PULL_REQUEST_NUMBER_VARIABLE = "PULL_REQUEST_NUMBER"
CREDENTIAL_VARIABLE = "SONAR_TOKEN"


@dataclass(frozen=True)
class AnalysisScope:
    """The single branch or pull request analysis this run asks about."""

    parameter_name: str
    value: str

    @property
    def label(self) -> str:
        return f"{self.parameter_name}={self.value}"

    @property
    def query_parameters(self) -> dict[str, str]:
        return {self.parameter_name: self.value}


@dataclass(frozen=True)
class Finding:
    rule: str
    file_path: str
    line: int | None
    message: str

    def as_report_line(self) -> str:
        line = UNKNOWN_LINE if self.line is None else self.line
        return f"{self.rule} {self.file_path}:{line} {self.message}"


@dataclass(frozen=True)
class OpenFindings:
    """What the issue search reported: its own total, and the findings read."""

    total: int
    listed: tuple[Finding, ...]

    @property
    def count(self) -> int:
        # A total the same answer contradicts by listing findings is not a zero.
        return max(self.total, len(self.listed))


class RedirectRefusingHandler(urllib.request.HTTPRedirectHandler):
    # A redirect would forward the Authorization header to another host; refuse
    # it instead of following it.
    def redirect_request(self, *args, **kwargs) -> None:
        return None


class SonarCloudProject:
    """One SonarCloud project, read through one analysis scope over HTTPS."""

    def __init__(self, component_key: str, scope: AnalysisScope, credential: str) -> None:
        self.component_key = component_key
        self.scope = scope
        self._authorization = basic_authorization(credential)
        self._opener = urllib.request.build_opener(RedirectRefusingHandler())

    def require_resolved(self) -> None:
        # An unauthenticated, misspelled or never-analysed scope answers the
        # issue search with zero findings and no error, so read the measures of
        # this exact scope first: it is 404 when the credential, the key or the
        # scope does not resolve.
        try:
            self.read(MEASURES_ENDPOINT, component=self.component_key, metricKeys="ncloc")
        except urllib.error.HTTPError as error:
            raise SystemExit(
                f"SonarCloud did not resolve {self.component_key} in scope "
                f"{self.scope.label} (HTTP {error.code}): either no analysis "
                "exists for this scope — an unanalysed branch, a pull "
                "request that does not exist, a mistyped key — or the "
                "credential cannot see the project. Its findings were "
                "never read."
            ) from error

    def open_findings(self) -> OpenFindings:
        findings: list[Finding] = []
        reported_total = 0
        for page in range(1, MAXIMUM_FINDINGS_PAGES + 1):
            report = self.read(
                FINDINGS_ENDPOINT,
                componentKeys=self.component_key,
                issueStatuses=OPEN_ISSUE_STATUSES,
                ps=FINDINGS_PAGE_SIZE,
                p=page,
            )
            reported_total = report["total"]
            if not report["issues"]:
                break
            findings.extend(finding_from_issue(issue) for issue in report["issues"])
            if len(findings) >= reported_total:
                break
        return OpenFindings(total=reported_total, listed=tuple(findings))

    def read(self, endpoint: str, **parameters: object) -> dict:
        # The scope carries a branch name the runner derived, and a ref name may
        # legally contain characters a query string reads differently.
        query = urllib.parse.urlencode({**parameters, **self.scope.query_parameters})
        request = urllib.request.Request(
            f"{SONAR_API_ROOT}/{endpoint}?{query}",
            headers={"Authorization": self._authorization},
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except OSError as error:
            if not timed_out(error):
                raise
            raise SystemExit(
                f"SonarCloud did not answer within {REQUEST_TIMEOUT_SECONDS}s: "
                f"the request for {self.component_key} in scope {self.scope.label} timed "
                "out. Its findings were never read, so re-run the job before "
                "investigating SonarCloud."
            ) from error
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"SonarCloud answered the {endpoint} request for {self.component_key} "
                f"in scope {self.scope.label} with a body that is not JSON. Its "
                "findings were never read."
            ) from error


def timed_out(error: OSError) -> bool:
    # A read that runs out of time raises TimeoutError itself, while a connect
    # that does arrives wrapped in URLError.
    return isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)
    )


def finding_from_issue(issue: dict) -> Finding:
    component = issue["component"]
    _, separator, file_path = component.partition(":")
    if not separator:
        raise SystemExit(
            f"SonarCloud reported a finding whose component {component!r} carries no "
            "file path, so this answer could not be read. An answer that could not be "
            "read is not a clean one."
        )
    return Finding(
        rule=issue["rule"],
        file_path=file_path,
        line=issue.get("line"),
        message=issue["message"],
    )


def basic_authorization(credential: str) -> str:
    # The credential is the basic-auth user name with an empty password, and it
    # travels in a header so no command line or process listing can carry it.
    return "Basic " + base64.b64encode(f"{credential}:".encode()).decode()


def read_project_key(properties_path: Path) -> str:
    for line in properties_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == PROJECT_KEY_PROPERTY:
            return value.strip()
    raise SystemExit(f"{properties_path} does not define {PROJECT_KEY_PROPERTY}.")


def analysis_scope(environment: Mapping[str, str]) -> AnalysisScope:
    if environment[EVENT_NAME_VARIABLE] == PULL_REQUEST_EVENT:
        return AnalysisScope("pullRequest", environment[PULL_REQUEST_NUMBER_VARIABLE])
    return AnalysisScope("branch", environment[BRANCH_NAME_VARIABLE])


def require_credential(environment: Mapping[str, str]) -> str:
    credential = environment.get(CREDENTIAL_VARIABLE, "")
    if not credential:
        raise SystemExit(
            f"{CREDENTIAL_VARIABLE} is empty or unset, so SonarCloud would be asked "
            "as an anonymous reader, and a private project answers such a query "
            "with zero findings. Its findings were never read."
        )
    return credential


def main() -> int:
    project = SonarCloudProject(
        component_key=read_project_key(SONAR_PROPERTIES_PATH),
        scope=analysis_scope(os.environ),
        credential=require_credential(os.environ),
    )
    project.require_resolved()

    findings = project.open_findings()
    for finding in findings.listed:
        print(finding.as_report_line())
    if findings.count > 0:
        print(
            f"{findings.count} open SonarCloud finding(s) in scope {project.scope.label}",
            file=sys.stderr,
        )
        return 1

    # A step that prints nothing on success cannot be shown to have run.
    print(
        f"SonarCloud clean for '{project.component_key}' ({project.scope.label}): "
        f"{findings.count} open finding(s)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point, no logic of its own
    raise SystemExit(main())
