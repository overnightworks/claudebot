"""Temporary probe for issue #10: proves the CI Python matrix has teeth.

`Path.resolve()` on a symlink loop raises on 3.12 and resolves lexically on
3.14, so this expectation is true on exactly one matrix entry. Reverted once
the matrix is shown to go red on 3.12 and stay green on 3.14.
"""

import os
from pathlib import Path


def test_resolves_a_symlink_loop_lexically(tmp_path: Path) -> None:
    loop = tmp_path / "loop"
    os.symlink(loop, loop)

    assert loop.resolve() == loop
