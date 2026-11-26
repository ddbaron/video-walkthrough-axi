"""Enforce a 999-code-line limit on tracked Python files."""

import os
import subprocess
import tokenize


def count_code_lines(path: str) -> int:
    with tokenize.open(path) as source:
        comment_lines = {
            token.start[0]
            for token in tokenize.generate_tokens(source.readline)
            if token.type == tokenize.COMMENT and not token.line[: token.start[1]].strip()
        }
        source.seek(0)
        return sum(
            bool(line.strip()) and row not in comment_lines
            for row, line in enumerate(source, 1)
        )


def main() -> int:
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"])
    failed = False
    for path in os.fsdecode(tracked).split("\0"):
        if path and (count := count_code_lines(path)) > 999:
            print(f"{path}: {count} code lines (limit: 999)")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
