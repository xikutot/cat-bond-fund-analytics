"""Validate an AGENTS.md report, write human-readable JSON proof, and print the stamp.

The accepted input is exactly the JSON printed by --example, with each
placeholder reason and friction field replaced. The generator supplies time;
--from-proof verifies the saved stamp from proof.json without rewriting it.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPORT = json.loads("""
{
  "reasons": {
    "reader.path": "<reason>",
    "reader.first_read": "<reason>",
    "code.reuse": "<reason>",
    "code.dependency": "<reason>",
    "code.sprawl": "<reason>",
    "code.concern": "<reason>",
    "code.locality": "<reason>",
    "change.bounds": "<reason>",
    "change.ground": "<reason>",
    "change.verify": "<reason>",
    "change.ide": "<reason>",
    "collaboration.correction": "<reason>",
    "collaboration.friction": "<reason>",
    "collaboration.reporting": "<reason>",
    "collaboration.completion": "<reason>"
  },
  "friction": [
    {"class": "<rule|gap>", "trigger": "<trigger>", "update": "<smallest useful update>"}
  ]
}
""")
KEYS = set(REPORT)
PROOF_KEYS = KEYS | {"time"}
IDS = list(REPORT["reasons"])
ID_SET = set(IDS)
PLACEHOLDER = REPORT["reasons"][IDS[0]]
FRICTION_EXAMPLE = REPORT["friction"][0]
FRICTION_FIELDS = set(FRICTION_EXAMPLE)
FRICTION_CLASSES = {"rule", "gap"}
GROUP_COUNT, GROUP_WIDTH = 2, 3
RECEIPT_CHARS = GROUP_COUNT * GROUP_WIDTH
PROOF = Path("/home/ws/.opencode/plan/proof.json")


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def one_line(value, message, placeholder=None):
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise SystemExit(message)
    if placeholder is not None and value.strip() == placeholder:
        raise SystemExit(message)
    return value.strip()


def validate(report, keys=KEYS):
    if not isinstance(report, dict) or set(report) != keys:
        raise SystemExit(f"report needs {', '.join(sorted(keys))}")

    if "time" in report:
        one_line(report["time"], "time must be one line")

    reasons = report["reasons"]
    if not isinstance(reasons, dict) or set(reasons) != ID_SET:
        raise SystemExit("reasons must cover every AGENTS.md bullet")
    for value in reasons.values():
        one_line(
            value,
            "replace every placeholder with a one-line reason",
            PLACEHOLDER,
        )

    friction = report["friction"]
    if not isinstance(friction, list):
        raise SystemExit("friction must be a list of objects")
    for item in friction:
        if not isinstance(item, dict) or set(item) != FRICTION_FIELDS:
            raise SystemExit("friction entries need class, trigger, update")
        if item["class"] not in FRICTION_CLASSES:
            raise SystemExit("friction class must be rule or gap")
        for field in ("trigger", "update"):
            one_line(
                item[field],
                "replace every friction placeholder",
                FRICTION_EXAMPLE[field],
            )


def friction_lines(report):
    return [
        f"{item['class']}: {item['trigger']}; smallest useful update: {item['update']}"
        for item in report["friction"]
    ]


def render(report):
    validate(report)
    proof = {"time": timestamp(), **report}
    return json.dumps(proof, indent=2) + "\n", friction_lines(proof)


def load_proof():
    text = PROOF.read_text()
    report = json.loads(text)
    validate(report, PROOF_KEYS)
    return text, friction_lines(report)


def reuses_reasons(report):
    if not PROOF.exists():
        return False
    previous = json.loads(PROOF.read_text())
    return (
        isinstance(previous, dict)
        and previous.get("reasons") == report.get("reasons")
    )


def receipt(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:RECEIPT_CHARS]
    return "-".join(
        digest[i : i + GROUP_WIDTH]
        for i in range(0, RECEIPT_CHARS, GROUP_WIDTH)
    )


def main():
    flags = set(sys.argv[1:])
    if "--example" in flags:
        print(json.dumps(REPORT, indent=2))
        return
    from_proof = "--from-proof" in flags
    if from_proof:
        text, friction = load_proof()
    else:
        report = json.load(sys.stdin)
        if reuses_reasons(report):
            raise SystemExit(
                "reasons match the existing proof; reason from the current change instead of copying"
            )
        text, friction = render(report)
        PROOF.parent.mkdir(parents=True, exist_ok=True)
        PROOF.write_text(text)

    stamp = "AGENTS.md check: " + ("FRICTION" if friction else "PASS")
    lines = ["---", stamp, f"receipt: {receipt(text)}"]
    lines += [f"- {item}" for item in friction]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
