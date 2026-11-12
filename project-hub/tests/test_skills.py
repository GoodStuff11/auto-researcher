import argparse
import re
from pathlib import Path

from hub.cli import build_parser

SKILLS = Path(__file__).resolve().parents[1] / "skills"
CMD_RE = re.compile(r"`hub ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?")


def _choices(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def test_every_cited_hub_command_exists():
    files = sorted(SKILLS.rglob("*.md"))
    assert files, "no skill files found"
    top = _choices(build_parser())
    bad = []
    for f in files:
        for m in CMD_RE.finditer(f.read_text(encoding="utf-8")):
            cmd, sub = m[1], m[2]
            if cmd not in top:
                bad.append(f"{f.relative_to(SKILLS)}: hub {cmd}")
                continue
            nested = _choices(top[cmd])
            if nested and sub is not None and sub not in nested:
                bad.append(f"{f.relative_to(SKILLS)}: hub {cmd} {sub}")
    assert bad == []


def test_skill_frontmatter():
    skills = sorted(SKILLS.glob("*/SKILL.md"))
    assert skills
    for f in skills:
        text = f.read_text(encoding="utf-8")
        head = text.split("\n---\n", 1)[0]
        assert text.startswith(f"---\nname: {f.parent.name}\n"), f
        assert "\ndescription: Use when" in head, f
