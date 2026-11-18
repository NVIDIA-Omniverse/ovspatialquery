# Consumer skills

Agent skills for people *using* ovspatialquery, in the open
[Agent Skills](https://agentskills.io) format: a folder with a `SKILL.md` (standard
frontmatter only), `references/` loaded on demand, `scripts/` to copy, and `assets/`
holding complete programs. They work in any harness that reads that format.

`skills/` is the canonical source of these skills; the copies in the SDK and the wheel are
built from it.

| Skill | Covers |
| --- | --- |
| [`ovspatialquery`](ovspatialquery/SKILL.md) | Raycasts, sweeps, overlaps, batched scans, pair tracking, time-sampled USD, explicit shapes; Python and C/C++ |

Packaged skills follow the library version. SDK and wheel builds copy the skill folder
unchanged, so a signed skill still verifies, and record the artifact version and the
library's source commit (with `-dirty` for local changes) in the package's `README.txt`.
The source skill has no separate version.

## Install

Copy the skill folder where your harness looks for skills:

| Harness | Project scope | User scope |
| --- | --- | --- |
| Claude Code | `.claude/skills/ovspatialquery/` | `~/.claude/skills/ovspatialquery/` |
| Codex | `.agents/skills/ovspatialquery/` | `~/.codex/skills/ovspatialquery/` |
| Cursor and others reading `.agents/skills` | `.agents/skills/ovspatialquery/` | per harness |

A harness without skill support can still use it: point the agent at `SKILL.md`, for
example from `AGENTS.md` ("For spatial queries, read `<path>/ovspatialquery/SKILL.md`").
`evals/` is for maintainers; keep it if you verify the skill's signature, as the user guide's
Coding Agents page describes.

