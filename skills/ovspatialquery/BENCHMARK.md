# Skill Benchmark: ovspatialquery

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `ovspatialquery`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 12 evaluation tasks (10 positive, 2 negative)
- Dataset digest: `sha256:29d96587d3f1b737f0cded6ac93a9f4aac71ae9cdb556e620fb66a64f28a8766` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| ovspatialquery | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ovspatialquery | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ovspatialquery | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ovspatialquery | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ovspatialquery | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `ovspatialquery`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 100.0% (+8.3 points) | 95.8% → 83.3% (-12.5 points) |
| Correctness | 41.7% → 100.0% (+58.3 points) | 40.0% → 90.0% (+50.0 points) |
| Discoverability | 96.5% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 18.8% → 73.0% (+54.2 points) | 24.3% → 67.5% (+43.2 points) |
| Efficiency | 86.0% — baseline ran, but no comparable score was available; uplift unavailable | 78.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.5% → 90.8% (-4.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.6% → 87.5% (+6.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,231,066 | 2,799,487 | +431,579 | +15.42% | skill 12/12; base 12/12 |
| claude-code | 1 | 442,706 | 351,710 | +90,996 | +25.87% | skill 1/1; base 1/1 |
| claude-code | 10 | 164,560 | 144,841 | +19,719 | +13.61% | skill 1/1; base 1/1 |
| claude-code | 11 | 116,387 | 142,946 | -26,559 | -18.58% | skill 1/1; base 1/1 |
| claude-code | 12 | 173,909 | 72,983 | +100,926 | +138.29% | skill 1/1; base 1/1 |
| claude-code | 2 | 635,520 | 865,338 | -229,818 | -26.56% | skill 1/1; base 1/1 |
| claude-code | 3 | 596,885 | 389,602 | +207,283 | +53.20% | skill 1/1; base 1/1 |
| claude-code | 4 | 422,651 | 329,822 | +92,829 | +28.15% | skill 1/1; base 1/1 |
| claude-code | 5 | 75,287 | 36,248 | +39,039 | +107.70% | skill 1/1; base 1/1 |
| claude-code | 6 | 108,795 | 107,415 | +1,380 | +1.28% | skill 1/1; base 1/1 |
| claude-code | 7 | 109,441 | 107,413 | +2,028 | +1.89% | skill 1/1; base 1/1 |
| claude-code | 8 | 169,966 | 107,358 | +62,608 | +58.32% | skill 1/1; base 1/1 |
| claude-code | 9 | 214,959 | 143,811 | +71,148 | +49.47% | skill 1/1; base 1/1 |
| codex | All cases | 1,802,207 | 1,647,323 | +154,884 | +9.40% | skill 12/12; base 12/12 |
| codex | 1 | 89,498 | 31,174 | +58,324 | +187.09% | skill 1/1; base 1/1 |
| codex | 10 | 114,764 | 150,785 | -36,021 | -23.89% | skill 1/1; base 1/1 |
| codex | 11 | 50,944 | 88,062 | -37,118 | -42.15% | skill 1/1; base 1/1 |
| codex | 12 | 129,225 | 82,435 | +46,790 | +56.76% | skill 1/1; base 1/1 |
| codex | 2 | 609,406 | 701,491 | -92,085 | -13.13% | skill 1/1; base 1/1 |
| codex | 3 | 122,548 | 40,702 | +81,846 | +201.09% | skill 1/1; base 1/1 |
| codex | 4 | 327,309 | 122,927 | +204,382 | +166.26% | skill 1/1; base 1/1 |
| codex | 5 | 89,927 | 100,505 | -10,578 | -10.52% | skill 1/1; base 1/1 |
| codex | 6 | 85,679 | 57,055 | +28,624 | +50.17% | skill 1/1; base 1/1 |
| codex | 7 | 27,475 | 41,680 | -14,205 | -34.08% | skill 1/1; base 1/1 |
| codex | 8 | 80,455 | 147,702 | -67,247 | -45.53% | skill 1/1; base 1/1 |
| codex | 9 | 74,977 | 82,805 | -7,828 | -9.45% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,033,273 | 4,446,810 | +586,463 | +13.19% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 21 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/batches.md and references/errors.md:
  "## Pitfalls" in references/batches.md (lines 146-171)
  vs "## Batches (plans)" in references/errors.md (lines 97-113) (`references/batches.md:146`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/errors.md and references/explicit-shapes.md:
  "## Explicit shapes" in references/errors.md (lines 125-135)
  vs "## Pitfalls (symptom -> fix)" in references/explicit-shapes.md (lines 141-164) (`references/errors.md:125`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across references/errors.md and references/native.md:
  "## Build and link (C/C++)" in references/errors.md (lines 22-33)
  vs "## Pitfalls (symptom -> fix)" in references/native.md (lines 119-161) (`references/errors.md:22`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across references/errors.md and references/queries.md:
  "## Queries" in references/errors.md (lines 83-96)
  vs "## Pitfalls (symptom -> fix)" in references/queries.md (lines 126-156) (`references/errors.md:83`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/ovspatialquery/SKILL.md`)
- 20 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
