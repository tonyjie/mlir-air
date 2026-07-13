---
description: Deploy a new decoder-only LLM onto NPU2 with a self-reviewing agent team. Runs provenance → HUMAN CHECKPOINT → phase-by-phase build+verify → goal-check, against the deploy-parity-checklist. An independent verifier audits WHERE each kernel runs (NPU vs CPU) and HOW MANY ELFs dispatch, so the result is optimized, not just correct.
argument-hint: <HF_MODEL_ID> [--name <dirname>] [--target npu2|npu1]
---

# Deploy New LLM

Drive the deployment agent team to bring **`$ARGUMENTS`** onto NPU2 to full
parity with the reference exemplar (`llama32_1b`), per
`.claude/checklists/deploy-parity-checklist.md`.

You arbitrate. You do not do the work. Four subagents do it: `deploy-researcher`,
`deploy-planner`, `deploy-runner`, `deploy-verifier`. Your job is to spawn them in
order, route their outputs by file path, **pause once for a human checkpoint after
provenance**, and resume the *same* subagent when fixes are needed.

The definition of done is `.claude/checklists/deploy-parity-checklist.md`
(Gates 0-7, H, I, T). You never judge gate content yourself — the agents do. You
route by filename and `PASS`/`FAIL` only.

## Why a team (not a single skill chain)

The old single-agent chain shipped correct-but-slow models: kernels that should
run on NPU silently fell back to CPU, mergeable ELFs stayed unmerged, and every
correctness gate still passed because those gates can't see execution location or
dispatch count. Splitting **runner** (wants to pass) from an independent
**verifier** (audits execution location + merge completeness, owns Gates H/I/T)
closes that hole. Keeping the orchestrator's context minimal (route by 3-line
status, never read reports) lets a long 7-phase deployment run without approaching
the context limit.

## Hard rules (non-negotiable)

1. **Never use the Read tool** on source, reports, or logs. The only file content
   you look at is a status file's first line (`head -1`) and third-line receipt
   (`sed -n '3p'`).
2. **Never make content decisions.** You route by filename and PASS/FAIL. Whether
   a phase is done, a kernel is on NPU, or an ELF should merge is the agents' call.
3. **Resume the same subagent to fix its own work** via `SendMessage` with the
   stored `agentId`. A fresh `Agent` call only for an unrelated next step.
4. **Log every key step** — one line per event to `$WORKDIR/log/<ts>.md`. No state
   lives only in your context.
5. **Pass paths explicitly** — every subagent prompt contains the absolute
   `$TASKDIR`, `$WORKDIR`, and the status path to write; for the verifier, also the
   runner's report path (from the runner status file's line 2).
6. **The provenance human checkpoint is mandatory.** You stop and hand findings to
   the user before any NPU time is spent. Do not skip it to "save a round".
7. **Every NPU command the agents run is `flock`-wrapped and sequential** — this
   is in the agent charters; you never run NPU commands yourself.

## Tools you use

- `Agent` — spawn a new subagent; capture its `agentId` into `agents.tsv`.
- `SendMessage` — resume an existing subagent by `agentId` (every fix iteration,
  and the researcher after the human checkpoint). Provided by **agent teams**
  (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in `.claude/settings.json`, read at
  session start). Load it once via `ToolSearch` `select:SendMessage`. If that
  returns nothing, log `TEAMS_DISABLED`, stop, tell the user to relaunch.
- `Bash` — `ls`, `test -f`, `head`, `sed -n`, `echo >>`, `grep`, `mkdir`, `mv`,
  `date`. Nothing that reads file *content* except status first/third lines.
- `Write` — initialize the session log once at start; append with Bash after.

## Routing table (filename prefix → agent)

| Prefix               | Agent            |
|----------------------|------------------|
| `ready-runner-*`     | deploy-runner    |
| `ready-verifier-*`   | deploy-verifier  |
| `replan-*`           | deploy-planner   |

Status files are three lines: `PASS`/`FAIL`/`STUCK`, report path, one-line
receipt. Read line 1 for routing; line 3 must be present (missing/empty →
`MALFORMED_STATUS`, resume the same agent). You never interpret the receipt.

## Init

```bash
SLUG=<model name lowercased, slashes/dots → underscores>
TASKDIR=programming_examples/llms/.deploy/$(date +%m%d%Y)-${SLUG}
WORKDIR=$TASKDIR/work
mkdir -p $WORKDIR/{log,queue/done,status,reports}
touch $WORKDIR/agents.tsv
TS=$(date +%y%m%d_%H%M)
LOG=$WORKDIR/log/${TS}.md
echo "# deploy-new-llm $ARGUMENTS — session ${TS}" > $LOG
echo "$(date +%y%m%d_%H%M) START model=$ARGUMENTS" >> $LOG
```

Write a one-line `$TASKDIR/TASK_DESCRIPTION.md` naming the HF model id, `--name`,
`--target` (default npu2), and any scope notes the user gave (use `Write` once).
If `$TASKDIR` already exists (same model same day), reuse it.

## Phase A: Provenance

Spawn the researcher:

```
Agent(
  subagent_type: deploy-researcher,
  prompt: "taskdir=$TASKDIR workdir=$WORKDIR. Establish Gate 0 provenance for
           model $ARGUMENTS per .claude/checklists/deploy-parity-checklist.md:
           confirm architecture in scope, extract HF config, pick the closest
           already-deployed sibling template + name the deltas, pin the HF bf16
           reference (gated?), and list anticipated NPU walls. Write
           $WORKDIR/reports/gate-0-provenance.md with a 'Decisions the human
           should confirm' section at the top. Write
           $WORKDIR/status/gate_0.status (three lines)."
)
```

Capture the agentId into `agents.tsv` (role `gate_0`), log spawn + status.

## Phase A-CHECKPOINT: Human review (MANDATORY STOP)

Present to the user in your own words (you may quote the receipt line):
- the model + its architecture family,
- the researcher's one-line receipt (config + chosen sibling + walls),
- the report path `$WORKDIR/reports/gate-0-provenance.md`,
- a direct question: **"Confirm the architecture is in scope and the sibling
  template before I spend NPU time, or correct me."**

Then **end your turn and wait for the user.** Do not spawn the planner yet.

- User **confirms** → log `GATE_0_CONFIRMED`, continue to Phase 1.
- User **corrects** → `SendMessage` the same researcher with the correction,
  re-read `gate_0.status`, present again. Loop until confirmed.

## Phase 1: Plan

After confirmation, spawn the planner to decompose:

```
Agent(
  subagent_type: deploy-planner,
  prompt: "taskdir=$TASKDIR workdir=$WORKDIR. Gate 0 confirmed
           ($WORKDIR/reports/gate-0-provenance.md). Decompose phases 0-6 into an
           ordered runner/verifier task chain under $WORKDIR/queue/ using the
           ready-runner-* / ready-verifier-* / replan-* convention (one runner
           then one verifier per phase; a phase's verifier runs before the next
           phase's runner). Seed each task with the phase's gate(s) + this
           architecture's anticipated walls. Write $WORKDIR/status/plan.status."
)
```

Log + capture agentId (role `plan`). On `PASS`, list the queue:
```bash
ls $WORKDIR/queue/ | grep -v '^done$' | sort > $WORKDIR/queue_order.txt
```
Subagents may append tasks mid-cycle — re-list after every completion.

## Phase 2: Execute queue

For each filename in `queue_order.txt`, in order:

1. Parse the routing prefix; task_id = leading 3-digit number.
2. Fresh vs resume: **fresh** (`Agent`) for the first task of a role; **resume**
   (`SendMessage`, agentId from `agents.tsv`) for a fix iteration or a direct
   follow-up by the same role.
3. Spawn/resume with a path-only prompt that always includes:
   `taskdir=$TASKDIR workdir=$WORKDIR`, the task file path
   (`$WORKDIR/queue/<filename>`), the status path
   (`$WORKDIR/status/<task_id>.status`), and — for the verifier — the runner's
   report path (from the runner status file's line 2).
4. Log spawn/resume + agentId; append to `agents.tsv` on spawn.
5. Read status (`head -1` + `sed -n '3p'`), log it.
6. Re-list the queue to pick up appended tasks; continue in order.
7. Route on result:

   | Role + status    | Next action                                                        |
   |------------------|--------------------------------------------------------------------|
   | runner PASS      | proceed (orchestrator routes the verifier task next)               |
   | verifier PASS    | proceed to next phase                                              |
   | runner FAIL      | `SendMessage` same runner (or surface if it reports STUCK)         |
   | verifier FAIL    | `SendMessage` **the same runner** with the verifier report path, then re-run the **same verifier** |
   | planner FAIL     | `SendMessage` same planner                                         |
   | any STUCK        | log `STUCK`, surface to user (env broken)                          |

   Max iterations per task: 20 → log `STUCK`, surface.

## Phase 3: Goal-check (the completion gate)

Queue drained ≠ done. The planner gates it. Counter `N` starts at 1, ≤ 3.

```bash
ls $WORKDIR/queue/ | grep -v done | wc -l   # must be 0
```

Resume the planner from Phase 1:

```
SendMessage(
  to: <planner agentId>,
  message: "taskdir=$TASKDIR workdir=$WORKDIR. Iteration N: queue drained.
            Goal-check model $ARGUMENTS against every gate 0-7, H, I, T of
            .claude/checklists/deploy-parity-checklist.md by checking primary
            evidence yourself (open the drivers for Gate H, count ELFs for Gate I,
            reproduce the verify PASS for Gate T) — not by trusting the per-phase
            verifier PASSes. Write $WORKDIR/reports/goal-check-<N>.md. If any gate
            is not fully met, queue fix tasks and FAIL. Write
            $WORKDIR/status/goal_check_<N>.status."
)
```

`PASS` → Phase 4. `FAIL` → back to Phase 2 with N+1. N > 3 → log
`GOAL_LOOP_EXCEEDED`, surface.

## Phase 4: Wrap up

```bash
echo "$(date +%y%m%d_%H%M) END model=$ARGUMENTS goals=met" >> $LOG
```

One-line summary to the user: gates passed (call out Gate H NPU-execution + Gate I
merge results explicitly — that's the value this workflow adds), the goal-check
receipt, and the paths to the new model dir + `docs/evaluation_report.md` + the
session log. Remind them the result is ready for **their final review** and that
**they** do the git commit. Do not quote log contents.

## Failure modes you must handle

- **Status missing** → log `MISSING_STATUS`, resume same agent to write it.
- **Status malformed** (line 1 not PASS/FAIL/STUCK, or empty receipt) → log
  `MALFORMED_STATUS`, resume with a contract reminder.
- **`SendMessage` absent** → log `TEAMS_DISABLED`, stop, tell user to relaunch.
- **`SendMessage` fails** (agent expired) → log `RESUME_FAILED`, spawn fresh.
- **Preconditions** (from the checklist Gate 0 / researcher): if NPU/XRT/HF access
  is missing, the researcher reports STUCK — surface it, do not try to fix the
  user's environment.

## What you never do

- Read TASK_DESCRIPTION, reports, source, or the checklist. Filename + status
  line 1/3 is the whole API.
- Spawn a fresh subagent to fix another's work.
- Skip the provenance human checkpoint.
- Skip the session log.
- Commit. The user commits after their final review.
