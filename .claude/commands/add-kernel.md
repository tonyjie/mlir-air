---
description: Add one kernel to programming_examples/kernel_registry with a self-reviewing agent team. Runs provenance → HUMAN CHECKPOINT → measure+document → verify → goal-check, against the parity checklist. Hands you a complete, trustworthy result, not a back-and-forth.
argument-hint: <KERNEL_NAME> (e.g. SiLU_Mul, RoPE, EltwiseAdd)
---

# Add Kernel

Drive the kernel-registry agent team to add **`$ARGUMENTS`** to
`programming_examples/kernel_registry`, to full parity with an already-trusted
kernel (GEMV is the gold standard).

You arbitrate. You do not do the work. Four subagents do it:
`kernel-researcher`, `kernel-runner`, `kernel-verifier`, `kernel-planner`. Your
job is to spawn them in order, route their outputs by file path, **pause once for
a human checkpoint after provenance**, and resume the *same* subagent when fixes
are needed.

The definition of done is `.claude/checklists/kernel-parity-checklist.md` (Gates A–F). You
never judge gate content yourself — the agents do. You route by filename and by
`PASS`/`FAIL` only.

## Hard rules (non-negotiable)

1. **Never use the Read tool** — not on source, reports, or logs. The only file
   content you look at is the first line of a status file (`head -1`) and its
   third-line receipt (`sed -n '3p'`).
2. **Never make content decisions.** You route by filename and PASS/FAIL. Whether
   a sweep is complete or a precision gate is sound is the agents' call.
3. **Resume the same subagent to fix its own work** via `SendMessage` with the
   stored `agentId`. A fresh `Agent` call only for an unrelated next step.
4. **Log every key step** — one line per event to `$WORKDIR/log/<ts>.md`. No state
   lives only in your context.
5. **Pass paths explicitly** — every subagent prompt contains the absolute
   `$TASKDIR` and `$WORKDIR` and the status path to write.
6. **The Gate-A human checkpoint is mandatory.** You stop and hand findings to the
   user before any NPU time is spent. Do not skip it to "save a round".

## Tools you use

- `Agent` — spawn a new subagent; capture its `agentId` into `agents.tsv`.
- `SendMessage` — resume an existing subagent by `agentId` (every fix iteration,
  and the researcher after the human checkpoint). Provided by **agent teams**
  (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in `.claude/settings.json`, read at
  session start). Load it once via `ToolSearch` `select:SendMessage`. If that
  returns nothing, the flag wasn't active — log `TEAMS_DISABLED`, stop, tell the
  user to relaunch Claude Code.
- `Bash` — `ls`, `test -f`, `head`, `sed -n`, `echo >>`, `grep`, `mkdir`, `mv`,
  `date`. Nothing that reads file *content* except status first/third lines.
- `Write` — initialize the session log once at start; append with Bash after.

## Routing table (filename prefix → agent)

| Prefix               | Agent               |
|----------------------|---------------------|
| `ready-runner-*`     | kernel-runner       |
| `ready-verifier-*`   | kernel-verifier     |
| `ready-docreview-*`  | kernel-doc-reviewer |
| `replan-*`           | kernel-planner      |

Status files are three lines: `PASS`/`FAIL`/`STUCK`, report path, one-line receipt.
You read line 1 for routing; line 3 must be present and non-empty (missing/empty →
`MALFORMED_STATUS`, resume the same agent to finish it). You never interpret the
receipt — it's an audit trail.

## Init

```bash
SLUG=<kernel name lowercased, hyphenated>
TASKDIR=programming_examples/kernel_registry/.add-kernel/$(date +%m%d%Y)-${SLUG}
WORKDIR=$TASKDIR/work
mkdir -p $WORKDIR/{log,queue/done,status,reports}
touch $WORKDIR/agents.tsv
TS=$(date +%y%m%d_%H%M)
LOG=$WORKDIR/log/${TS}.md
echo "# add-kernel $ARGUMENTS — session ${TS}" > $LOG
echo "$(date +%y%m%d_%H%M) START kernel=$ARGUMENTS" >> $LOG
```

Write a one-line `$TASKDIR/TASK_DESCRIPTION.md` naming the kernel and any scope
notes the user gave (use `Write` once). If `$TASKDIR` already exists (same kernel
same day), reuse it.

## Phase A: Provenance

Spawn the researcher:

```
Agent(
  subagent_type: kernel-researcher,
  prompt: "taskdir=$TASKDIR workdir=$WORKDIR. Establish Gate A provenance for
           kernel $ARGUMENTS per .claude/checklists/kernel-parity-checklist.md: confirm the
           standalone harness == the variant llama uses (A1), verify llama's
           config (A2), locate the GPU/HF reference + threshold (A3). Write
           $WORKDIR/reports/gate-a-provenance.md with a 'Decisions the human
           should confirm' section at the top. Write
           $WORKDIR/status/gate_a.status (three lines)."
)
```

Capture the agentId:
```bash
echo "$(date +%y%m%d_%H%M) GATE_A spawned researcher id=<id>" >> $LOG
printf "gate_a\tkernel-researcher\t<id>\n" >> $WORKDIR/agents.tsv
STATUS=$(head -1 $WORKDIR/status/gate_a.status)
RECEIPT=$(sed -n '3p' $WORKDIR/status/gate_a.status)
echo "$(date +%y%m%d_%H%M) GATE_A status=$STATUS receipt=\"$RECEIPT\"" >> $LOG
```

On `FAIL` (provenance genuinely open), proceed to the checkpoint anyway — the
human resolves it there.

## Phase A-CHECKPOINT: Human review (MANDATORY STOP)

Stop and present to the user, in your own words (you may quote the receipt line —
that's a status line, not file content):

- the kernel,
- the researcher's one-line receipt (variant + GPU reference),
- the report path `$WORKDIR/reports/gate-a-provenance.md` for them to open,
- a direct question: **"Confirm the harness variant and GPU reference before I
  spend NPU time, or correct me."**

Then **end your turn and wait for the user.** Do not spawn the runner yet.

- If the user **confirms**: log `GATE_A_CONFIRMED`, continue to Phase 1.
- If the user **corrects**: `SendMessage` the same researcher with the correction,
  re-read `gate_a.status`, present again. Loop until the user confirms.

```bash
echo "$(date +%y%m%d_%H%M) CHECKPOINT awaiting human confirmation" >> $LOG
```

## Phase 1: Plan

After human confirmation, spawn the planner to decompose:

```
Agent(
  subagent_type: kernel-planner,
  prompt: "taskdir=$TASKDIR workdir=$WORKDIR. Gate A is confirmed
           ($WORKDIR/reports/gate-a-provenance.md). Decompose the remaining work
           (Gates B-F) into task files under $WORKDIR/queue/ using the
           ready-runner-* / ready-verifier-* / replan-* naming convention. Write
           $WORKDIR/status/plan.status (three lines)."
)
```

Log + capture agentId into `agents.tsv` (role `plan`). On `PASS`, list the queue:
```bash
ls $WORKDIR/queue/ | grep -v '^done$' | sort > $WORKDIR/queue_order.txt
```
Subagents may append tasks mid-cycle — re-list after every completion before
declaring the queue drained.

## Phase 2: Execute queue

For each filename in `queue_order.txt`, in order:

1. Parse the routing prefix; task_id = leading 3-digit number.
2. Fresh vs resume: **fresh** (`Agent`) for the first task of a role; **resume**
   (`SendMessage`, agentId from `agents.tsv`) for a fix iteration or a direct
   follow-up by the same role.
3. Spawn/resume with a path-only prompt that always includes:
   `taskdir=$TASKDIR workdir=$WORKDIR`, the task file path
   (`$WORKDIR/queue/<filename>`), the status path to write
   (`$WORKDIR/status/<task_id>.status`), and — for the verifier — the runner's
   report path (from the runner status file's line 2).
4. Log spawn/resume + agentId; append to `agents.tsv` on spawn (not resume).
5. Read status:
   ```bash
   STATUS=$(head -1 $WORKDIR/status/${task_id}.status)
   RECEIPT=$(sed -n '3p' $WORKDIR/status/${task_id}.status)
   echo "$(date +%y%m%d_%H%M) TASK ${task_id} status=$STATUS receipt=\"$RECEIPT\"" >> $LOG
   ```
6. Re-list the queue to pick up appended tasks; continue in order.
7. Route on result:

   | Role + status      | Next action                                                            |
   |--------------------|------------------------------------------------------------------------|
   | runner PASS        | proceed (orchestrator will route the verifier task next)               |
   | verifier PASS      | proceed (doc-review task runs after the numbers are trusted)           |
   | doc-reviewer PASS  | proceed; closes the task                                                |
   | runner FAIL        | `SendMessage` same runner (numerics ambiguity → surface to user)       |
   | verifier FAIL      | `SendMessage` **the same runner** with the verifier report path, then re-run the **same verifier** |
   | doc-reviewer FAIL  | `SendMessage` **the same runner** with the doc-reviewer report path, then re-run the **same doc-reviewer** |
   | planner FAIL       | `SendMessage` same planner                                              |
   | any STUCK          | log `STUCK`, surface to user (env broken)                               |

   Max iterations per task: 20 → log `STUCK`, surface.

## Phase 3: Goal-check (the completion gate)

Queue drained ≠ done. The planner gates it. Counter `N` starts at 1, ≤ 3 outer
iterations.

```bash
ls $WORKDIR/queue/ | grep -v done | wc -l   # must be 0
```

Resume the planner from Phase 1:

```
SendMessage(
  to: <planner agentId>,
  message: "taskdir=$TASKDIR workdir=$WORKDIR. Iteration N: queue drained.
            Goal-check kernel $ARGUMENTS against every gate A-F of
            .claude/checklists/kernel-parity-checklist.md, by checking the actual registry
            files (not the verifier's PASS). Write
            $WORKDIR/reports/goal-check-<N>.md. If any gate is not fully met,
            queue more tasks and FAIL. Write
            $WORKDIR/status/goal_check_<N>.status (three lines)."
)
```

`PASS` → Phase 4. `FAIL` → back to Phase 2 with N+1. N > 3 → log
`GOAL_LOOP_EXCEEDED`, surface to user.

## Phase 4: Wrap up

```bash
echo "$(date +%y%m%d_%H%M) END kernel=$ARGUMENTS goals=met" >> $LOG
```

One-line summary to the user: gates passed, the goal-check receipt, the paths to
the new registry pages and the session log. Remind them the result is ready for
**their final review** and that **they** do the git commit. Do not quote log
contents.

## Failure modes you must handle

- **Status missing** → log `MISSING_STATUS`, resume same agent to write it.
- **Status malformed** (line 1 not PASS/FAIL/STUCK, or empty receipt) → log
  `MALFORMED_STATUS`, resume with a contract reminder.
- **`SendMessage` absent** (`select:SendMessage` empty) → log `TEAMS_DISABLED`,
  stop, tell the user to relaunch (the teams flag is read only at session start).
- **`SendMessage` fails** (agent expired) → log `RESUME_FAILED`, spawn fresh
  (only case where fresh replaces resume).

## What you never do

- Read TASK_DESCRIPTION, reports, source, the detail pages, or sweep CSVs.
  Filename + status line 1/3 is the whole API.
- Spawn a fresh subagent to fix another's work.
- Skip the Gate-A human checkpoint.
- Skip the session log.
- Commit. The user commits after their final review.
