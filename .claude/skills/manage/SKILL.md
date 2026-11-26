---
name: manage
description: Project-manager loop for emd_paper. Reads ROADMAP.md, picks goals, runs parallel code audits, verifies findings, applies approved cleanups as small commits, has slurm-planner write submit scripts for the compute the goals need, then updates ROADMAP.md. Use when the user types /manage (optionally with a focus such as "audit dirreco", "plan jobs", or a goal).
---
# /manage: one project-management pass

Optional focus from the user: **$ARGUMENTS**
- Empty: run every step.
- `audit [area]`: steps 1, 3, 4 and 5 only.
- `jobs` or `plan jobs`: steps 1, 2 and 7.
- Anything else: treat it as the goal for this pass.

## 1. Orient (do this yourself; it takes a few tool calls)
- Read `ROADMAP.md` and `CLAUDE.md`.
- Run `git status --short`, `git log --oneline -10`, and `squeue -u zimani`. Look in `results/slurm_logs/` and `results/studies/*/logs/run.log` for anything that finished since the last update in `ROADMAP.md`.
- If a run finished, read its `summary.csv` and figures before planning. A result can change the goals.
- If the working tree has uncommitted changes you didn't make, stop. Ask whether to commit them first; another session may be mid-edit.

## 2. Set goals
- From `ROADMAP.md`, propose the **1–3 goals** that would move the paper most. Unblocked items come first, and items on the critical path to final figures beat nice-to-haves.
- For each goal, give what "done" means, as a file, a figure or a number.
- Confirm the goals with the user before spending compute or making large edits. If `ROADMAP.md` lists a decision only the user can make (under "Decisions needed"), ask it now.

## 3. Audit (parallel subagents)
Launch `code-auditor` agents **in one message** so they run concurrently. Use one per area relevant to the goals, by default:
- `dirreco/`
- `cnn/`
- `ldm/` (the repo's own files; under `ldm/ldm/` only the lines this repo changed)
- `pipeline`: `run.py`, `plot.py`, `configs/`, `slurm/`, and the READMEs

Use at most four agents. If the focus names one area, audit only that area. Each prompt says which area to audit and what the current goals are, so the agent can rank findings that touch them higher.

## 4. Verify (yourself; don't skip this)
Subagents can report wrong claims confidently. For every finding you'd act on:
- Open the cited lines and rerun the cited grep.
- Drop findings that don't hold up. Downgrade "dead" to "unclear" if anything dynamic could reach the code: the method registry, `getattr`, config strings, or `plot.py` figure names.
- Sort the survivors into three groups:
  - **Safe**: no numerical change, e.g. dead code, a duplicate whose replacement is identical, or a doc fix.
  - **Numerical**: changes any estimator's output; needs a rerun.
  - **Perf**: same numbers, faster; needs an equivalence check like `selftest_emd` check 0.

## 5. Plan and get approval
Present the verified list in plan mode (`EnterPlanMode`), grouped as above. For each item give the location, the change, the risk, and how you'll check it. Numerical changes also list the studies they invalidate. Wait for approval. The user may approve only some items.

## 6. Apply
- **One commit per logical change**, in the repo's commit style, with the attribution line. Commit locally only. `git push` asks for permission; push only if the user says so.
- After each change, run `python -m py_compile` on the touched files.
- Checks that load the dataset or a model (`python -m dirreco.selftest_emd`, `python ldm/test_sampler.py`, `python run.py configs/smoke.yaml --jobs 4`) need a compute node. Ask the user for an interactive job ID, check it with `squeue`/`scontrol show job`, and run with `srun --jobid=<ID> --overlap -n1 ...`. If no node is available, commit with "unverified: <check>" in the message and add the check to the "Verification owed" list in `ROADMAP.md`.
- Never move or delete anything under `results/` or `archive/` without asking.
- Docs: update the README that describes the changed code in the same commit. Date any new measured number.

## 7. Compute plan (subagent)
If the goals need runs, launch the `slurm-planner` agent. Give it the goal, the configs or commands, the K and grid wanted, and any recent timings you already know. When it returns:
- Check its scripts: the sizing arithmetic, the `-o` path, the conda/LD_LIBRARY_PATH lines, and the dependency chain.
- Commit the scripts.
- Give the user the exact commands to submit. Never run `sbatch`, `scancel` or `submit_*.sh` yourself.

## 8. Update the roadmap
Edit `ROADMAP.md`:
- Tick or annotate finished items, with the date and the commit hash or job ID.
- Add new open items and decisions found this pass.
- Keep "Running / waiting on" current, and set "Last /manage pass" to today's date.
- Commit it.

If something learned this pass would matter in future sessions and isn't in the repo (e.g. "the user decided to drop the CoordConv CNN"), save it to auto-memory as well.

## 9. Report
End with a short summary:
- what was done (commits)
- what's running, or what the user should submit
- what's waiting on the user (decisions, job IDs, verification owed)
- the suggested next `/manage` focus
