# Handover: hw6 pitch predictor

Last updated: 2026-11-02. Read this first in a new session, then `../CLAUDE.md`. Austin is the owner; they are strong on baseball and moderate on ML, so explain ML choices plainly.

## Where we are
**Phase: proposal drafted, awaiting Austin's review. Nothing is built or downloaded.** Austin has not yet said "full prototype". Do not start building until they do.

## Files in `hw6/`
| File | What it is |
|---|---|
| `human_notes/proj_proposal` | Austin's own original pitch. **Their words; don't edit.** |
| `project_brainstorming.md` | Raw interview log (rounds 1-5, postseason follow-up, research summary). Source of truth for what Austin actually decided. |
| `research_notes.md` | Prior work, Statcast/pybaseball facts, regime changes (pitch clock, 2026 ABS challenge), leakage pitfalls, with links. Marks what is unverified. |
| `proposal.md` | The design: problem, data, features, model ladder, splits, metrics, leakage controls, demo, report, repo layout, phases, risks, open questions. |
| `HANDOVER.md` | This file. |

Also: `../CLAUDE.md` (repo-level context, includes an hw6 section). **Git state:** `hw6/` and the CLAUDE.md edits are uncommitted. Austin hasn't asked for a commit.

## Decisions made by Austin (firm)
- Goal: working prototype first, with some rigorous eval and interpretability; not required to prove a sequence model wins.
- User: a hitter pre-pitch, so pre-pitch information only and calibrated probabilities.
- Inputs: count/outs/bases, batter handedness, pitcher via **arsenal stats (not ID embeddings)**, current at-bat sequence, **prior-meeting summary** (pitcher vs this batter).
- Outputs: **full Statcast pitch_type codes**; location as a **continuous (x, z) density**.
- Data: **3 most recent seasons** (= 2024, 2025, 2026; 2026 appears complete). Compute: Colab.
- Splits: several (time primary, plus held-out pitchers and held-out games).
- Metrics: log loss/perplexity, calibration, per-pitcher skill vs baseline, top-1/top-3.
- Deliverables: **interactive demo + written report**, with the report readable by non-expert fans first and statistical detail layered on.
- Success = a compelling demo/story. Drop position players pitching.
- Postseason: confirmed default of regular-season training, postseason held out as a stress test (no separate postseason model).
- Working mode: propose now; **full prototype only when asked**, and be ready to start at Phase 0.

## Proposed by Claude, not explicitly confirmed (flag these when Austin reviews)
- **Location order:** Austin chose continuous density; the proposal does a **26-cell grid first** and treats the mixture-density head as a stretch (stability risk). Needs Austin's OK.
- **2023 burn-in download** for feature warm-up only.
- Drop intentional balls, pitchouts, automatic balls/strikes; regular season only for training.
- Time split cut points (train through 2025-08-01, validate rest of 2025, test 2026), and splits B and C as defined.
- Model ladder includes a **bigram rung (B3)** tying back to hw5, plus LightGBM, GRU, small transformer.
- Gradio demo launched from Colab (optional Hugging Face Space later).
- Hypotheses H1-H5 and the "hitter payoff" stretch metric.

## Open threads
1. **Deadline and scope unknown.** No project spec in the repo (searched). Austin to check the syllabus or course site. This decides how far past Phase 2 to go.
2. **Solo or team?** Asked in the proposal; not answered.
3. **Report format** the course wants (Markdown/PDF/Doc) is unknown.
4. **Colab plan** (free vs Pro) not decided.
5. **Position-player rule** is undecided (candidates: appears as a fielder at another position, or MLB Stats API position).
6. The **sweeper/slurve and rare pitch-code handling** (frequency floor about 0.5%) is to be decided from the data.

## Facts stated from memory or search summaries: verify in Phase 0
- Savant's per-download cap (about 25k rows; one user guessed ~30k) and whether `pybaseball.statcast()` chunks internally.
- Exact Statcast column names and which are pre- vs post-pitch (the allowlist in `proposal.md` §8 is from memory). Check against https://baseballsavant.mlb.com/csv-docs.
- Pitch-code definitions for rare codes; sources disagree.
- Sign convention for `plate_x` mirroring (intended: +x means away from the batter for both handedness).
- Rough numbers: ~700k pitches per season, ~4 pitches per plate appearance, ~15-20k postseason pitches per year. These are estimates.
- Prior-work numbers (39.1% vs 39.6%, etc.) come from blog and abstract summaries, not full papers. Don't quote them in the report without reading the sources.
- 2026 season completeness: confirm the last pitch date in the data is late October 2026. I never fetched an official MLB page.

## Environment notes
- Local: Windows 11, Python 3.12.8, **numpy only** (no pandas, torch, pybaseball). Heavy work happens in Colab, which the assistant can't drive, so Austin must run notebooks there and share outputs.
- Planned installs in Colab: `pybaseball`, `lightgbm`, `gradio`, `pyarrow` (PyTorch is preinstalled).
- `hw6/data/` should be git-ignored and raw Statcast shouldn't be committed (size, MLB data terms). Create that `.gitignore` in Phase 0.
- Some connectors (GitHub MCP, Slack) were unavailable this session; not needed for the project.

## Next steps
1. Austin reviews `proposal.md`, answers the open threads above, and confirms or changes the flagged proposals (especially grid-first location).
2. Update `proposal.md` and `CLAUDE.md` with any changes.
3. On "full prototype": start at Phase 0 (download and explore), following the phases, repo layout and leakage controls in `proposal.md` §8, §11, §12. Pause for review after each phase.
