# Proposal: Predicting the next pitch, and where it goes

CPSC 1710 project · Austin Kwoun · drafted 2026-11-02 · status: **proposal, not yet built**

Supporting files: [raw interview notes](project_brainstorming.md) · [research notes with sources](research_notes.md)

---

## 1. One-paragraph summary

Given everything a hitter could know before a pitch (count, outs, runners, score, who's pitching and hitting, the pitcher's usual arsenal, what's already been thrown this at-bat, and how this pitcher has attacked this batter before), predict **two things**: the probability of each pitch type, and the probability of where it will cross the plate. We train on three recent Statcast seasons, compare a ladder of models from "league average by count" up to a small transformer, and judge all of them on held-out future games using strict leakage controls. Deliverables: an interactive demo, a fan-readable written report, and a notebook that reproduces everything on Colab.

## 2. The honest framing (decided up front)

- The **count alone predicts a lot**. In published and blog work, a lookup table of "most common pitch in this count" lands within about half a point of top-1 accuracy of a tuned XGBoost (39.1% vs 39.6%). So expect modest absolute gains. Our story is **where and for whom** pitches are predictable, not "AI solves pitching."
- We judge by **log loss (how surprised the model is on average) and calibration (does "60%" mean 60%?)**, not only top-1 accuracy. A hitter needs probabilities, not a single guess.
- Location is the less-explored half and probably the more interesting one.
- "A sequence model beats a tabular model" is a hypothesis we test, not a promise. At-bats are short (about 4 pitches on average), so the sequence advantage may be small. A clear "it doesn't help much, here's why" is still a good result.

## 3. Problem definition

**Unit of prediction:** the next pitch in a plate appearance, at pitch position *t+1*, given everything known before it is thrown.

**Outputs**
1. `P(type | context)`: distribution over Statcast pitch types (FF, SI, FC, SL, ST, SV, CU, KC, CH, FS, ...). Rare classes below a frequency floor (decide from data, about 0.5%) merge into `OTHER`.
2. `P(location | context, type)`: where the pitch crosses the plate.
3. Joint: `P(type) × P(location | type)`. Location conditional on type is the natural order: a hitter first guesses *what*, then *where*.

**Location coordinates**
- Measured by Statcast `plate_x` (feet, catcher's view) and `plate_z` (feet high).
- Normalize height by the batter's zone: `z_norm = (plate_z − sz_bot) / (sz_top − sz_bot)`, so 0 to 1 is the vertical strike zone for every batter. Scale `plate_x` by the plate half-width plus ball radius (about 0.83 ft).
- **Mirror x for left-handed batters** so +x always means "away from the batter." Then "inside" and "away" mean the same thing for everyone (verify sign conventions in Phase 0).
- **Reporting grid (26 cells):** the classic 3×3 strike zone, a ring of 16 "shadow" cells one third-zone beyond each edge (a 5×5 grid in total), plus one "way outside" bucket for everything else. This is what we show fans and what every model is scored on, so all models are comparable.

**Excluded rows:** pitches by position players, intentional balls, pitchouts, automatic balls/strikes from clock violations, rows with missing type or location (counted and reported, not silently dropped). **Regular season only** for training and primary testing.

## 4. Data

| Item | Plan |
|---|---|
| Source | Statcast through `pybaseball`, one day per request (Savant caps ~25k rows per download) |
| Seasons | **2024, 2025, 2026** for train/validate/test (~700k pitches each, ~2M total) |
| Burn-in | **2023**, used *only* to compute arsenal and head-to-head features so early-2024 pitchers aren't cold-start. Never used as training targets. (Optional; cheap and recommended) |
| Postseason | Excluded from training; used as a held-out stress test (see §7) |
| Storage | Parquet on Google Drive; `hw6/data/` git-ignored; download once |
| Position players | Flag with a rule to be chosen in Phase 0 (candidate: pitcher also appears as a position-player fielder, or MLB Stats API position); log how many pitches are dropped |

## 5. Features (everything strictly pre-pitch)

**A. Situation:** balls, strikes, outs, runners on each base, inning, home/away, score difference, plate-appearance number in the game, times through the order.

**B. Batter:** handedness; a few historical-rate summaries computed as-of (whiff rate, chase rate, strikeout rate, maybe by pitch group). Batter ID embedding is an *ablation*, not the default.

**C. Pitcher arsenal (as-of, shrunk):** for each pitch type, usage rate, average velocity, average horizontal and vertical movement, overall and by count bucket (ahead / even / behind, 2-strike). **Shrinkage:** blend the pitcher's own to-date numbers with the league (or previous-season) numbers: `(n·own + k·prior) / (n + k)`, so pitchers with few pitches don't produce noisy extremes. Pitcher handedness included.

**D. At-bat sequence:** the pitches already thrown in this plate appearance. Each earlier pitch is a token with: type, normalized location, velocity (z-scored within pitch type), and outcome (ball, called strike, swinging strike, foul, in play).

**E. Prior-meeting summary:** this pitcher vs this batter before this pitch: pitch-type mix and outcomes from earlier plate appearances in the same game, plus (shrunk) history in earlier games. Includes "how did the last plate appearance end?"

**Pitcher and batter IDs are *not* inputs by default.** That's the point of using arsenal stats: a pitcher the model never trained on still gets sensible predictions, which the held-out-pitcher test checks.

## 6. Model ladder

Each rung adds one idea, so differences between rungs are interpretable.

| # | Model | What it adds | Plain-language role |
|---|---|---|---|
| B0 | League overall | Nothing | "Throw the usual pitch" |
| B1 | League × count × batter hand × pitcher hand | Situation | The strong baseline to beat |
| B2 | Pitcher's own mix by count (shrunk toward B1) | The individual pitcher | "Does knowing the pitcher matter?" |
| B3 | Bigram: previous pitch type → next (conditioned on count) | One pitch of memory; **direct link to hw5's bigram model** | Course connection |
| M1 | LightGBM on all tabular features plus last 1-3 pitches as lag features | Strong non-sequence model | The honest "does sequence matter?" control |
| M2 | GRU over the at-bat tokens + context vector | Learned sequence summary | Small RNN (like hw4) |
| M3 | Small causal transformer over the at-bat tokens + context | Attention over history | Like hw5's GPT, tiny scale |

**Location heads**
- **Stage 1 (prototype default):** classify into the 26 reporting cells, conditioned on the (true, during training) pitch type. Stable, easy to evaluate, easy to explain.
- **Stage 2 (stretch):** mixture-density head: predict a small mixture of 2D Gaussians over (x, z_norm), conditioned on type. This is the "continuous density" you asked for. Mixture training can be unstable, so it comes after Stage 1 works, and is scored by (a) density log-likelihood and (b) probability mass integrated into the 26 cells, so it stays comparable with Stage 1.
- Baselines B0-B2 use the empirical histogram over the 26 cells given type (league, league × count, pitcher-own shrunk).

**Sizes and training:** M2/M3 are under about 1M parameters; at-bats are at most about 15 pitches. Train on a Colab T4 in minutes per epoch. AdamW, early stopping on validation log loss, loss = type cross-entropy + location NLL, predicted at every position in the at-bat (like GPT next-token training). Save checkpoints to Drive.

## 7. Evaluation design

**Splits**

| Split | Train | Validate | Test | Question answered |
|---|---|---|---|---|
| **A. Time (primary)** | 2024 + 2025 up to Aug 1 | 2025 Aug 1 to season end | **2026 regular season** | Would this have worked on the next season? |
| **B. New pitchers** | A random ~80% of pitchers (stratified by volume), 2024-2025 | same pitchers | **Held-out ~20% of pitchers**, 2026 | Does it work for pitchers it never saw? (cold start) |
| **C. October stress test** | Same as A | n/a | 2024, 2025, 2026 **postseason** | Does it transfer when stakes rise? |

"Held-out games per pitcher" (your original idea) is covered by A: the test games are later games for the same pitchers. If you want a *random*-games variant as well, it's a cheap addition but easier than A, so it would only be a reference point.

**Metrics** (all on the test set; computed for type, location, and joint)
1. **Log loss / perplexity**, primary. Report **skill score** = 1 − (model loss ÷ baseline loss), i.e., "% less surprised than the baseline."
2. **Top-1 / top-3 accuracy**, for intuition and to compare to published numbers.
3. **Calibration:** reliability curves and expected calibration error, for the top few pitch types and for the location cells.
4. **Per-pitcher skill vs baseline:** for each pitcher with enough pitches, skill versus B1 (league by count) *and* versus B2 (their own habits). This produces the "most and least predictable pitchers" lists.
5. **Hitter payoff (stretch):** if a hitter "sat" on the model's single most likely (type, 3×3 cell) guess, how often would they be right, compared with sitting on B1's guess? This is the direct test of your "Why."

**Uncertainty:** pitches within a game aren't independent, so confidence intervals use a **bootstrap over games**, not over pitches. Pairwise model comparisons use the same resampled games.

**Pre-registered hypotheses** (written now so we can't fool ourselves later)
- H1: Every model beats B0 by a lot; B1 captures most of the achievable gain; M1-M3 beat B1 by a small but positive log-loss margin.
- H2: B2 beats B1 clearly (pitcher habits matter more than count alone).
- H3: M2/M3 beat M1 by a small margin if at all; the margin is larger for pitchers with larger arsenals.
- H4: Predictability varies a lot by pitcher and situation (the story).
- H5: Postseason skill is similar to the regular season; any gap is smaller than the gap between B1 and B2.

## 8. Leakage controls (the part that decides whether results are believable)

1. **Feature allowlist.** Only named pre-pitch columns can reach a model. A banned list (every current-pitch measurement, outcomes, `post_*`, `events`, `launch_*`, `estimated_*`) raises an error if present.
2. **Sort first.** Every dataset is sorted by `game_date, game_pk, at_bat_number, pitch_number` before any lag or cumulative feature.
3. **As-of aggregates.** Arsenal, batter, and head-to-head features use cumulative sums shifted so a game's own pitches never count toward its features (burn-in season 2023 seeds the start).
4. **Truncation-invariance test** (automated): recompute the features for pitch *i* using a dataset with every later pitch deleted. They must match the full-data features exactly. This single test catches most leaks.
5. **Date-based splits** with a buffer-free boundary; no pitch-level random splits.
6. **Fit on train only:** scalers, class frequency floors, shrinkage priors for the test period use only earlier data.
7. **Sanity checks:** shuffled-label model must score at the B0 level; a model with *only* the current pitch's own velocity (a banned feature) is run once to show how a leak would look.

## 9. Interactive demo

**Choice:** a Gradio app launched from the Colab notebook (shareable link), with an optional Hugging Face Space later for a permanent URL. (A static page is possible but can't run the model live.)

**Inputs:** pitcher (dropdown), batter hand, count, outs, runners, score/inning, and up to a few previous pitches in the at-bat (type, rough location, result).

**Outputs**
- A bar chart of predicted pitch-type probabilities, with the **league-by-count** and the **pitcher's usual mix** alongside, so the model's contribution is visible.
- A strike-zone heatmap (catcher's view, 26 cells or the smooth density), switchable by pitch type or "all pitches."
- A one-line plain-English summary: "In a 1-2 count against a righty, Pitcher X goes to the slider 41% of the time (league: 29%), most often low and away."

## 10. Written report (layered for readers)

Written so a casual fan can follow the main text and a data scientist can drill into the detail.

- **Layer 1, main text (plain English, about 4-6 pages):** the question, what the model sees, what "predictable" means, 4-5 headline figures, what hitters could do with it, limits.
- **Layer 2, callout boxes:** "What is log loss?" "What is calibration?" "What is a transformer, in one paragraph?" Each box can be skipped.
- **Layer 3, technical appendix:** feature definitions, model details, split design, leakage tests, bootstrap method, full tables.

**Candidate headline figures (the "compelling story")**
1. A count-by-count pitch mix chart: how the count shapes what gets thrown (fans recognize this immediately).
2. "Most and least predictable pitchers": skill vs league by count and vs own habits.
3. One pitcher's example sequence: a walk-through of predicted vs actual at each pitch of an at-bat.
4. Calibration plot, with an explanation of why it matters for hitters.
5. October vs regular season; and (stretch) 2025 vs 2026 locations after ABS.
6. The hitter-payoff result, if the stretch metric gets done.

## 11. Repository layout (ready for "full prototype")

```
hw6/
  project_brainstorming.md     raw interview notes (exists)
  research_notes.md            sources and findings (exists)
  proposal.md                  this file
  requirements.txt             pybaseball, pandas, pyarrow, lightgbm, torch, gradio, matplotlib, scikit-learn
  run-in-colab.ipynb           thin notebook: install, mount Drive, call into pitchpred/
  pitchpred/
    data.py                    download by day, cache to parquet, schema checks
    clean.py                   drop rules, pitch-type mapping, position-player flag, row accounting
    coords.py                  normalize/mirror location; 26-cell grid
    features.py                as-of arsenal, batter, head-to-head, sequence tokens
    splits.py                  splits A, B, C
    baselines.py               B0-B3
    gbdt.py                    M1
    seqmodels.py               M2 GRU, M3 transformer, location heads (grid, then MDN)
    metrics.py                 log loss, skill, calibration, bootstrap over games
    demo.py                    Gradio app
  tests/
    test_leakage.py            allowlist, truncation invariance, sort order
    test_coords.py             grid and mirroring sanity
  report/
    report.md                  layered write-up
    figures/
  data/                        git-ignored
```

## 12. Plan in phases (each phase leaves a usable result)

| Phase | Work | If we stop here, we have |
|---|---|---|
| 0 | Download 2023-2026 to parquet; explore; confirm columns, pitch codes, position-player rule, season completeness | Clean dataset and a data summary |
| 1 | Cleaning, coordinates, splits, metrics, B0-B3, leakage tests | A defensible baseline study with strong baselines |
| 2 | Pitcher/batter/head-to-head features; M1 (LightGBM) for type and grid location | A solid predictive model and the main numbers |
| 3 | M2 GRU with grid location head; compare to M1 | The sequence-vs-tabular answer |
| 4 | M3 transformer; MDN location head (stretch) | Course-relevant architecture and the continuous-density version |
| 5 | Analyses (per-pitcher skill, calibration, postseason, hitter payoff); Gradio demo; report | The full deliverable |

**Deadline unknown.** Phases 0-2 are the minimum credible project; Phase 5's demo and report can be built from Phase 2 results if time runs short.

## 13. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Subtle leakage inflates results | High (the common failure) | §8; truncation test; report the test, not just the result |
| Downloads slow or pybaseball breaks | Medium | Daily chunks, parquet cache, direct-CSV fallback |
| Gains over B1 are tiny | High | Framed in §2; the story is the variation by pitcher and situation, plus calibration and location |
| Rare pitch types make unstable classes | Medium | Frequency floor, `OTHER` class, label smoothing if needed |
| MDN location head unstable | Medium | Grid head first; MDN is stretch |
| Per-pitcher estimates noisy | High | Shrinkage; minimum pitch counts; bootstrap intervals; show uncertainty |
| 2026 shift (ABS) hurts time-split results | Low-Medium | Report 2025 validation and 2026 test separately; turn the shift into an analysis |
| Scope creep | High | Phases; each is independently shippable |
| Colab session limits | Medium | Checkpoint to Drive; small models; cache features as parquet |

## 14. Open questions for Austin

1. **Deadline and expected scope.** Check the syllabus or course site. It sets how far into Phase 4-5 we go.
2. **Solo or team?** Affects how we split the work and what the report credits.
3. **Burn-in season (2023):** OK to download for feature warm-up? (Recommended; costs about one more season's download.)
4. **Postseason default** (train regular-season only; postseason held-out): confirm.
5. **Report format:** Markdown/PDF, a Google Doc, or something the course specifies?
6. **Colab plan:** free tier is probably enough for M1-M3; Pro makes long runs less fragile.

## 15. What "full prototype" will mean when you say it

I'll start at Phase 0 and work through the phases in order: write the package and tests, run the downloads and trainings in Colab (you'll need to run the notebook there, since I can't reach your Colab session from here), and produce results tables, figures, the demo, and a draft report, pausing for your review at the end of each phase. I'll explain each ML piece as I go.
