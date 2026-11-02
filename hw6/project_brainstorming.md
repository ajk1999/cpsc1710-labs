# Project Brainstorming: Pitch type predictor

Raw interview notes. Appended as we go; not polished. Deep research and the prototype/proposal go in separate files in this folder.

## Initial pitch (Austin, verbatim summary, 2026-11-02)
- **Project:** Pitch type predictor. Austin likes baseball; self-described as fun, "not exactly groundbreaking."
- **What:** Given game context, predict probability of each pitch type and location for the next pitch.
- **Why:** Pitching is so good now (100 mph fastballs tunneling with 90 mph sliders, sweepers with ~35 in. horizontal break) that hitters often sit on specific pitches in specific zones they can drive instead of reacting. Better prediction -> better hitting. (Teams already have far more sophisticated models.)
- **How:**
  - Data: millions of labeled pitches, free via `pybaseball` (MLB Statcast).
  - Treat an at-bat as a sequence of pitches, like the bigram case; try a small transformer or RNN over pitch history.
  - Two outputs: P(pitch type) and P(location). Location = classic 3x3 zone grid plus a ring around the zone ("shadow"). Since the two are correlated, simplest is location | type.
  - Baselines: middle-middle fastball only (weak); more interesting is league-average pitch selection and location.
  - Evaluation idea: hold out games per pitcher and compare to expected output. Not yet worked out.

## Interview log
(Q&A appended below)

### Round 1 (goals / context)
- **Goal:** "All of the above" with flavors of everything (rigorous eval, interpretability, seq model comparison). Primary = **working prototype**. Doesn't need to *prove* a sequence model wins, but expects it would.
- **User of predictions:** A hitter pre-pitch (implies calibrated probabilities, only information available before the pitch; no leakage).
- **Deadline/size:** Unknown. No project spec found in the repo (searched). TODO: Austin to check the course site/syllabus.
- **Compute:** Colab (free/Pro). Local machine has Python 3.12, numpy only (no torch/pybaseball installed yet).

### Round 2 (inputs / splits / data size)
- **Inputs (all pre-pitch):** count, outs, base state (+ inning, score diff); batter identity/handedness; pitcher identity; prior pitches in the at-bat (type, location, velo, result).
- **Pitcher handling:** **Arsenal stats as input** (historical pitch mix/velo/movement), so unseen/new pitchers can be handled. Need to compute these from *past* data only (no leakage from the test period).
- **Eval splits:** **Several**: time split as primary, plus held-out-games and held-out-pitchers analyses.
- **Data size:** **2-3 seasons** (~2M pitches).

### Round 3 (outputs / models / metrics)
- **Location:** **Continuous (x,z) density** (e.g. mixture of Gaussians conditioned on pitch type), not the grid. Implication: evaluate with NLL of the density; can still *report* on a 3x3+shadow grid by integrating the density, for readability.
- **Pitch types:** **Full Statcast `pitch_type` codes** (FF, SI, SL, ST, CU, CH, FC, FS, KC, ...). Rare classes (EP, SC, FO, etc.) will need handling (merge into "other" or drop).
- **Models:** Austin asked for a recommendation. Recommended ladder: (1) league-average, (2) league-average by count x handedness, (3) pitcher's own historical mix (arsenal) by count, (4) GBDT/logistic on the same tabular features, (5) GRU over at-bat, (6) small causal transformer. Comparing 4 vs 5/6 isolates whether *sequence* helps.
- **Metrics (all four):** log loss/perplexity (primary), calibration, per-pitcher skill vs baseline, top-1/top-3 accuracy.

### Round 4 (history / deliverables / working mode)
- **History:** **Current at-bat sequence + prior-meeting summary** (summary stats of earlier PAs by this pitcher vs this batter, e.g. same game or season). Must be built from strictly earlier pitches.
- **Deliverables:** **Both an interactive demo and a written report.** The report must be readable at a baseline for non-intense fans/scientists first; statistical nuance layered on top (e.g. plain-language main text, technical appendix/sidebars).
- **Experience:** Baseball domain strong, ML moderate -> explain ML choices plainly; lean on his baseball intuition for features and sanity checks.
- **Working mode:** For now, **propose only** (research + proposal). Austin will later say "full prototype"; at that point I should be ready to build end to end. So the proposal should be concrete enough to execute directly (file layout, data pipeline, model code plan).

### Round 5 (data scope / success / worries)
- **Seasons:** **Most recent 3 available** (verify in research whether 2026 is complete; today is 2026-11-02). All are post-2023 pitch clock if 2024-2026, which is a consistent regime.
- **Edge cases:** Drop **position players pitching**. (Intentional balls, pitchouts, spring training, postseason: not decided; default proposal = drop intentional balls/pitchouts, regular season only, and say so in the report.)
- **Success:** **Compelling demo/story** (fan-interesting insights, even if gains are modest). Still keep honest metrics, per the earlier "all of the above."
- **Worries:** **Data leakage** (top concern), plus "whatever else is realistic." My list to cover in the proposal:
  - Leakage: arsenal and prior-meeting features computed only from data before the prediction date; time split must respect this; same-game info only from earlier pitches.
  - Pitch-label ambiguity: Statcast reclassifies pitch types over time (e.g. sweeper `ST` added 2023); rare classes.
  - Scraping/download time and rate limits: cache to parquet once.
  - Mixture-density training instability; fall back to a grid for the first prototype.
  - Pitcher-level variation: single-pitcher sample sizes small; use shrinkage toward league average.
  - Intrinsic unpredictability: realistic gains over count-conditioned baselines are likely small; frame the story around where predictability is high (e.g. 3-0, 0-2, pitcher tells).
  - Scope creep: lock a minimum viable path first.

## Status
Interview done through round 5. Next: Austin asks for deep research, then proposal. Then "full prototype" on request.

### Follow-up: postseason
- Austin curious about postseason. Recommendation: **no separate postseason model** (~15-20k pitches vs ~700k regular season; overfits). Default: train on regular season only, **hold out postseason as a stress test** ("does the model transfer to October?"). Optional later: `is_postseason` flag, compare vs regular-only on the postseason set. Caveat: selection effects (only strong teams/pitchers); control by comparing each pitcher to his own regular-season mix.
- Awaiting Austin's confirmation of this default.
