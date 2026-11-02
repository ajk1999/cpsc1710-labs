# Research notes: pitch type + location predictor

Date: 2026-11-02. Compiled from web search results. **Caveat:** I read search-result summaries, not full papers or pages, so items marked (unverified) should be checked before they go in the report. Items marked (verify in Phase 0) are things the first data-exploration step will settle.

## 1. Prior work: what accuracy is realistic?

| Source | Setup | Reported result | Trust |
|---|---|---|---|
| ["The count tells you everything"](https://calledthird.pages.dev/analysis/the-count-tells-you-everything) (blog) | XGBoost, 2025 Statcast, ~730k pitches, 1,081 pitchers | 39.61% top-1 vs 37.47% most-common-pitch baseline vs **39.09% count-conditional baseline**. Author says the first result was inflated by leakage, found by auditing. | Blog; single author; but the leakage story is the most useful lesson here |
| [Follow-up: "which pitchers can you predict"](https://calledthird.pages.dev/analysis/which-pitchers-can-you-predict) (same blog) | Pooled XGBoost | 43.83%, about +1.7 points over baseline; most pitchers not reliably predictable beyond count; predictability falls as repertoire gets more complex | Blog |
| [FanGraphs community: "No Pitch Is an Island"](https://community.fangraphs.com/no-pitch-is-an-island-pitch-prediction-with-sequence-to-sequence-deep-learning) | Seq2seq LSTM with attention, predicting pitch pairs | ~46% / ~48% accuracy vs ~35% four-seam rate | Community post; baseline is the weakest one (always-fastball) |
| [Attention-LSTM, ICMEW 2022](https://scholars.ncu.edu.tw/en/publications/decide-the-next-pitch-a-pitch-prediction-model-using-attention-ba/) | 201 pitchers, 2016-2021 | 76.7% average accuracy | Suspiciously high; evaluation protocol unclear (unverified). Don't use as a target |
| [JSAI 2026 presentation](https://pub.confit.atlas.jp/en/event/jsai2026/presentation/2Yin-A-45) | 2017-2024 Statcast, time-blocked CV | Transformers best overall; tree ensembles degrade on sparse joint classes | Abstract only (unverified) |
| [UNB thesis 2020](https://unbscholar.lib.unb.ca/handle/1882/14075) | RF, SVM, LSTM on Statcast | LSTM needed more feature engineering to beat simpler classifiers | Thesis |

**Takeaways**
1. The count-conditional lookup table is a **very strong baseline**. A model that adds only ~0.5 points of top-1 accuracy over it is normal, not a failure.
2. Big accuracy numbers (70%+) almost always trace to leakage or a weak baseline. Our credibility comes from the leakage tests and strong baselines (see proposal).
3. Top-1 accuracy hides a lot. Log loss and calibration will show differences accuracy can't, and are what a hitter actually needs (probabilities, not a single guess).
4. Sequence models may help most where pitchers have big arsenals, since the count alone says little there.
5. None of the above predict **location**; that part is more novel and less benchmarked.

## 2. Data: Statcast via pybaseball

- [pybaseball](https://github.com/jldbc/pybaseball): `statcast(start_dt, end_dt)` takes `YYYY-MM-DD` dates. Per-player variants: `statcast_pitcher`, `statcast_batter`.
- **Caching:** off by default; turn on with `pybaseball.cache.enable()`. Cache lives in `~/.pybaseball/cache`, overridable with the `PYBASEBALL_CACHE` env var. It matches **exact calls**, so overlapping date ranges re-download. Future dates can cache empty results (`cache.purge()` fixes it).
- **Baseball Savant row cap:** about 25,000 rows per download ([R docs](https://beanumber.github.io/abdwr3edata/reference/statcast_daily.html), [textbook](https://beanumber.github.io/abdwr3e/12-large.html)); one user guesses closer to 30k, so treat the exact number as uncertain. A week usually exceeds it, so **pull one day at a time** (pybaseball's `statcast()` chunks internally; verify in Phase 0).
- **Speed:** anecdotal reports of about an hour for a season (unverified). Plan: download once per season, save to parquet on Google Drive, never re-download.
- **Column semantics** from [Savant CSV docs](https://baseballsavant.mlb.com/csv-docs) and the [Statcast reference appendix](https://beanumber.github.io/abdwr3e/C_statcast.html):
  - `plate_x`: horizontal position at the plate, catcher's view, feet. `plate_z`: height, feet.
  - `zone`: Statcast zone number, catcher's view.
  - `sz_top` / `sz_bot`: top and bottom of the batter's zone, in feet, set by the operator when the ball is halfway to the plate.
- **Pitch codes** (sources disagree on rare ones; verify against the CSV docs in Phase 0): common: FF, SI, FC, SL, ST (sweeper, added 2023), SV (slurve), CU, KC, CS, CH, FS, FO, KN, EP, SC. Non-pitches or junk to drop: PO/FO-style pitchouts, IN (intentional ball), AB/AS (automatic ball/strike from pitch-clock violations), UN (unidentified), nulls.
- **Season availability:** the 2026 season, including the postseason, appears to be finished (World Series reportedly concluded late October 2026). So the three most recent seasons are **2024, 2025, 2026**. Verify by checking that the last pitch dates in a pull are late October 2026.

## 3. Regime changes the model should know about

- **2023 onward:** pitch clock, shift ban, bigger bases. Sweeper (`ST`) classification introduced. Pre-2023 data is a different environment, so we stay post-2023.
- **2026:** the [ABS challenge system](https://www.mlb.com/news/abs-challenge-system-2026-mlb-season-overview): human umpires call every pitch, each team can challenge two calls per game, using Hawk-Eye (same tracking as Statcast). This could slightly change how pitchers attack the edges of the zone. Treat it as a possible distribution shift between 2025 and 2026, and **a story**: "did pitchers change location after ABS?" (stretch analysis).
- Statcast's `zone` and `plate_x/z` are measured geometry, not umpire calls, so the labels themselves are unaffected.

## 4. Leakage: how pitch-prediction projects typically go wrong

(Synthesized from the blog above and general practice.)

1. **Current-pitch columns as features.** Anything measured on the pitch being predicted: `release_speed`, `release_spin_rate`, `pfx_x/z`, `plate_x/z`, `zone`, `description`, `type`, `events`, `launch_*`, `estimated_*`, `delta_*`, `post_*_score`, and so on. All of these are post-pitch.
2. **Wrong row order.** Savant CSVs come newest-first. Features like "previous pitch" computed without sorting by `game_date, game_pk, at_bat_number, pitch_number` use the future.
3. **Pitcher-level aggregates that include the test pitch.** A "pitcher's usage by type" over the full season includes the pitch you're predicting. Must be **as-of** (only earlier pitches).
4. **Random split by pitch.** Neighbouring pitches in the same at-bat or game land on both sides. Split by date.
5. **Normalization or encoders fit on all data.** Fit on train only.
6. **Identity memorization mistaken for skill.** Held-out-pitcher test guards against this.

## 5. Practical environment notes

- Local machine: Python 3.12, numpy only. Colab is the compute plan. Install in Colab with `pip install pybaseball` (plus LightGBM, PyTorch is preinstalled there).
- Raw Statcast data should **not** be committed to git (size, and MLB data terms). Keep `hw6/data/` in `.gitignore`; check the Savant/MLB usage terms before redistributing anything derived. Coursework use is normal; a public demo should ship predictions and aggregates, not raw rows.
- pybaseball is a community scraper that can break when Savant changes. Fallback: request Savant's CSV endpoint directly, one day per request.
