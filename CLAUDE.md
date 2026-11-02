# CPSC 1710 labs and homework

Austin's working repo for CPSC 1710 (Fall 2026). Forked from the course's `labs` repo; `upstream/main` is the course repo.

## Layout
- `lab-01` … `lab-05`, `index.html`, `docs/`, `assets/`: course-provided material. Don't edit; pull from upstream.
- `hw4-starter/`: tiny character-level RNN (`tiny_rnn.py`), tokenizers, generated samples.
- `hw5/`: bigram language model (`bigram-code.py`, `bigram-pseudocode.txt`), `gpt_dev.ipynb`, `input.txt` training text.
- `hw6/`: course project: **pitch type + location predictor** from Statcast (via `pybaseball`). **Start with `hw6/HANDOVER.md`** (current state, open threads, what's unverified). Raw interview notes in `hw6/project_brainstorming.md`; `hw6/human_notes/` holds Austin's own writing (don't edit). Phase: interview, research (`hw6/research_notes.md`) and proposal (`hw6/proposal.md`) are done; Austin is reviewing. Don't build until Austin says "full prototype"; then start at proposal Phase 0 and follow its phases, repo layout and leakage controls (allowlist, as-of features, truncation-invariance test). Key facts: seasons 2024-2026 (2023 burn-in for features only), regular season only for training, postseason held out; count-conditional baseline is strong, so report log loss/calibration, not just accuracy.
- `skills/grill-me`: midterm practice skill.

## Working conventions
- Python scripts are run directly (`python tiny_rnn.py`); deps are in `hw4-starter/requirements.txt`. Colab is an alternative (`run-in-colab.ipynb`).
- Shell is PowerShell on Windows; Git Bash is also available.
- This is coursework: explain what code does and why, and keep changes small and readable so Austin can follow them. Don't write whole solutions unprompted when an assignment asks Austin to write the code or pseudocode.
- hw6 project decisions: predict pre-pitch only (hitter's view), no leakage; Colab compute; 3 most recent Statcast seasons; time split primary; deliverables are an interactive demo plus a report readable by non-expert fans first, with statistical detail layered on. Austin is strong on baseball, moderate on ML: explain ML choices plainly.
- Don't modify course-provided files or `input.txt` unless asked.
- Commit only when asked. Merge upstream rather than rewriting history.

## Keeping this file current
- Add a line when Claude gets something wrong twice or a convention isn't obvious from the code.
- Delete lines that go stale. Keep the file under ~100 lines.
