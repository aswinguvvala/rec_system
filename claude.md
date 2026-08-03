# CLAUDE.md

## Project
Hybrid Movie Recommendation System — a portfolio project demonstrating production ML engineering practices: a real data pipeline, hybrid content-based + SVD collaborative filtering, proper ranking/rating evaluation, cold-start handling, and a deployed live demo. Primary audience: technical recruiters and hiring managers reviewing the GitHub repo and the live demo link.

## Stack
- Python 3.12 (3.11 unavailable on dev machine; all deps fully support 3.12)
- pandas==2.2.3, numpy==2.5.1, scikit-learn==1.9.0, scipy==1.18.0 (see `requirements.txt`)
- SVD: manual Funk-SVD (SGD + user/item biases + L2 regularization) via numpy/scipy —
  chosen over `surprise` (unmaintained, frequent build failures on Windows/free hosting
  tiers) and `implicit` (built for implicit-feedback ALS, wrong fit for explicit 1-5
  star ratings). No extra heavy dependency, full control, more interesting to discuss
  in an interview.
- Streamlit for `app.py`
- pytest for `tests/`

## Data pipeline status
- Phase 1 complete and verified: `python -m src.data_pipeline` downloads (cached
  thereafter), cleans, and splits MovieLens 100K into
  `data/processed/{train,test,movies,users}.csv`.
- Split strategy: per-user chronological split (80/20), users with <5 ratings kept
  entirely in train. See `chronological_train_test_split` docstring in
  `src/data_pipeline.py` for the full justification (avoids leakage from a random
  split; per-user rather than global so every active user has test coverage).
- Verified: 100,000 ratings, 1,682 movies, 943 users -> 80,367 train / 19,633 test rows.

## Models status (Phase 2)
- `src/models.py` complete and smoke-tested: `BaseRecommender` ABC (`fit`/`predict`/
  `recommend_for_user`) implemented by `ContentBasedRecommender` (genre one-hot +
  cosine similarity, no TF-IDF since ml-100k has no text/plot field),
  `SVDRecommender` (manual Funk-SVD, SGD, 50 factors/20 epochs, ~12s fit time, train
  RMSE 1.03->0.78), `PopularityRecommender` (IMDB-style Bayesian weighted rating;
  doubles as the Phase 3 baseline), `HybridRecommender` (weighted or switching combo
  of content+SVD), and `ColdStartRecommender` (wrapper: users with <5 train ratings
  -> genre-matched popularity; items with 0 train ratings -> global popularity).
  Every result carries a `source` tag (`content`/`svd`/`hybrid`/`cold_start`) for the
  app's side-by-side comparison view.
- Verified against real data: cold-item path triggers correctly for the 67 catalog
  movies with zero training ratings; cold-user path verified with a synthetic
  2-rating user.

## Evaluation status (Phase 3)
- `src/evaluate.py` complete and run: `python -m src.evaluate` fits all 4 models on the
  real train/test split, evaluates RMSE/MAE + Precision/Recall/NDCG@{5,10} on the full
  1682-movie catalog (no negative sampling -- harder but honest), and writes
  `results/metrics.json`. Real numbers, not placeholders.
- Headline result: SVD wins RMSE/MAE (0.987 / 0.781); Hybrid (weighted, alpha=0.6)
  wins ranking quality (best Precision@5 0.057 and NDCG@5 0.064 of all 4 models).
  Content-based alone is weak on absolute rating calibration (RMSE 1.42) because
  cosine similarity -> [1,5] scaling is crude, but still contributes enough
  complementary signal that blending it with SVD improves ranking over SVD alone.
  This is the real, defensible "why hybrid" story for the README.
- Full table is in `results/metrics.json`; don't hand-copy numbers into the README,
  read the file so it can't drift out of sync with a re-run.

## App status (Phase 4)
- `app.py` complete. Auto-runs the pipeline on first load (`load_data` -> cached via
  `st.cache_data`), fits all 5 models once (`load_models` -> `st.cache_resource`:
  content, svd, popularity, hybrid, cold_start-wrapped-hybrid). Sidebar: pick a real
  user (or "simulate a brand-new user" via sentinel `user_id=-1` -- real ml-100k users
  all have 20+ ratings so this is the only way to trigger the cold-start path live),
  N-recommendations slider, "compare side-by-side" toggle. Every recommendation shows
  its `source` badge. Bottom of page renders `results/metrics.json` as a table.
- Verified with Streamlit's `AppTest` headless harness (browser extension was declined
  this session, so no literal screenshot) -- confirmed: title/recs render for the
  default user; side-by-side mode renders all 4 columns with distinct results per
  model; toggling the simulated cold-start user in side-by-side mode correctly shows
  the raw content/SVD/hybrid columns empty ("no cold-start handling in this raw
  model") while Popularity Baseline still works, and single-view mode shows the
  `cold_start`-tagged fallback recommendations; metrics table renders real numbers.
  No exceptions in any state.
- Fixed one real bug found during testing: `st.dataframe(..., use_container_width=True)`
  triggered a deprecation warning in streamlit==1.60.0 (past its stated removal date) --
  changed to `width="stretch"`.
- Gotcha for any future headless testing: `streamlit.testing.v1.AppTest.from_file()`
  does NOT go through `streamlit/web/bootstrap.py`, so it never inserts the project
  root into `sys.path` the way `streamlit run app.py` does -- a test script must
  `sys.path.insert(0, os.getcwd())` itself before importing `AppTest`, or `from src...`
  imports inside `app.py` fail with `ModuleNotFoundError` (this is a test-harness-only
  issue; the real `streamlit run` server is unaffected).

## Tests status (Phase 5)
- 41 tests across `tests/test_data_pipeline.py`, `tests/test_models.py`,
  `tests/test_evaluate.py`. All green via both `pytest tests/ -v` and
  `python -m pytest tests/ -v`.
- Added `pyproject.toml` with `[tool.pytest.ini_options] pythonpath = ["."]` so bare
  `pytest` (not just `python -m pytest`) resolves `from src...` imports -- without it,
  pytest's default import-mode only puts `tests/` itself on `sys.path`, not the
  project root (same class of issue as the `AppTest` sys.path quirk above).
- Explicitly covers the required cases: cold-start fallback triggers correctly for
  both under-threshold users and zero-rating items (`TestColdStartRecommender`), and
  ranking/rating metrics match hand-computed values on small known examples
  (`tests/test_evaluate.py`).
- Real bug caught during test-writing: the shared `movies_df`/`train_df` fixture only
  has 5 movies, and one fixture user rates all 5 -- an early version of the
  "above-threshold uses base model" cold-start test passed vacuously against an empty
  recommendation list. Fixed by giving `TestColdStartRecommender` its own 6-movie
  fixture with one movie nobody rates, guaranteeing a real unseen candidate exists.

## Polish status (Phase 6)
- `README.md` written: problem statement, Mermaid architecture diagram, real metrics
  table (pulled from `results/metrics.json`, not placeholders), Design Decisions
  section (SVD choice, split choice, cold-start wrapper, "why hybrid despite SVD
  winning RMSE"), run-locally instructions, project structure, tech stack.
- Screenshot NOT captured -- browser extension access was declined this session, so
  there was no way to take one. README says so explicitly rather than faking it or
  omitting the caveat. Revisit with `/chrome` if a real screenshot is wanted later.
- Fixed `.gitignore`: removed the `results/*.json` exclusion. That file is small and
  is the actual "real numbers" artifact the README and a fresh deploy both depend on
  -- gitignoring it would force every fresh clone/deploy to either show no metrics or
  manually rerun `python -m src.evaluate` before the app looks right.
- "Clean clone" verified as best as possible without git (git is not installed on
  this machine, confirmed in Phase 0 and still true): mirrored only the
  git-trackable files (respecting `.gitignore`) into a fresh directory, created a new
  venv there, ran `pip install -r requirements.txt` fresh, ran `pytest tests/ -v`
  (41 passed), then launched `streamlit run app.py` against a completely empty
  `data/` and drove it with `AppTest` -- it auto-downloaded MovieLens 100K, built
  `data/processed/`, trained all 5 models, and rendered with zero exceptions. This
  covers everything git itself would add. Real `git clone` still needs to be verified
  once git is installed and the repo is actually pushed somewhere.
- Environment quirk hit during this: the sandboxed scratchpad temp path was too deep
  for Windows `MAX_PATH`, breaking `pip install` (`WinError 206`). Worked around by
  using a short path directly under the user's home directory instead.

## Known environment quirks
- pandas 3.0.5's compiled Cython DLLs were blocked by this machine's Windows
  Application Control policy on install (numpy/scipy were unaffected). Pinned to
  pandas==2.2.3 instead, which installs and imports cleanly.
- Git is not installed on this dev machine, so the repo has not been `git init`'d yet.
  Do not assume git commands will work until this is resolved.

## Commands
- Run app: `streamlit run app.py`
- Run tests: `pytest tests/ -v`
- Run data pipeline standalone: `python -m src.data_pipeline`
- Lint: `ruff check src/`
- Install deps: `pip install -r requirements.txt`

## Conventions
- Type hints on every function signature
- Google-style docstrings on all public functions/classes
- `logging` module, never `print`, anywhere in `src/`
- try/except with specific exception types and actionable messages around all I/O (downloads, file reads)
- Single-responsibility modules: `data_pipeline.py` never contains model logic; `models.py` never does I/O
- Any new feature gets a corresponding test in `tests/` before it's considered done

## Architecture
- `src/data_pipeline.py` — download/cache MovieLens 100K, clean, split
- `src/models.py` — `ContentBasedRecommender`, `SVDRecommender`, `HybridRecommender`, cold-start fallback wrapper
- `src/evaluate.py` — RMSE/MAE plus Precision@K, Recall@K, NDCG@K
- `app.py` — Streamlit demo; must run standalone from a fresh clone with no manual setup beyond `pip install`

## Do not
- Don't commit `data/` (raw MovieLens files) — it's gitignored and re-downloaded on first run
- Don't hardcode file paths — use `pathlib`, relative to project root
- Don't add a dependency without pinning its version in `requirements.txt`

## Definition of done
Code runs without errors, has tests, has type hints and docstrings, and logs meaningfully instead of printing.