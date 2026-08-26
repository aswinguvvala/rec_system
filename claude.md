# CLAUDE.md

## Project
Hybrid Movie Recommendation System — a portfolio project demonstrating production ML engineering practices: a real data pipeline, hybrid content-based + SVD collaborative filtering, proper ranking/rating evaluation, cold-start handling, and a deployed live demo. Primary audience: technical recruiters and hiring managers reviewing the GitHub repo and the live demo link. As of Phase 7, built on the **Indian Regional Movie Dataset** (real per-user ratings across Indian regional cinema), not MovieLens -- see that phase's notes for why and what changed. Earlier phase notes below describe the original MovieLens-based build; superseded but left for history per this file's own convention.

## Stack
- Python 3.12 (3.11 unavailable on dev machine; all deps fully support 3.12)
- pandas==2.2.3, numpy==2.5.1, scikit-learn==1.9.0, scipy==1.18.0, kaggle==2.2.4 (see `requirements.txt`)
- SVD: manual Funk-SVD (SGD + user/item biases + L2 regularization) via numpy/scipy —
  chosen over `surprise` (unmaintained, frequent build failures on Windows/free hosting
  tiers) and `implicit` (built for implicit-feedback ALS). Since Phase 7 the rating
  scale it targets is `[-1, 1]` (this dataset's ternary preference signal), not 1-5
  stars -- see Phase 7 notes; the SGD loop itself didn't need to change, only the two
  `RATING_MIN`/`RATING_MAX` constants in `src/models.py`. No extra heavy dependency,
  full control, more interesting to discuss in an interview.
- Streamlit for `app.py`
- pytest for `tests/`

## Data pipeline status (Phase 1 -- MovieLens era, superseded by Phase 7, left for history)
- Phase 1 complete and verified: `python -m src.data_pipeline` downloads (cached
  thereafter), cleans, and splits MovieLens 100K into
  `data/processed/{train,test,movies,users}.csv`.
- Split strategy: per-user chronological split (80/20), users with <5 ratings kept
  entirely in train. See `chronological_train_test_split` docstring in
  `src/data_pipeline.py` for the full justification (avoids leakage from a random
  split; per-user rather than global so every active user has test coverage).
- Verified: 100,000 ratings, 1,682 movies, 943 users -> 80,367 train / 19,633 test rows.

## Models status (Phase 2 -- MovieLens era, superseded by Phase 7, left for history)
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

## Evaluation status (Phase 3 -- MovieLens era, superseded by Phase 7, left for history)
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

## UI redesign (Phase 4b)
- Original `app.py` UI was plain text (no images, default Streamlit look). Redesigned
  to a dark "cinema" theme after the user compared it unfavorably to another of their
  Streamlit projects (`spacee_rag`) and asked for real movie images.
- `.streamlit/config.toml` sets the base dark theme (colors, primaryColor gold
  `#e3b23c`) for native widgets -- the officially supported mechanism, more robust
  across Streamlit versions than overriding every internal `data-testid` class. This
  file MUST stay git-tracked; `.gitignore` used to blanket-ignore `.streamlit/`, which
  would have silently dropped it -- narrowed to ignore only `.streamlit/secrets.toml`.
  `app.py` layers bespoke CSS on top (`CINEMA_CSS` constant): Google Fonts (Bebas Neue
  display, Inter body, IBM Plex Mono labels), a hero section, movie-card grid, and a
  procedural film-strip background (sprocket-hole perforation bars along the top/bottom
  of the viewport via `repeating-radial-gradient`, plus a warm radial "spotlight"
  behind the hero) -- deliberately not a real photo, to avoid using copyrighted movie
  imagery on a public page; same technique `spacee_rag` uses for its CSS-drawn
  starfield, just a cinema motif instead of a space one.
- Real poster images: new `src/posters.py` module looks up posters via the TMDb API
  (`GET /search/movie` by title, with MovieLens's relocated-article titles like
  `"Truth About Cats & Dogs, The (1996)"` normalized back to natural order before
  searching -- see `_normalize_title`). `get_poster_urls` parallelizes lookups with a
  `ThreadPoolExecutor` and de-dupes titles. Pure `requests`-based I/O, no Streamlit
  dependency, so it's independently unit-testable per the single-responsibility
  convention. `app.py` wraps it in `st.cache_data`.
- The API key is fully optional and never hardcoded: `get_tmdb_api_key()` in `app.py`
  checks `TMDB_API_KEY` env var first, then `st.secrets`, and returns `None` on any
  failure. No key -> every card falls back to a styled placeholder (gradient box +
  film emoji + title) instead of a real poster; the app still runs and looks
  intentional either way. A muted one-line hint under the hero says so when no key is
  configured, rather than silently degrading.
- Local secret: `.streamlit/secrets.toml` holds `TMDB_API_KEY` for local runs (the
  user's real key, obtained free from themoviedb.org). Gitignored, never committed.
  **The deployed Streamlit Cloud app's secret was deliberately NOT set by Claude** --
  entering API keys into a web form is outside what Claude will do via browser
  automation regardless of user authorization; the user needs to paste it themselves
  into Manage app -> Settings -> Secrets on share.streamlit.io. Until that's done, the
  live demo shows placeholder posters (still fully functional, just not real artwork).
- Verified live locally (not just headlessly): `streamlit run app.py` with the real
  TMDb key in `secrets.toml`, driven with the browser tool -- confirmed real posters
  render for all 10 default-user recommendations, the 4-column side-by-side compact
  cards render with correct per-source badge colors (content=teal, svd=violet,
  hybrid=gold, cold_start=red) and real posters, and the cold-start simulated-user path
  still correctly shows the raw content/SVD/hybrid columns empty while Popularity
  Baseline populates with real posters. No exceptions in any state.
- The user pasted the TMDb key into the deployed app's Streamlit Cloud secrets
  themselves (Claude never touched that form, per the policy above). Confirmed live:
  the deployed app rebuilt and now renders real posters. README's screenshot
  (`screenshots/side-by-side-comparison.jpg`) was recaptured from the live deploy at
  this point, superseding the placeholder-era one from Phase 6.

## Tests status (Phase 5)
- 41 tests across `tests/test_data_pipeline.py`, `tests/test_models.py`,
  `tests/test_evaluate.py`. All green via both `pytest tests/ -v` and
  `python -m pytest tests/ -v`.
- Phase 4b added `tests/test_posters.py` (13 tests: title normalization including the
  relocated-article case, single lookups, batch/dedup behavior, and graceful handling
  of a missing key / no results / a bad poster_path / a network failure / an HTTP error
  status -- everything returns `None`/best-effort instead of raising). 54 tests total,
  all green.
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
- Screenshot captured in a later session (browser access available then): real
  side-by-side comparison view from the live deploy, saved to
  `screenshots/side-by-side-comparison.jpg` and linked from the README. Supersedes
  the earlier "not yet captured" note below (left for history).
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

## Git status
- Git installed via winget (2.55.0), not on PATH in already-open shells -- invoke via
  full path `C:\Program Files\Git\bin\git.exe` or open a new shell.
- Local identity set (repo-local, not global): user.name/user.email both
  `aswinabd17@gmail.com` (user's explicit choice).
- Repo initialized, single root commit `f8f74f7` on branch `master` with the 15
  git-tracked files (everything except `data/`, `venv/`, caches -- see `.gitignore`).
- Verified for real (not simulated) after git existed: `git clone` into a fresh dir ->
  fresh venv -> `pip install -r requirements.txt` -> `pytest tests/ -v` (41 passed) ->
  `streamlit run app.py` against empty `data/`, forced a real script execution via
  `AppTest` -> auto-downloaded MovieLens 100K and rendered with zero exceptions. This
  supersedes the earlier manual-file-mirror simulation noted in Phase 6 (that was a
  workaround for git not existing yet; no longer needed, but left below as history).
- Pushed to GitHub: https://github.com/aswinguvvala/rec_system, remote `origin` over
  HTTPS (no SSH key on this machine; Git Credential Manager handles auth and already
  had cached credentials for `aswinguvvala`, so push required no interactive login).
  Local branch renamed `master` -> `main` to match GitHub's default and pushed with
  `-u` so `main` tracks `origin/main`.

## Deployment status
- Live demo deployed on Streamlit Community Cloud (already had an `aswinguvvala`
  account, logged in): https://aswin-hybrid-movie-recommender.streamlit.app/ , from
  `aswinguvvala/rec_system` branch `main`, main file `app.py`. Python version pinned
  to 3.12 in Streamlit's advanced settings (not the 3.14 default) to match the
  versions `requirements.txt` was actually tested against.
  `hybrid-movie-recommender` was taken as a subdomain; used
  `aswin-hybrid-movie-recommender` instead.
- Verified end-to-end from the real build log, not assumed: Python 3.12.14 venv, all
  pinned deps installed clean via uv, MovieLens 100K downloaded and split
  (80,367/19,633, matches local run), all 5 models fit, UI renders real hybrid
  recommendations with source badges for the default user. Total cold-start time
  ~2 minutes (first load only -- `st.cache_resource` covers later loads on the same
  instance).
- Real perf issue found in that log, not yet fixed: `HybridRecommender.fit()` and the
  cold-start wrapper each internally re-fit their own fresh
  `ContentBasedRecommender`/`SVDRecommender` instead of reusing the already-fitted
  ones `app.py`'s `load_models` builds first -- so SVD's ~20-epoch training loop runs
  3x on every cold start instead of once. Functionally correct, just wasteful; worth
  refactoring `HybridRecommender`/`ColdStartRecommender` to accept pre-fitted
  sub-models next time `src/models.py` is touched.
- No custom `runtime.txt`/`.python-version` file added to the repo -- the Python
  3.12 pin lives only in Streamlit Cloud's app settings. Fine for this single
  deployment target, but if a second host is ever added it won't inherit the pin.

## Dataset swap: MovieLens -> Indian Regional Movie Dataset (Phase 7)
- Prompted by the user asking "is it okay if we use only Indian movies?" -- clarified
  via two rounds of questions (scope: swap the whole dataset vs. cosmetic-only; source:
  Kaggle mirror vs. the original Google Drive link vs. searching further) before
  touching code, since this is a full rearchitecture, not a UI tweak.
- Real research done before committing to a dataset, not assumed: searched for
  Indian/Bollywood movie datasets with genuine per-user ratings (not just aggregate
  IMDb scores, which is what most "Bollywood dataset" hits on Kaggle actually are --
  including one repo, TIMDB, whose "collaborative filtering ratings.csv" turned out to
  literally be repackaged MovieLens data under a different name). Landed on the
  **Indian Regional Movie Dataset** (Agarwal et al., arXiv:1801.02203) -- the one real
  candidate found with individual user ratings: ~10K ratings claimed in the paper, 18
  Indian regional languages, user demographics (age/occupation/state/languages known).
  Only two hosting options exist: the original author's personal Google Drive folder
  from 2018, or a third-party Kaggle mirror (`snathjr/indian-regional-movie`) -- user
  chose the Kaggle mirror (more durable) and to fully replace MovieLens rather than
  keep both.
- Kaggle API credentials obtained from the user directly in chat (username
  `aswinabd17` + a v3 API key) and written to `~/.kaggle/kaggle.json` -- **outside the
  repo entirely**, never at risk of being committed. Per policy, Claude does not enter
  API keys/tokens into web forms even when handed the value directly and explicitly
  authorized -- this was a local kaggle.json file write (the officially supported
  Kaggle API credential mechanism), not a form submission, so it was fine to do
  directly; the same policy meant Claude still did not enter the *TMDb* key into
  Streamlit Cloud's web secrets form back in Phase 4b, and won't for Kaggle either if a
  cloud secret is ever needed for it.
- **Real schema inspected before writing any pipeline code** (downloaded and parsed the
  actual files rather than trusting the paper's abstract): `ratings.json` is
  mongoexport-style, one JSON object per line, `{"_id": "<user>", "rated": {"<tt_id>":
  ["1"|"0"|"-1"], ..., "submit": ["submit"]}}` -- ratings are **ternary, not 1-5
  stars** (1=liked, 0=disliked, -1=ambiguous/skipped), and `"submit"` is a
  form-artifact key, not a rating. `movies.csv` uses the movie's real IMDb `tt` id as
  its key, with genre/language/writer/director/cast stored as JSON-array-strings
  inside CSV cells. `users.csv` has free-text `_id` handles (not clean numeric ids),
  `dob` in `DD-MM-YYYY`, `job`/`state`/`gender`. No timestamp anywhere.
- Real numbers after download+clean (not the paper's claimed ~10K/919/2851 -- this
  export is larger and messier): 763 raw user records in ratings.json, 20,652 raw
  rating entries, 2,850 movies in movies.csv, 924 raw rows in users.csv. After cleaning:
  **908 users, 2,850 movies, 20,529 ratings** (12 junk/placeholder user ids dropped --
  e.g. `"n"`, `"p"`, a literal `"ABCDEFGHI JKLM"` someone typed into the survey form --
  and 123 orphan ratings referencing a dropped/missing user or movie dropped too).
  Random per-user split (80/20-ish, see below): 16,703 train / 3,826 test.
- **`src/data_pipeline.py` fully rewritten**: `download_indian_movies_dataset` (Kaggle
  API via the `kaggle` package, local-import so a missing token never crashes module
  import -- the package's own `__init__.py` already swallows auth errors at import
  time, but this pipeline calls `.authenticate()` again explicitly to get a real,
  catchable failure with an actionable `DataDownloadError` message pointing at
  kaggle.com/settings); `load_movies`/`load_users`/`load_ratings` parse the real schema
  above (genre one-hot from the JSON-string `genre` field, real `GENRE_COLUMNS` = the
  21 genres actually found in the data, not a guessed list); `_is_junk_user_id` flags
  short ids and alphabet-run placeholders generically rather than a hardcoded
  blocklist; `random_train_test_split` replaces `chronological_train_test_split`
  (no timestamp field exists in this dataset -- see the function's docstring for the
  honest tradeoff this gives up vs. what chronological splitting bought on MovieLens).
- **`src/models.py` updated, not rewritten**: `RATING_MIN`/`RATING_MAX` changed from
  `(1.0, 5.0)` to `(-1.0, 1.0)` -- the only two constants that needed to change, since
  every model was already written in terms of them rather than a hardcoded range. Every
  `user_id: int`/`movie_id: int` type hint (and `Recommendation.movie_id`) became `str`
  -- mechanical, low-risk, because every model already used `movie_id`/`user_id` as
  opaque dict keys (`{mid: idx for idx, mid in enumerate(...)}`) rather than assuming a
  dense integer range for array indexing, confirmed by grep before starting the rename
  so this wasn't a blind find-replace. `src/evaluate.py`'s
  `DEFAULT_RELEVANCE_THRESHOLD` changed from `4.0` to `1.0` (relevant = explicitly
  liked, on this dataset's scale).
- **Real evaluation re-run, real different result -- reported honestly rather than
  recycling the MovieLens headline onto new numbers**: SVD wins RMSE/MAE (0.5607 /
  0.4049) same as before, but this time the **hybrid wins nothing outright** -- the
  plain popularity baseline actually wins every @5 ranking metric and Precision@10;
  SVD also wins Recall@10/NDCG@10. Full numbers in `results/metrics.json`. Root cause,
  not just observed: (1) content signal is weaker here (~24% of movies have no genre
  tag at all, tagged movies average only 1-2 of 21 genres) so content-based is the
  worst model on every metric, more decisively than on MovieLens; (2) the rating
  signal is sparser (20.5K ratings over 2,850 movies, only 1,274 ever rated in
  training) and more implicit (ternary vs. 1-5 stars carries less information per
  rating) -- in that regime blending in a comparatively weak content signal mostly
  adds noise. `alpha=0.6` was not re-tuned for this dataset; a real next step. This is
  now the README's "why the hybrid doesn't win here" section, written as a genuine
  finding rather than smoothed over.
- **Poster lookup upgraded, not just adapted**: since `movie_id` is now a real IMDb
  `tt` id (unlike MovieLens where posters had to be found by fuzzy title search),
  added `get_poster_url_by_imdb_id`/`get_poster_urls_by_imdb_id` to `src/posters.py`
  using TMDb's `/find/{imdb_id}?external_source=imdb_id` endpoint -- strictly more
  reliable than title search, no normalization heuristics needed. The original
  title-search path (`get_poster_url`/`get_poster_urls`) is kept for general reuse, not
  deleted. 8 new tests added alongside the existing poster tests.
- **Tests fully rewritten for the new schema** (`tests/test_data_pipeline.py`,
  `tests/test_models.py`) rather than patched -- old fixtures used int ids and 1-5
  star ratings; new ones use string ids and hand-recomputed ternary-scale expected
  values (e.g. the content-based cosine-similarity test's expected profile vector and
  similarity scores were recalculated by hand for the new rating values, not just
  reused with the type changed). `tests/test_evaluate.py` needed **zero** changes --
  it only tests the metric math functions in isolation against fabricated numbers, not
  tied to the real rating scale.
- `requirements.txt`: added `kaggle==2.2.4` (the version actually installed and
  verified working).
- `app.py` updated to match: dataset-name copy, cold-start sentinel changed from the
  MovieLens-specific int `-1` to a string (`"__simulated_new_user__"`, since real user
  ids are now free-text handles), and poster lookup switched to
  `get_poster_urls_by_imdb_id` (keyed by `movie_id`, which *is* the IMDb id now,
  instead of by constructed title strings). Verified locally end-to-end with the
  browser tool: real Bollywood titles and posters render correctly (3 Idiots, Munna
  Bhai M.B.B.S., Lage Raho Munna Bhai, ...), the 4-column compare view, and the
  cold-start simulated-user path all work with zero exceptions.
- **Real bug hit on the actual redeploy, not caught locally**: after pushing and
  redeploying, the live app crashed with `Failed to train models: "['Biography',
  'Family', 'History', 'Music', 'News', 'Sport'] not in index"`. Root cause: `data/`
  is gitignored, so a code-only redeploy doesn't wipe it, and Streamlit Cloud reuses
  the same container across deploys -- the *previous* MovieLens-era
  `data/processed/movies.csv` (19 genre columns) was still sitting on disk and
  satisfied `run_pipeline`'s old cache check ("do the files exist"), so it got loaded
  as-is instead of being rebuilt from the new Kaggle source, and every model choked on
  the six genre columns that only exist in the new dataset's taxonomy. This couldn't
  have been caught locally since the local machine never had a MovieLens-era cache
  sitting next to the new code. Fixed with `_processed_cache_is_usable` in
  `src/data_pipeline.py`: before trusting a cache hit, `run_pipeline` now also checks
  the cached `movies.csv` header actually contains every column in `GENRE_COLUMNS`,
  and rebuilds from scratch (with a logged warning) if not. Covered by a real
  integration test (`TestRunPipelineCacheInvalidation`) that reproduces the exact
  failure: a stale processed cache missing most genre columns sitting next to a valid
  raw dataset, asserting the pipeline rebuilds rather than trusting the stale files.
  General lesson worth remembering: an on-disk cache keyed only on "do the expected
  files exist" is unsafe across any change to what those files are expected to
  contain, not just across dataset swaps -- worth a second look if this pipeline's
  processed-CSV schema changes again for any reason.
- **Second real issue hit on redeploy, after the cache fix**: pushing the cache fix
  didn't actually fix the live app -- it kept showing the same stale error. Root
  cause: Streamlit Cloud's git-triggered redeploy for a Python-only change (no new
  dependency) doesn't restart the underlying process, so `load_data()`'s
  `st.cache_data` result from the *first* bad run stayed pinned in memory and
  `run_pipeline` (my fix included) never ran again. Forced a real fix via the
  Cloud dashboard's app menu -> **Reboot** (not just another git push) -- confirmed
  via the logs this does a genuine fresh container (full re-clone, full dependency
  reinstall from scratch, "Uvicorn server started" from zero), unlike a normal
  redeploy. After that reboot, the app hung completely -- zero log output for over a
  minute, not even the pipeline's own first log line, with the CSS/hero rendering
  fine (proving the script started) but never reaching a rendered spinner. Suspected
  (not fully confirmed) cause: the `kaggle` package's own network calls
  (`kaggle.api.authenticate()`, `dataset_download_files()`) don't reliably apply a
  request timeout across versions, so a slow/unreachable hop to Kaggle's API from
  that specific container could hang indefinitely with no error and no log output --
  unlike every TMDb call in `src/posters.py`, which has always had an explicit
  `REQUEST_TIMEOUT_SECONDS`. Fixed defensively regardless of full confirmation:
  `download_indian_movies_dataset` now wraps just that network call in
  `socket.setdefaulttimeout(KAGGLE_SOCKET_TIMEOUT_SECONDS)` (30s), restored in a
  `finally` right after, so a stalled Kaggle call now fails fast with an actionable
  `DataDownloadError` instead of silently wedging the whole app. Verified locally
  with a genuinely fresh raw-data cache (moved `data/raw/indian_movies` and
  `data/processed` aside first) that this doesn't slow down or break the real,
  successful download path: ~4.8s end-to-end, unchanged from before the fix.

## Known environment quirks
- pandas 3.0.5's compiled Cython DLLs were blocked by this machine's Windows
  Application Control policy on install (numpy/scipy were unaffected). Pinned to
  pandas==2.2.3 instead, which installs and imports cleanly.

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
- `src/data_pipeline.py` — download (via Kaggle API) / cache the Indian Regional Movie Dataset, clean, random per-user split
- `src/models.py` — `ContentBasedRecommender`, `SVDRecommender`, `HybridRecommender`, cold-start fallback wrapper; rating scale is `[-1, 1]` (see Phase 7), ids are strings
- `src/evaluate.py` — RMSE/MAE plus Precision@K, Recall@K, NDCG@K
- `src/posters.py` — optional TMDb poster-image lookup, by IMDb id (preferred, exact) or by title (fallback); no Streamlit dependency
- `app.py` — Streamlit demo; must run standalone from a fresh clone with no manual setup beyond `pip install` **and a Kaggle API token** (see Phase 7 -- unlike MovieLens, this dataset has no unauthenticated public URL, so this is the one hard prerequisite that can't be made optional). Also owns the cinema theme's bespoke CSS (`.streamlit/config.toml` sets the base widget theme)

## Do not
- Don't commit `data/` (raw dataset files) — it's gitignored and re-downloaded on first run
- Don't commit `~/.kaggle/kaggle.json` or any Kaggle/TMDb credential — Kaggle creds live outside the repo entirely (home directory), TMDb's local key lives in gitignored `.streamlit/secrets.toml`
- Don't hardcode file paths — use `pathlib`, relative to project root
- Don't add a dependency without pinning its version in `requirements.txt`

## Definition of done
Code runs without errors, has tests, has type hints and docstrings, and logs meaningfully instead of printing.