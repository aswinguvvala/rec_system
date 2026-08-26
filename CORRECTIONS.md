# Corrections Log

A running record of real mistakes found in this project, how they were diagnosed, and
how they were fixed — kept separate from `CLAUDE.md`'s phase-by-phase build log so it's
easy to scan just "what went wrong and how we caught it." Every entry below was a real
bug that shipped or nearly shipped, found either by the user asking a pointed question
("are the recommendations correct?") or by writing tests, not by inspection alone. Each
one is verified against real data or the real running app, not just re-read and assumed
fixed.

---

## 1. `st.dataframe` deprecation warning

**What happened:** `st.dataframe(..., use_container_width=True)` printed a deprecation
warning in the installed `streamlit==1.60.0`, past the flag's stated removal date.

**Found by:** Headless `AppTest` verification during Phase 4.

**Fix:** Changed to `st.dataframe(..., width="stretch")`.

**Verified:** Re-ran `AppTest` — no warning, table still renders identically.

---

## 2. A cold-start test that passed for the wrong reason

**What happened:** `TestColdStartRecommender`'s "above-threshold user uses the base
model, not the fallback" test used the shared 5-movie fixture, in which one fixture user
had already rated all 5 movies. With no unrated movie left as a candidate, the
recommender legitimately returned an empty list either way — the test passed whether or
not the cold-start branching logic was actually correct.

**Found by:** Re-reading the test while writing more cold-start tests in Phase 5 and
asking whether it could ever fail.

**Fix:** Gave `TestColdStartRecommender` its own 6-movie fixture with one movie nobody
rates, guaranteeing a real unseen candidate exists for the assertion to actually check.

**Verified:** Deliberately broke the cold-start threshold logic locally and confirmed
the new fixture's test failed (the old one wouldn't have).

---

## 3. A "Bollywood ratings" dataset that was secretly MovieLens

**What happened:** While researching Indian-movie datasets with real per-user ratings
(Phase 7), one promising-looking Kaggle/GitHub hit (`TIMDB`) claimed to have a
"collaborative filtering ratings.csv" for Bollywood movies.

**Found by:** Actually opening and inspecting the file's contents before committing to
it, instead of trusting the repo description — the ratings were the *original MovieLens*
ids and structure, repackaged under new movie titles.

**Fix:** Rejected that source; used the **Indian Regional Movie Dataset** (Agarwal et
al., arXiv:1801.02203) instead, whose raw `ratings.json`/`movies.csv`/`users.csv` were
inspected directly before any pipeline code was written.

**Verified:** Manual inspection of raw file contents against the paper's own schema
description — real IMDb `tt` ids, ternary ratings, no MovieLens fingerprints.

---

## 4. Live deploy crash: stale on-disk cache from the old dataset

**What happened:** After swapping datasets and redeploying, the live app crashed with
`KeyError: "['Biography', 'Family', 'History', 'Music', 'News', 'Sport'] not in index"`.

**Root cause:** `data/` is gitignored, so a code-only redeploy doesn't wipe it, and
Streamlit Cloud reuses the same container across deploys. The *previous* MovieLens-era
`data/processed/movies.csv` (19 genre columns) was still on disk and satisfied the
pipeline's old cache check ("do the files exist?"), so it got loaded as-is instead of
rebuilt from the new source — every model then choked on genre columns that only exist
in the new dataset's taxonomy. This couldn't be caught locally, since the local machine
never had an old cache sitting next to the new code.

**Fix:** Added `_processed_cache_is_usable()` in `src/data_pipeline.py` — before
trusting a cache hit, checks the cached `movies.csv` header actually contains every
expected genre column, and rebuilds from scratch (with a logged warning) if not.

**Verified:** Real integration test (`TestRunPipelineCacheInvalidation`) reproduces the
exact failure — a stale cache missing genre columns next to a valid raw dataset — and
asserts the pipeline rebuilds instead of trusting the stale files.

**General lesson:** An on-disk cache keyed only on "do the expected files exist" is
unsafe across any change to what those files are expected to *contain*.

---

## 5. Live deploy hang: no timeout on the Kaggle download

**What happened:** After fix #4 was pushed, the live app didn't crash anymore — it hung
completely, zero log output for over a minute.

**Root cause (suspected, fixed defensively regardless):** The `kaggle` package's network
calls don't reliably apply a request timeout across versions, so a slow/unreachable hop
to Kaggle's API from that container could hang indefinitely with no error — unlike every
TMDb call in `src/posters.py`, which already had an explicit timeout.

**Fix:** Wrapped the Kaggle download call in `socket.setdefaulttimeout(30)`, restored in
a `finally` block, so a stalled call now fails fast with an actionable error instead of
wedging the whole app.

**Verified:** Locally, with a genuinely fresh raw-data cache — confirmed the real,
successful download path is unaffected (~4.8s end-to-end, unchanged).

---

## 6. Live deploy auth error: Streamlit secrets aren't OS environment variables

**What happened:** After fix #5, the hang became a clear error:
`Authentication required to call the Kaggle API` — even though `KAGGLE_USERNAME`/
`KAGGLE_KEY` were set in Streamlit Cloud's secrets.

**Root cause:** An unverified assumption — Streamlit's `st.secrets` values are **not**
automatically exposed as real OS environment variables. The `kaggle` package
authenticates by reading `os.environ` / `~/.kaggle/kaggle.json` itself, not by taking
credentials as a function argument, so it never saw the Cloud secrets at all. This never
surfaced locally because a real `kaggle.json` file already satisfied the package's auth
independent of anything Streamlit does.

**Fix:** Added `configure_kaggle_credentials_from_secrets()` in `app.py` — explicitly
copies `st.secrets["KAGGLE_USERNAME"]`/`["KAGGLE_KEY"]` into `os.environ` (only if not
already set) before the pipeline ever runs.

**Verified:** Locally reproduced the Cloud scenario faithfully — moved the real
`kaggle.json` aside, added the same two secrets to local `.streamlit/secrets.toml`,
cleared all caches, and confirmed the app downloads/trains/renders using *only* the
secrets-bridge path.

**General lesson:** Don't assume a platform's "secrets" mechanism transparently becomes
environment variables for every library that reads `os.environ` directly — verify it, or
bridge it explicitly.

---

## 7. Cold-start picks ignored language entirely

**What happened:** User picked three Telugu movies ("1 - Nenokkadine",
"#Pellichoopulu", "Nuvvu Naaku Nachchav") in the live onboarding flow and got back an
all-Hindi recommendation list (Bawarchi, DDLJ, OMG: Oh My God!, 3 Idiots, PK, ...).

**Found by:** The user directly questioning the output rather than assuming it was
fine: *"i picked three telugu language movies and all the recommendations are hindi
movies... how are we recommending?"*

**Root cause:** The genre-affinity profile *was* being applied correctly (verified by
comparing against pure-popularity output side by side — a genuinely different ranking,
not a silent fallback) — but this dataset's 21 genres carry **no language information at
all**, and broad genres like Comedy/Drama are shared by huge Hindi blockbusters with far
more ratings than any Telugu title. They won on popularity even with a correctly-applied
genre match.

**Fix:** Added `movie_ids_matching_languages()` — restricts the candidate pool to movies
sharing a language with the picks *before* popularity/genre scoring runs, as a second,
orthogonal signal rather than a bigger genre weight. `PopularityRecommender
.recommend_for_genre_profile` gained an optional `candidate_movie_ids` parameter for
this, defaulting to `None` (fully backward compatible).

**Verified:** Re-ran the user's exact three picks against the real fitted model — top 10
went from all-Hindi to Telugu/Tamil-Telugu titles, confirmed live in the browser.

---

## 8. Live deploy `TypeError`: stale `st.cache_resource` model instance

**What happened:** Pushed the language fix (#7); the very first live load crashed with
`TypeError: PopularityRecommender.recommend_for_genre_profile() got an unexpected
keyword argument 'candidate_movie_ids'`.

**Root cause:** `app.py` was running the new code (it called the new keyword argument),
but the `st.cache_resource`-cached `PopularityRecommender` instance was still the *old*
class. `st.cache_resource` hashes the **decorated function's own source**, not anything
it transitively calls — `load_models()`'s own source hadn't changed, so its cache stayed
valid even though the class it constructs had. Same root-cause pattern as #4, in a
different cache.

**Fix:** No code change — an explicit **Reboot** from the Streamlit Cloud dashboard,
which forces a genuinely fresh process and unconditionally clears every cache regardless
of the hash it was stored under. A normal git push does not do this for a
dependency-unchanged deploy.

**Verified:** Confirmed via the Cloud build logs that Reboot performs a true fresh
container start ("Uvicorn server started" from zero), then re-tested the exact failing
interaction live.

**General lesson (confirmed twice now, in two different caches):** Any time a
class/function's *signature* changes but the cached function that constructs/calls it
does not, assume the old object can survive a plain push — verify with a real "zero
state" interaction after every such deploy, and reach for Reboot rather than re-pushing.

---

## 9. "Dil Se.." bug: language matching was too loose

**What happened:** User asked directly, *"But think about this, are the recommendations
correct?"* — prompting a second look at #7's fix rather than assuming it was complete.

**Root cause:** `movie_ids_matching_languages` matched on **any** shared language tag in
the comma-joined `languages` column. A famous Hindi film (dubbed into Telugu among
several other languages) has Telugu somewhere in that list, so it slipped into
"Telugu-picked" results even though Hindi is its real industry/language.

**Fix:** Spot-checked 7 well-known films and confirmed the *first*-listed language
reliably indicates the true original industry in this dataset. Added `_primary_language()`
and rewrote `movie_ids_matching_languages` to match on primary language only, not any
shared tag.

**Verified:** Re-ran the fitted model — Dil Se.. now correctly excluded, all 10 results
for the Telugu picks genuinely Telugu-primary.

---

## 10. The bigger gap: "similar" recommendations weren't actually similar to anything

**What happened:** Even with language fixed, the user pushed further: *"why are we
recommending these movies to the user when they liked 2 movies because they should be
similar... or other users may have liked similar movies."*

**Root cause:** The whole cold-start ranking was really just "popularity, restricted to
one aggregated genre-affinity vector blended across all picks" — it never checked
whether any *specific* recommended movie resembled any *specific* picked movie, and used
no collaborative-filtering signal at all, despite SVD already being fit and sitting
unused for this path. `ContentBasedRecommender.similar_items()` (genre-based item-item
similarity) already existed in the codebase but was never called from cold-start.

**Fix:**
- Added `SVDRecommender.similar_items()` — cosine similarity over the learned item-factor
  matrix, i.e. real "users who rated similar movies similarly" signal. Returns `[]` for
  a movie with zero training ratings (~51% of the catalog) rather than fabricating one.
- Added `recommend_similar_to_picks()` — for each individual pick (not one aggregated
  blob), pulls real neighbors from both content and collaborative `similar_items`, takes
  the best score per candidate across picks, min-max normalizes each signal and blends
  them (same normalize-then-weight pattern `HybridRecommender` already used for users,
  now applied per item). Falls back to the old language-restricted popularity ranking
  only as a backfill when too few real neighbors are found.

**Verified:** Ran the user's exact three picks through a standalone script *and* live in
the browser (single-view and compare-mode). Confirmed the results are genuine
neighbors — e.g. picking "1 - Nenokkadine" alone surfaces C.I.D. (an exact genre match)
and Prasthanam/1971/Aligarh (real SVD item-factor nearest neighbors, checked by printing
the raw `similar_items` output, not assumed). Also checked a suspicious-looking tie
plateau in the scores (several results all scoring exactly 0.500) and confirmed by
genre inspection that it's a real tie (multiple movies sharing an identical genre vector
with one pick, cosine similarity 1.0), not a normalization bug.

---

## 11. A latent schema-assumption bug, caught while writing tests for #10

**What happened:** While writing tests for `recommend_similar_to_picks()` against the
small test fixture, `genre_profile_from_movie_ids()` raised `KeyError` for genre columns
that don't exist in the fixture.

**Root cause:** The function's default genre-column list was the full production
`GENRE_COLUMNS` constant, applied unconditionally — unlike `PopularityRecommender.fit()`,
which already intersected `GENRE_COLUMNS` with whatever columns the real `movies_df`
actually has. Harmless in production today (the real data always has every column), but
one schema change or dataset swap away from a real crash.

**Fix:** Changed the default to intersect with `movies_df.columns`, matching the pattern
`PopularityRecommender.fit()` already used.

**Verified:** Full test suite (105 tests) passes; re-confirmed against the real
production dataset that behavior is unchanged there.

---

## Standing rule that shaped several of the above

Every credential handled in this project (TMDb key, Kaggle username/key) was written to
a local file (`.streamlit/secrets.toml`, `~/.kaggle/kaggle.json`) directly by Claude when
the user provided it in chat — but never entered into a web form (e.g. Streamlit Cloud's
secrets UI), even when the user explicitly handed over the value and asked for it. That
distinction (local file write = fine; web form entry = never) was held consistently
throughout, meaning the user pasted secrets into Streamlit Cloud's UI themselves each
time a cloud secret was needed.
