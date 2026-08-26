"""Streamlit demo for the Hybrid Movie Recommendation System.

Run with ``streamlit run app.py``. On a completely fresh clone (empty
``data/``), the first load triggers the full pipeline automatically --
download the Indian Regional Movie Dataset via the Kaggle API, clean it,
split it, and train all five models -- so there is no manual data-prep
step beyond configuring a Kaggle API token (see README). Subsequent loads
reuse Streamlit's cache and the pipeline's own on-disk cache, so they're
fast.

Poster images are optional, best-effort enrichment from TMDb (see
``src/posters.py``) -- the app runs fine with no ``TMDB_API_KEY``
configured, falling back to a placeholder poster per card. The dark
"cinema" theme lives partly in ``.streamlit/config.toml`` (native widgets)
and partly in the CSS injected below (hero, cards, background motif).

Note: this is the one file in the project where ``print`` isn't banned,
per ``claude.md`` -- but this app deliberately doesn't use it. Streamlit's
own ``st.error``/``st.info`` calls are the idiomatic way to surface
status and error messages in a Streamlit UI.
"""

import html
import json
import os

import pandas as pd
import streamlit as st

from src.data_pipeline import DataPipelineError, run_pipeline
from src.models import (
    ColdStartRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    PopularityRecommender,
    Recommendation,
    SVDRecommender,
    genre_profile_from_movie_ids,
    movie_ids_matching_languages,
)
from src.posters import get_poster_urls_by_imdb_id
from src.utils import RESULTS_DIR

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="\U0001f3ac", layout="wide")

# Sentinel guaranteed absent from this dataset's real (free-text, human-typed) user ids.
COLD_START_USER_ID = "__simulated_new_user__"

# label, emoji, accent color (used for both the CSS badge class and the compact dot).
SOURCE_META: dict[str, tuple[str, str, str]] = {
    "content": ("Content-Based", "\U0001f3ad", "#7ec3dd"),
    "svd": ("SVD (Collaborative)", "\U0001f91d", "#b79bf0"),
    "hybrid": ("Hybrid", "\U0001f500", "#e3b23c"),
    "cold_start": ("Cold-Start Popularity", "\U0001f195", "#e2837c"),
}

CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0908;
    --bg-card: rgba(255,255,255,0.035);
    --bg-card-hover: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.09);
    --text: #f2ede4;
    --text-muted: #9c9186;
    --accent-gold: #e3b23c;
    --font-display: 'Bebas Neue', 'Arial Narrow', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

html, body, [class*="css"] { font-family: var(--font-body); }

/* Procedural cinema background: warm spotlight glow behind the hero, a dark
   vignette toward the edges, and a filmstrip perforation strip along the
   very top and bottom of the viewport -- no external image, same technique
   as a hand-drawn CSS starfield, just a film motif instead of a space one. */
.stApp {
    background:
        radial-gradient(ellipse 900px 480px at 50% -8%, rgba(227,178,60,0.10), transparent 60%),
        radial-gradient(ellipse 1100px 700px at 50% 105%, rgba(0,0,0,0.55), transparent 70%),
        var(--bg);
}
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    left: 0; right: 0;
    height: 22px;
    background-color: #050403;
    background-image: repeating-radial-gradient(circle at 16px 11px, rgba(242,237,228,0.16) 0 3px, transparent 3px 32px);
    z-index: 0;
    pointer-events: none;
}
.stApp::before { top: 0; }
.stApp::after { bottom: 0; }

#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] { background: #100e0b; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h2 {
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: .1em;
    font-size: .8rem;
    color: var(--accent-gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: .6rem;
}

.stApp h3 {
    font-family: var(--font-display);
    font-weight: 400;
    letter-spacing: .02em;
    font-size: 1.7rem;
    color: var(--text);
}

/* Hero */
.hero { padding: .5rem 0 1.25rem; position: relative; z-index: 1; }
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: .72rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: .5rem;
}
.hero-title {
    font-family: var(--font-display);
    font-weight: 400;
    font-size: 3.4rem;
    line-height: 1.05;
    letter-spacing: .01em;
    color: var(--text);
    margin: 0 0 .6rem;
}
.hero-tagline {
    font-family: var(--font-body);
    font-size: .95rem;
    color: var(--text-muted);
    max-width: 62ch;
    line-height: 1.5;
}
.key-hint {
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--text-muted);
    margin-top: .5rem;
}

/* Movie cards */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 1.1rem;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
}
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.poster-wrap { position: relative; width: 100%; aspect-ratio: 2 / 3; background: #1c1812; overflow: hidden; }
.poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-placeholder {
    width: 100%; height: 100%;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .4rem;
    background: linear-gradient(160deg, #221c14, #14110d);
    color: rgba(242,237,228,0.32);
    font-size: 2rem;
    text-align: center;
    padding: .75rem;
}
.poster-placeholder span {
    font-family: var(--font-mono);
    font-size: .62rem;
    letter-spacing: .02em;
    color: rgba(242,237,228,0.45);
    line-height: 1.3;
}
.rank-badge {
    position: absolute; top: 8px; left: 8px;
    width: 24px; height: 24px; border-radius: 50%;
    background: rgba(10,9,8,0.75);
    border: 1px solid var(--accent-gold);
    color: var(--accent-gold);
    font-family: var(--font-mono);
    font-size: .72rem;
    display: flex; align-items: center; justify-content: center;
}
.movie-card-body { padding: .65rem .75rem .8rem; display: flex; flex-direction: column; gap: .45rem; flex: 1; }
.movie-title { font-size: .85rem; font-weight: 600; line-height: 1.25; color: var(--text); }
.movie-meta { display: flex; align-items: center; justify-content: space-between; gap: .5rem; flex-wrap: wrap; }
.badge {
    font-family: var(--font-mono);
    font-size: .6rem;
    letter-spacing: .02em;
    padding: .2rem .5rem;
    border-radius: 999px;
    white-space: nowrap;
    border: 1px solid;
}
.score { font-family: var(--font-mono); font-size: .66rem; color: var(--text-muted); }

/* Compact horizontal card, used in the 4-column side-by-side view */
.movie-list--compact { display: flex; flex-direction: column; gap: .55rem; margin-top: .75rem; position: relative; z-index: 1; }
.movie-card--compact { flex-direction: row; align-items: stretch; }
.movie-card--compact .poster-wrap { width: 50px; flex: 0 0 50px; aspect-ratio: auto; }
.movie-card--compact .rank-badge { width: 18px; height: 18px; font-size: .6rem; top: 4px; left: 4px; }
.movie-card--compact .movie-card-body { padding: .4rem .55rem; justify-content: center; gap: .25rem; }
.movie-card--compact .movie-title { font-size: .72rem; }
.movie-card--compact .badge { font-size: .55rem; padding: .12rem .4rem; }
.movie-card--compact .score { font-size: .6rem; }

.empty-hint { font-family: var(--font-mono); font-size: .78rem; color: var(--text-muted); padding: .75rem 0; }
</style>
"""


@st.cache_data(show_spinner="Loading the Indian Regional Movie Dataset (first run downloads it via the Kaggle API)...")
def load_data() -> dict[str, pd.DataFrame]:
    return run_pipeline()


@st.cache_resource(show_spinner="Training recommender models (SVD takes ~15s on first launch)...")
def load_models() -> dict[str, object]:
    data = load_data()
    train_df, movies_df = data["train"], data["movies"]

    registry: dict[str, object] = {
        "content": ContentBasedRecommender(),
        "svd": SVDRecommender(),
        "popularity": PopularityRecommender(),
        "hybrid": HybridRecommender(ContentBasedRecommender(), SVDRecommender(), strategy="weighted", alpha=0.6),
        "cold_start": ColdStartRecommender(
            base_model=HybridRecommender(ContentBasedRecommender(), SVDRecommender(), strategy="weighted", alpha=0.6),
            popularity_model=PopularityRecommender(),
            min_user_ratings=5,
        ),
    }
    for model in registry.values():
        model.fit(train_df, movies_df)
    return registry


@st.cache_data
def load_metrics_table() -> pd.DataFrame | None:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text())).T.round(4)


def get_tmdb_api_key() -> str | None:
    """Read an optional TMDb API key from the environment or Streamlit secrets.

    Checked in that order so local development (an env var) and Streamlit
    Community Cloud (a secret) both work without code changes. Absence of a
    key is never an error -- callers fall back to placeholder posters.

    Returns:
        The API key if configured, else ``None``.
    """
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("TMDB_API_KEY")
    except Exception:  # noqa: BLE001 - st.secrets' behavior with no secrets.toml varies across
        # Streamlit versions/hosts; a missing key must never be fatal here regardless of how it fails.
        return None


def configure_kaggle_credentials_from_secrets() -> None:
    """Bridge Kaggle credentials from Streamlit secrets into real environment variables.

    Unlike TMDb (where this app passes the key explicitly to every call), the
    `kaggle` package authenticates itself, internally, purely via the
    KAGGLE_USERNAME/KAGGLE_KEY environment variables or a ~/.kaggle/kaggle.json
    file -- it has no notion of Streamlit secrets. Locally, a kaggle.json file
    (see README) already satisfies this with no code involved. On Streamlit
    Community Cloud, secrets.toml values are *not* automatically exposed as
    real process environment variables -- confirmed the hard way: the deployed
    app's Kaggle download failed with the kaggle package's own "Authentication
    required" message even with KAGGLE_USERNAME/KAGGLE_KEY set as Cloud
    secrets, because nothing had ever copied them into os.environ where the
    kaggle package actually looks. This bridges them explicitly, once, before
    the pipeline runs. A no-op wherever the env vars are already set (e.g.
    locally) or no matching secrets exist.
    """
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - same reasoning as get_tmdb_api_key: st.secrets'
            # behavior with no secrets.toml varies across Streamlit versions/hosts, and a
            # missing secret here must never be fatal -- download_indian_movies_dataset
            # already raises its own actionable DataDownloadError if credentials are absent.
            value = None
        if value:
            os.environ[key] = value


@st.cache_data(show_spinner=False, ttl=3600)
def load_posters(movie_ids: tuple[str, ...], api_key: str | None) -> dict[str, str | None]:
    # movie_id is the movie's real IMDb tt id in this dataset, so an exact
    # find-by-id lookup is used instead of the fuzzy title search src/posters.py
    # also offers -- see claude.md's Phase 7 notes.
    return get_poster_urls_by_imdb_id(movie_ids, api_key)


def _movie_card_html(rank: int, title: str, rec: Recommendation, poster_url: str | None, *, compact: bool = False) -> str:
    """Render one recommendation as a movie-card HTML snippet."""
    label, emoji, color = SOURCE_META.get(rec.source, (rec.source, "•", "#9c9186"))
    safe_title = html.escape(title)

    if poster_url:
        poster_html = f'<img src="{html.escape(poster_url)}" alt="{safe_title}" loading="lazy">'
    else:
        short_title = title if len(title) <= 40 else title[:37] + "..."
        poster_html = f'<div class="poster-placeholder">\U0001f3ac<span>{html.escape(short_title)}</span></div>'

    card_class = "movie-card movie-card--compact" if compact else "movie-card"
    return f"""
<div class="{card_class}">
  <div class="poster-wrap">
    {poster_html}
    <span class="rank-badge">{rank}</span>
  </div>
  <div class="movie-card-body">
    <div class="movie-title">{safe_title}</div>
    <div class="movie-meta">
      <span class="badge" style="background:{color}26; color:{color}; border-color:{color}59;">{emoji} {label}</span>
      <span class="score">score {rec.score:.3f}</span>
    </div>
  </div>
</div>
"""


def render_recommendations(
    recs: list[Recommendation],
    movie_id_to_title: dict[str, str],
    poster_map: dict[str, str | None],
    container,
    *,
    compact: bool = False,
    empty_hint: str = "",
) -> None:
    """Render a ranked list of recommendations as a grid (or compact list) of movie cards."""
    if not recs:
        container.markdown(f'<div class="empty-hint">No recommendations returned. {empty_hint}</div>'.strip(), unsafe_allow_html=True)
        return
    cards = []
    for rank, rec in enumerate(recs, start=1):
        title = movie_id_to_title.get(rec.movie_id, f"Movie {rec.movie_id}")
        poster_url = poster_map.get(rec.movie_id)
        cards.append(_movie_card_html(rank, title, rec, poster_url, compact=compact))
    wrapper_class = "movie-list--compact" if compact else "movie-grid"
    container.markdown(f'<div class="{wrapper_class}">{"".join(cards)}</div>', unsafe_allow_html=True)


def recommend_for_new_user(
    popularity_model: PopularityRecommender,
    movies_df: pd.DataFrame,
    picked_movie_ids: list[str],
    n: int,
) -> list[Recommendation]:
    """Recommend movies for a brand-new user based on movies they just said they like.

    This is the live counterpart to ``ColdStartRecommender``'s training-data-driven
    fallback: instead of a real rating history, it turns the user's on-the-spot
    picks into a genre-preference vector (:func:`genre_profile_from_movie_ids`) and
    blends that with overall popularity via the same
    ``PopularityRecommender.recommend_for_genre_profile`` the wrapper uses
    internally. With no picks yet, this is identical to pure trending popularity.

    Genre affinity alone is language-blind (see
    :func:`movie_ids_matching_languages`), so picks are also used to narrow the
    ranked pool to movies sharing at least one language with them -- otherwise a
    handful of blockbuster hits in the catalog's dominant language can dominate
    every profile regardless of what the user actually picked, since a broad
    genre match plus much higher raw popularity outranks a smaller-language film
    that's a better match. Falls back to the unrestricted catalog if the picks
    have no parsed language info at all.

    Args:
        popularity_model: A fitted ``PopularityRecommender``.
        movies_df: Movie metadata, used to build the genre/language profile
            from picks.
        picked_movie_ids: IDs of movies the user picked as "movies I like".
        n: Number of recommendations to return.

    Returns:
        Up to ``n`` :class:`Recommendation` objects, none of them one of the
        user's own picks.
    """
    genre_profile = genre_profile_from_movie_ids(picked_movie_ids, movies_df)
    language_candidates = movie_ids_matching_languages(picked_movie_ids, movies_df)
    # Fetch slack beyond n: a movie the user just picked as "liked" is exactly the
    # kind of item this ranking tends to surface, so it's likely to appear in the
    # raw results and needs to be filtered back out below.
    raw_recs = popularity_model.recommend_for_genre_profile(
        genre_weights=genre_profile,
        n=n + len(picked_movie_ids),
        exclude_seen=False,
        candidate_movie_ids=language_candidates,
    )
    picked = set(picked_movie_ids)
    return [r for r in raw_recs if r.movie_id not in picked][:n]


st.markdown(CINEMA_CSS, unsafe_allow_html=True)

configure_kaggle_credentials_from_secrets()
try:
    data = load_data()
except DataPipelineError as exc:
    st.error(f"Failed to prepare the movie dataset: {exc}")
    st.stop()

try:
    models = load_models()
except Exception as exc:  # noqa: BLE001 - top-level UI boundary, must not crash on any model error
    st.error(f"Failed to train models: {exc}")
    st.stop()

train_df, movies_df, users_df = data["train"], data["movies"], data["users"]
movie_id_to_title: dict[str, str] = dict(zip(movies_df["movie_id"], movies_df["title"]))
tmdb_api_key = get_tmdb_api_key()

st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">Indian Regional Movie Dataset &middot; Content-Based + SVD + Hybrid</div>
  <h1 class="hero-title">\U0001f3ac Hybrid Movie<br>Recommendation System</h1>
  <p class="hero-tagline">Content-based genre similarity and a from-scratch SVD collaborative filter,
  combined by a weighted hybrid, with an explicit cold-start fallback for new users and unrated movies.</p>
</div>
""",
    unsafe_allow_html=True,
)
if not tmdb_api_key:
    st.markdown(
        '<div class="key-hint">\U0001f511 No TMDB_API_KEY configured &mdash; showing placeholder posters. '
        "Add one as an env var or a Streamlit secret for real artwork.</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Controls")
    simulate_cold = st.checkbox(
        "Simulate a brand-new user (cold start)",
        value=False,
        help=(
            "Every real user in this dataset already has training ratings by construction, "
            "so this is the only way to see the cold-start path trigger live."
        ),
    )
    picked_movie_ids: list[str] = []
    if simulate_cold:
        selected_user_id = COLD_START_USER_ID
        sorted_movie_ids = sorted(movie_id_to_title, key=lambda mid: movie_id_to_title[mid])
        picked_movie_ids = st.multiselect(
            "Movies you like (pick a few)",
            options=sorted_movie_ids,
            format_func=lambda mid: movie_id_to_title.get(mid, mid),
            help="No rating history needed -- recommendations below update live from these picks.",
        )
        if picked_movie_ids:
            st.caption(f"Personalizing from {len(picked_movie_ids)} pick(s) -- matched by genre and language.")
        else:
            st.caption("No picks yet -- showing overall trending movies until you pick a few.")
    else:
        rating_counts = train_df.groupby("user_id").size()
        user_ids = sorted(rating_counts.index.tolist())
        selected_user_id = st.selectbox(
            "Pick a user",
            user_ids,
            format_func=lambda uid: f"{uid} ({rating_counts.get(uid, 0)} ratings)",
        )
        user_row = users_df.loc[users_df["user_id"] == selected_user_id]
        if not user_row.empty:
            row = user_row.iloc[0]
            st.caption(f"Age {row['age']} · {row['gender']} · {row['occupation']}")

    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    compare_mode = st.checkbox("Compare recommenders side-by-side", value=False)

user_label = "a simulated new user" if simulate_cold else f"User {selected_user_id}"

if compare_mode:
    st.subheader(f"Side-by-side comparison for {user_label}")
    if simulate_cold:
        st.caption(
            "The raw Content-Based, SVD, and Hybrid models have no ratings to work with for a brand-new "
            "user and return nothing -- that gap is exactly what the cold-start wrapper (right) exists to fill."
        )
    panel_order = [
        ("content", "Content-Based \U0001f3ad"),
        ("svd", "SVD (Collaborative) \U0001f91d"),
        ("hybrid", "Hybrid \U0001f500"),
        ("popularity", "Popularity Baseline \U0001f4ca"),
    ]
    panel_recs = {}
    for key, _ in panel_order:
        if simulate_cold and key == "popularity":
            panel_recs[key] = recommend_for_new_user(models["popularity"], movies_df, picked_movie_ids, n_recs)
        else:
            panel_recs[key] = models[key].recommend_for_user(selected_user_id, n=n_recs)
    needed_movie_ids = {rec.movie_id for recs in panel_recs.values() for rec in recs}
    poster_map = load_posters(tuple(sorted(needed_movie_ids)), tmdb_api_key)

    columns = st.columns(4)
    for col, (model_key, title) in zip(columns, panel_order):
        col.markdown(f"### {title}")
        render_recommendations(
            panel_recs[model_key],
            movie_id_to_title,
            poster_map,
            col,
            compact=True,
            empty_hint="(no cold-start handling in this raw model)",
        )
else:
    st.subheader(f"Top {n_recs} recommendations for {user_label}")
    if simulate_cold:
        recs = recommend_for_new_user(models["popularity"], movies_df, picked_movie_ids, n_recs)
    else:
        recs = models["cold_start"].recommend_for_user(selected_user_id, n=n_recs)
    needed_movie_ids = {rec.movie_id for rec in recs}
    poster_map = load_posters(tuple(sorted(needed_movie_ids)), tmdb_api_key)
    render_recommendations(recs, movie_id_to_title, poster_map, st)

st.divider()
st.subheader("Model performance (held-out test set)")
metrics_df = load_metrics_table()
if metrics_df is not None:
    st.dataframe(metrics_df, width="stretch")
    st.caption(
        "RMSE/MAE: preference-score prediction error on this dataset's [-1, 1] scale, lower is better. "
        "Precision/Recall/NDCG@K: ranking quality against the full movie catalog (no negative sampling), "
        "higher is better, computed on a per-user random train/test split -- see claude.md for why "
        "random rather than chronological (this dataset has no timestamps)."
    )
else:
    st.info("Run `python -m src.evaluate` to generate results/metrics.json and see real metrics here.")
