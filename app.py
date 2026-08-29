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
configured, falling back to a placeholder poster per card. The UI is a
dark, poster-forward "streaming service" theme (Phase 10): a slim top
navbar, a single horizontal toolbar in place of the old sidebar, minimal
per-card chrome (detail appears on hover), and Netflix-style horizontal
shelves for the model-comparison view. Theme base colors live in
``.streamlit/config.toml``; bespoke layout (navbar, hero, cards, shelves)
is CSS injected below.

Note: this is the one file in the project where ``print`` isn't banned,
per ``claude.md`` -- but this app deliberately doesn't use it. Streamlit's
own ``st.error``/``st.info`` calls are the idiomatic way to surface
status and error messages in a Streamlit UI.
"""

import html
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
    recommend_similar_to_picks,
)
from src.posters import get_poster_urls_by_imdb_id

st.set_page_config(
    page_title="MovieMatch",
    page_icon="\U0001f3ac",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sentinel guaranteed absent from this dataset's real (free-text, human-typed) user ids.
COLD_START_USER_ID = "__simulated_new_user__"

# label, emoji (used in shelf headers / tooltips only, not on every card), accent color.
SOURCE_META: dict[str, tuple[str, str, str]] = {
    "content": ("Content-Based", "\U0001f3ad", "#2dd4bf"),
    "svd": ("SVD (Collaborative)", "\U0001f91d", "#a78bfa"),
    "hybrid": ("Hybrid", "\U0001f500", "#f5c518"),
    "cold_start": ("Popularity", "\U0001f4ca", "#38bdf8"),
}

NETFLIX_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #141414;
    --bg-elev: #1f1f1f;
    --bg-card: #181818;
    --border: rgba(255,255,255,0.08);
    --text: #f5f5f1;
    --text-muted: #a3a3a3;
    --accent-red: #e50914;
    --font: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

html, body, [class*="css"] { font-family: var(--font); }

.stApp { background: var(--bg); }
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    background: radial-gradient(ellipse 1100px 600px at 50% -10%, rgba(229,9,20,0.16), transparent 60%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, [data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; } /* controls live in the top toolbar instead */

.block-container { padding-top: 1.75rem; max-width: 1400px; }

/* Navbar */
.navbar {
    display: flex; justify-content: center;
    padding-bottom: 1rem; margin-bottom: .25rem;
    border-bottom: 1px solid var(--border);
    position: relative; z-index: 1;
}
.navbar-brand {
    font-weight: 900; font-size: 1.4rem; letter-spacing: .01em;
    color: var(--accent-red); text-transform: uppercase;
}

/* Hero */
.hero { padding: 1.5rem 0 1.1rem; position: relative; z-index: 1; text-align: center; }
.hero-title {
    font-weight: 800; font-size: 2.3rem; line-height: 1.15;
    color: var(--text); margin: 0 0 .5rem; letter-spacing: -.01em;
}
.hero-tagline {
    font-size: .92rem; color: var(--text-muted); line-height: 1.5; text-align: center;
    width: fit-content; max-width: 68ch;
    /* !important is load-bearing, confirmed via computed styles: Streamlit's own
    generated CSS (a rule like ".st-emotion-cache-XXXX p") sets margin-left/
    margin-right: 0px on every <p> with higher specificity (one class + one element
    selector) than a single-class selector here, so a plain `margin: 0 auto` loses
    and the box sits flush left despite text-align: center being applied correctly. */
    margin: 0 auto !important;
}
.key-hint { font-size: .78rem; color: var(--text-muted); margin: -.4rem 0 .5rem; }

/* Toolbar -- replaces the old sidebar; one horizontal bar of native widgets */
[data-testid="stHorizontalBlock"] {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem .25rem;
    margin-bottom: 1.75rem;
    position: relative; z-index: 1;
}
[data-testid="stHorizontalBlock"] label { font-size: .72rem !important; }

.stApp h3 { font-weight: 700; font-size: 1.25rem; color: var(--text); letter-spacing: -.01em; }

/* Shelves -- Netflix-style horizontal scroll rows, used in compare mode */
.shelf { position: relative; z-index: 1; margin-bottom: 1.6rem; }
.shelf-header { display: flex; align-items: center; gap: .5rem; margin-bottom: .6rem; }
.shelf-title { font-weight: 700; font-size: 1.02rem; color: var(--text); }
.shelf-row {
    display: flex; gap: .8rem; overflow-x: auto; padding: .15rem .15rem 1rem;
    scroll-snap-type: x proximity;
}
.shelf-row::-webkit-scrollbar { height: 6px; }
.shelf-row::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.shelf-row::-webkit-scrollbar-track { background: transparent; }
.shelf-row .movie-card { flex: 0 0 140px; scroll-snap-align: start; }

/* Grid, used in single-model view */
.movie-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1.1rem; margin-top: .25rem; position: relative; z-index: 1;
}

/* Movie card -- poster-forward, minimal chrome; detail surfaces on hover */
.movie-card { position: relative; }
.poster-wrap {
    position: relative; width: 100%; aspect-ratio: 2 / 3; border-radius: 6px;
    background: var(--bg-card); overflow: hidden;
    transition: transform .2s ease, box-shadow .2s ease;
}
.movie-card:hover .poster-wrap { transform: scale(1.06); box-shadow: 0 16px 32px rgba(0,0,0,.6); z-index: 5; }
.poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-placeholder {
    width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: .35rem; background: linear-gradient(160deg, #232323, #141414);
    color: rgba(245,245,241,0.3); font-size: 1.8rem; text-align: center; padding: .6rem;
}
.poster-placeholder span { font-size: .6rem; color: rgba(245,245,241,0.45); line-height: 1.3; }

.rank-dot {
    position: absolute; top: 6px; left: 6px; width: 20px; height: 20px; border-radius: 50%;
    background: rgba(0,0,0,.65); color: #fff; font-size: .65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.source-dot {
    position: absolute; top: 8px; right: 8px; width: 10px; height: 10px; border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(0,0,0,.5);
}
.hover-meta {
    position: absolute; left: 0; right: 0; bottom: 0; padding: 1.6rem .5rem .4rem;
    background: linear-gradient(to top, rgba(0,0,0,.92) 25%, transparent 100%);
    opacity: 0; transform: translateY(4px);
    transition: opacity .18s ease, transform .18s ease;
    display: flex; justify-content: space-between; align-items: baseline; gap: .3rem;
}
.movie-card:hover .hover-meta { opacity: 1; transform: translateY(0); }
.hover-label { font-size: .62rem; font-weight: 700; }
.hover-score { font-size: .62rem; color: var(--text-muted); }

.card-title {
    font-size: .78rem; font-weight: 600; color: var(--text); margin-top: .45rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

.empty-hint { font-size: .82rem; color: var(--text-muted); padding: .5rem 0 1rem; }
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


def _movie_card_html(rank: int, title: str, rec: Recommendation, poster_url: str | None) -> str:
    """Render one recommendation as a poster-forward movie-card HTML snippet.

    Chrome is deliberately minimal: a rank dot and a colored source dot sit on
    the poster itself, and the model label + score only surface in a hover
    overlay -- keeping the default (and any static screenshot) clean while
    still exposing which model produced each pick for anyone who hovers.
    """
    label, _emoji, color = SOURCE_META.get(rec.source, (rec.source, "•", "#a3a3a3"))
    safe_title = html.escape(title)

    if poster_url:
        poster_html = f'<img src="{html.escape(poster_url)}" alt="{safe_title}" loading="lazy">'
    else:
        short_title = title if len(title) <= 40 else title[:37] + "..."
        poster_html = f'<div class="poster-placeholder">\U0001f3ac<span>{html.escape(short_title)}</span></div>'

    return f"""
<div class="movie-card">
  <div class="poster-wrap">
    {poster_html}
    <span class="rank-dot">{rank}</span>
    <span class="source-dot" style="background:{color};" title="{html.escape(label)}"></span>
    <div class="hover-meta">
      <span class="hover-label" style="color:{color};">{html.escape(label)}</span>
      <span class="hover-score">{rec.score:.2f}</span>
    </div>
  </div>
  <div class="card-title" title="{safe_title}">{safe_title}</div>
</div>
"""


def render_grid(
    recs: list[Recommendation],
    movie_id_to_title: dict[str, str],
    poster_map: dict[str, str | None],
    *,
    empty_hint: str = "",
) -> None:
    """Render a ranked list of recommendations as a responsive poster grid."""
    if not recs:
        st.markdown(f'<div class="empty-hint">Nothing to show yet. {empty_hint}</div>'.strip(), unsafe_allow_html=True)
        return
    cards = [
        _movie_card_html(rank, movie_id_to_title.get(rec.movie_id, f"Movie {rec.movie_id}"), rec, poster_map.get(rec.movie_id))
        for rank, rec in enumerate(recs, start=1)
    ]
    st.markdown(f'<div class="movie-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_shelf(
    icon: str,
    title: str,
    recs: list[Recommendation],
    movie_id_to_title: dict[str, str],
    poster_map: dict[str, str | None],
    *,
    empty_hint: str = "",
) -> None:
    """Render one model's recommendations as a Netflix-style horizontal scroll shelf."""
    if recs:
        cards = [
            _movie_card_html(
                rank, movie_id_to_title.get(rec.movie_id, f"Movie {rec.movie_id}"), rec, poster_map.get(rec.movie_id)
            )
            for rank, rec in enumerate(recs, start=1)
        ]
        body = f'<div class="shelf-row">{"".join(cards)}</div>'
    else:
        body = f'<div class="empty-hint">Nothing to show yet. {empty_hint}</div>'.strip()
    st.markdown(
        f"""
<div class="shelf">
  <div class="shelf-header"><span>{icon}</span><span class="shelf-title">{html.escape(title)}</span></div>
  {body}
</div>
""",
        unsafe_allow_html=True,
    )


def recommend_for_new_user(
    content_model: ContentBasedRecommender,
    svd_model: SVDRecommender,
    popularity_model: PopularityRecommender,
    movies_df: pd.DataFrame,
    picked_movie_ids: list[str],
    n: int,
) -> list[Recommendation]:
    """Recommend movies for a brand-new user based on movies they just said they like.

    With no picks yet, this is pure trending popularity -- there's nothing to
    be "similar to". Once there are picks, this is real similarity, not a
    single blended genre guess: see :func:`recommend_similar_to_picks` for the
    content + collaborative + language-aware logic.

    Args:
        content_model: A fitted ``ContentBasedRecommender``.
        svd_model: A fitted ``SVDRecommender``.
        popularity_model: A fitted ``PopularityRecommender``.
        movies_df: Movie metadata.
        picked_movie_ids: IDs of movies the user picked as "movies I like".
        n: Number of recommendations to return.

    Returns:
        Up to ``n`` :class:`Recommendation` objects, none of them one of the
        user's own picks.
    """
    if not picked_movie_ids:
        return popularity_model.recommend_for_genre_profile(genre_weights=None, n=n, exclude_seen=False)
    return recommend_similar_to_picks(picked_movie_ids, content_model, svd_model, popularity_model, movies_df, n=n)


st.markdown(NETFLIX_CSS, unsafe_allow_html=True)

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
<div class="navbar">
  <span class="navbar-brand">MovieMatch</span>
</div>
<div class="hero">
  <h1 class="hero-title">Indian Movie Recommendation</h1>
  <p class="hero-tagline">Get recommendations for movies to watch.</p>
</div>
""",
    unsafe_allow_html=True,
)
if not tmdb_api_key:
    st.markdown('<div class="key-hint">No poster API key configured &mdash; showing placeholders.</div>', unsafe_allow_html=True)

toolbar = st.columns([1, 2.2, 1, 1])
with toolbar[0]:
    simulate_cold = st.checkbox("New user", value=False, help="Simulate a brand-new user with no rating history.")

picked_movie_ids: list[str] = []
selected_user_id: str = COLD_START_USER_ID
with toolbar[1]:
    if simulate_cold:
        sorted_movie_ids = sorted(movie_id_to_title, key=lambda mid: movie_id_to_title[mid])
        picked_movie_ids = st.multiselect(
            "Movies you like",
            options=sorted_movie_ids,
            format_func=lambda mid: movie_id_to_title.get(mid, mid),
        )
    else:
        rating_counts = train_df.groupby("user_id").size()
        user_ids = sorted(rating_counts.index.tolist())
        selected_user_id = st.selectbox(
            "User", user_ids, format_func=lambda uid: f"{uid} ({rating_counts.get(uid, 0)} ratings)"
        )
with toolbar[2]:
    n_recs = st.slider("Results", min_value=5, max_value=20, value=10)
with toolbar[3]:
    compare_mode = st.checkbox("Compare models", value=False)

user_label = "a new user" if simulate_cold else f"User {selected_user_id}"

if compare_mode:
    st.subheader(f"Comparing models — {user_label}")
    # The 4th shelf is normally the plain popularity baseline, but once the user has
    # live picks it's actually showing the content+collaborative similarity blend
    # (see recommend_for_new_user) -- relabeled so the header stays honest about
    # what's actually being ranked.
    fourth_shelf_label = "Similar To Your Picks" if (simulate_cold and picked_movie_ids) else "Popularity Baseline"
    panel_order = [
        ("content", "Content-Based", "\U0001f3ad"),
        ("svd", "SVD (Collaborative)", "\U0001f91d"),
        ("hybrid", "Hybrid", "\U0001f500"),
        ("popularity", fourth_shelf_label, "\U0001f4ca"),
    ]
    panel_recs = {}
    for key, _, _ in panel_order:
        if simulate_cold and key == "popularity":
            panel_recs[key] = recommend_for_new_user(
                models["content"], models["svd"], models["popularity"], movies_df, picked_movie_ids, n_recs
            )
        else:
            panel_recs[key] = models[key].recommend_for_user(selected_user_id, n=n_recs)
    needed_movie_ids = {rec.movie_id for recs in panel_recs.values() for rec in recs}
    poster_map = load_posters(tuple(sorted(needed_movie_ids)), tmdb_api_key)

    for key, title, icon in panel_order:
        empty_hint = "Needs rating history." if simulate_cold and key != "popularity" else ""
        render_shelf(icon, title, panel_recs[key], movie_id_to_title, poster_map, empty_hint=empty_hint)
else:
    st.subheader(f"Recommended for {user_label}")
    if simulate_cold:
        recs = recommend_for_new_user(
            models["content"], models["svd"], models["popularity"], movies_df, picked_movie_ids, n_recs
        )
    else:
        recs = models["cold_start"].recommend_for_user(selected_user_id, n=n_recs)
    needed_movie_ids = {rec.movie_id for rec in recs}
    poster_map = load_posters(tuple(sorted(needed_movie_ids)), tmdb_api_key)
    render_grid(recs, movie_id_to_title, poster_map)
