"""Streamlit demo for the Hybrid Movie Recommendation System.

Run with ``streamlit run app.py``. On a completely fresh clone (empty
``data/``), the first load triggers the full pipeline automatically --
download MovieLens 100K, clean it, split it, and train all five models --
so there is no manual data-prep step. Subsequent loads reuse Streamlit's
cache and the pipeline's own on-disk cache, so they're fast.

Note: this is the one file in the project where ``print`` isn't banned,
per ``claude.md`` -- but this app deliberately doesn't use it. Streamlit's
own ``st.error``/``st.info`` calls are the idiomatic way to surface
status and error messages in a Streamlit UI.
"""

import json

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
)
from src.utils import RESULTS_DIR

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")

COLD_START_USER_ID = -1  # Sentinel: guaranteed absent from MovieLens 100K (real IDs are 1-943).

SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "content": ("Content-Based", "🎭"),
    "svd": ("SVD (Collaborative)", "🤝"),
    "hybrid": ("Hybrid", "🔀"),
    "cold_start": ("Cold-Start Popularity", "🆕"),
}


@st.cache_data(show_spinner="Loading MovieLens 100K (first run downloads ~5MB and processes it)...")
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


def render_recommendations(recs: list[Recommendation], container, empty_hint: str = "") -> None:
    """Render a ranked list of recommendations with their source badge."""
    if not recs:
        container.warning(f"No recommendations returned. {empty_hint}".strip())
        return
    for rank, rec in enumerate(recs, start=1):
        title_matches = movies_df.loc[movies_df["movie_id"] == rec.movie_id, "title"]
        title = title_matches.values[0] if len(title_matches) else f"Movie #{rec.movie_id}"
        label, emoji = SOURCE_LABELS.get(rec.source, (rec.source, "•"))
        container.markdown(f"**{rank}. {title}**  \n{emoji} {label} · score {rec.score:.3f}")


try:
    data = load_data()
except DataPipelineError as exc:
    st.error(f"Failed to prepare MovieLens data: {exc}")
    st.stop()

try:
    models = load_models()
except Exception as exc:  # noqa: BLE001 - top-level UI boundary, must not crash on any model error
    st.error(f"Failed to train models: {exc}")
    st.stop()

train_df, movies_df, users_df = data["train"], data["movies"], data["users"]

st.title("\U0001f3ac Hybrid Movie Recommendation System")
st.caption(
    "Content-based (genre similarity) + SVD collaborative filtering on MovieLens 100K, "
    "combined by a weighted hybrid with a cold-start fallback for new users and unrated movies."
)

with st.sidebar:
    st.header("Controls")
    simulate_cold = st.checkbox(
        "Simulate a brand-new user (cold start)",
        value=False,
        help=(
            "Every real user in MovieLens 100K already has 20+ ratings by construction, "
            "so this is the only way to see the cold-start path trigger live."
        ),
    )
    if simulate_cold:
        selected_user_id = COLD_START_USER_ID
        st.info("Using a synthetic user with zero rating history.")
    else:
        rating_counts = train_df.groupby("user_id").size()
        user_ids = sorted(rating_counts.index.tolist())
        selected_user_id = st.selectbox(
            "Pick a user",
            user_ids,
            format_func=lambda uid: f"User {uid} ({rating_counts.get(uid, 0)} ratings)",
        )
        user_row = users_df.loc[users_df["user_id"] == selected_user_id]
        if not user_row.empty:
            row = user_row.iloc[0]
            st.caption(f"Age {row['age']} · {row['gender']} · {row['occupation']}")

    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    compare_mode = st.checkbox("Compare recommenders side-by-side", value=False)

user_label = f"User {selected_user_id}" + (" (simulated new user)" if simulate_cold else "")

if compare_mode:
    st.subheader(f"Side-by-side comparison for {user_label}")
    if simulate_cold:
        st.caption(
            "The raw Content-Based, SVD, and Hybrid models have no ratings to work with for a brand-new "
            "user and return nothing -- that gap is exactly what the cold-start wrapper (right) exists to fill."
        )
    columns = st.columns(4)
    panel_order = [
        ("content", "Content-Based \U0001f3ad"),
        ("svd", "SVD (Collaborative) \U0001f91d"),
        ("hybrid", "Hybrid \U0001f500"),
        ("popularity", "Popularity Baseline \U0001f4ca"),
    ]
    for col, (model_key, title) in zip(columns, panel_order):
        col.markdown(f"### {title}")
        recs = models[model_key].recommend_for_user(selected_user_id, n=n_recs)
        render_recommendations(recs, col, empty_hint="(no cold-start handling in this raw model)")
else:
    st.subheader(f"Top {n_recs} recommendations for {user_label}")
    recs = models["cold_start"].recommend_for_user(selected_user_id, n=n_recs)
    render_recommendations(recs, st)

st.divider()
st.subheader("Model performance (held-out test set)")
metrics_df = load_metrics_table()
if metrics_df is not None:
    st.dataframe(metrics_df, width="stretch")
    st.caption(
        "RMSE/MAE: rating-prediction error, lower is better. Precision/Recall/NDCG@K: ranking quality "
        "against the full 1,682-movie catalog (no negative sampling), higher is better. Computed on a "
        "per-user chronological train/test split -- see claude.md for why."
    )
else:
    st.info("Run `python -m src.evaluate` to generate results/metrics.json and see real metrics here.")
