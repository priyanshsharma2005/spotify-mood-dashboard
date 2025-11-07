# data_processing.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from joblib import Memory
import os

# Attempt to import VADER, but download lexicon at runtime if missing
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    # will try to import again after ensuring data
    SentimentIntensityAnalyzer = None

memory = Memory(location=os.path.join(".cache"), verbose=0)

@memory.cache
def build_track_dataframe(tracks: List[Dict], audio_features: List[Dict]) -> pd.DataFrame:
    rows = []
    af_map = {af['id']: af for af in audio_features if af}
    for t in tracks:
        if not t or not t.get('id'):
            continue
        tid = t['id']
        artists = t.get('artists', [])
        artist_ids = [a['id'] for a in artists if a.get('id')]
        row = {
            "track_id": tid,
            "track_name": t.get("name"),
            "album_name": (t.get("album") or {}).get("name"),
            "artist_names": ", ".join([a.get("name","") for a in artists]),
            "artist_ids": artist_ids,
            "popularity": t.get("popularity", 0),
            "release_date": (t.get("album") or {}).get("release_date")
        }
        af = af_map.get(tid, {})
        for feat in ["danceability","energy","valence","tempo","acousticness","speechiness","instrumentalness","liveness"]:
            row[feat] = af.get(feat) if af else None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def compute_mood_score(df: pd.DataFrame, valence_weight: float = 0.7, energy_weight: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    df['valence_scaled'] = df['valence'].fillna(0)
    df['energy_scaled'] = df['energy'].fillna(0)
    df['mood_score'] = valence_weight * df['valence_scaled'] + energy_weight * df['energy_scaled']
    bins = [0.0, 0.33, 0.66, 1.0]
    labels = ["Low", "Medium", "High"]
    df['mood_category'] = pd.cut(df['mood_score'], bins=bins, labels=labels, include_lowest=True)
    return df

def aggregate_for_region(df: pd.DataFrame, region_name: str) -> Dict:
    out = {
        "region": region_name,
        "avg_mood": float(df['mood_score'].mean()) if not df['mood_score'].isna().all() else None,
        "median_mood": float(df['mood_score'].median()) if not df['mood_score'].isna().all() else None,
        "n_tracks": int(len(df))
    }
    cat_counts = df['mood_category'].value_counts(normalize=True).to_dict()
    out.update({f"pct_{k}": float(v) for k,v in cat_counts.items()})
    return out

# --- Title sentiment with safe fallback ---
def _ensure_vader():
    """
    Ensure that nltk vader lexicon is available and return a SentimentIntensityAnalyzer.
    This will download the resource if missing (works on Streamlit Cloud).
    """
    import nltk
    global SentimentIntensityAnalyzer
    if SentimentIntensityAnalyzer is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
            return SIA()
        except Exception:
            # try to download the lexicon and import again
            nltk.download('vader_lexicon')
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
            return SIA()
    else:
        return SentimentIntensityAnalyzer()

def title_sentiment_score(title: str) -> float:
    if title is None:
        return 0.0
    try:
        sia = _ensure_vader()
        s = sia.polarity_scores(title)
        return s.get('compound', 0.0)
    except Exception:
        # fallback: neutral sentiment if anything goes wrong
        return 0.0

