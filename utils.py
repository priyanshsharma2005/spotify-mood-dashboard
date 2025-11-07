# utils.py
import pandas as pd
import numpy as np
import os
from typing import List
import logging

logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure that the NLTK VADER lexicon is available (downloads it if missing)."""
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except Exception:
        nltk.download('vader_lexicon')

def sample_mock_tracks(n: int = 50) -> pd.DataFrame:
    """Generate mock track data for offline demo/testing."""
    import random
    rows = []
    for i in range(n):
        rows.append({
            "track_id": f"mock_{i}",
            "track_name": f"Mock Song {i}",
            "album_name": f"Album {i % 5}",
            "artist_names": f"Artist {i}",
            "valence": random.random(),
            "energy": random.random(),
            "popularity": random.randint(20, 100)
        })
    df = pd.DataFrame(rows)
    df["mood_score"] = 0.7 * df["valence"] + 0.3 * df["energy"]
    bins = [0.0, 0.33, 0.66, 1.0]
    labels = ["Low", "Medium", "High"]
    df["mood_category"] = pd.cut(df["mood_score"], bins=bins, labels=labels, include_lowest=True)
    return df
