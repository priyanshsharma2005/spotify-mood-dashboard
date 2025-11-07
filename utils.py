# utils.py
import pandas as pd
import numpy as np
import os
from typing import List
import logging

logger = logging.getLogger(__name__)

def ensure_nltk_data():
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except Exception:
        nltk.download('vader_lexicon')

def sample_mock_tracks(n=50):
    import random
    rows = []
    for i in range(n):
        rows.append({
            "track_id": f"mock_{i}",
            "track_name": f"Mock Song {i}",
            "album_name": f"Album {i%5}",
            "artist_names": f"Artist {i%
