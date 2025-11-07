# regional_aggregator.py
from typing import List, Dict, Optional
from spotify_client import SpotifyClient
from data_processing import build_track_dataframe, compute_mood_score
import pandas as pd
import time

# A compact default city list for India
COMMON_CITIES = [
    "Mumbai","Delhi","Bengaluru","Kolkata","Chennai","Hyderabad","Pune","Ahmedabad",
    "Jaipur","Lucknow","Surat","Nagpur","Indore","Bhopal","Visakhapatnam",
    "Patna","Vadodara","Ghaziabad","Ludhiana"
]

def search_regional_playlists(sp: SpotifyClient, query_template: str = "Top 50 {city}", cities: Optional[List[str]] = None, per_city_limit: int = 1, playlist_fetch_limit: int = 100) -> Dict[str, List[Dict]]:
    """
    Search Spotify for playlists for each city and return a dict city -> list of playlist objects.
    query_template controls the search string (we try one pattern here).
    """
    cities = cities or COMMON_CITIES
    out: Dict[str, List[Dict]] = {}
    for city in cities:
        q = query_template.format(city=city)
        try:
            found = sp.search_playlists(q, limit=per_city_limit)
            out[city] = found or []
            # small sleep to be polite to API
            time.sleep(0.1)
        except Exception:
            # on any error, record an empty list for the city
            out[city] = []
    return out

def aggregate_playlists_by_region(sp: SpotifyClient, playlist_ids_by_city: Dict[str, List[Dict]], tracks_per_playlist: int = 50) -> pd.DataFrame:
    """
    For each city, fetch tracks from the first playlist(s) and compute aggregated mood metrics.
    Returns a dataframe with one row per city.
    """
    rows = []
    for city, playlists in playlist_ids_by_city.items():
        all_tracks = []
        for p in playlists:
            pid = p.get('id')
            if not pid:
                continue
            try:
                tracks = sp.get_playlist_tracks(pid, limit=tracks_per_playlist)
                all_tracks.extend(tracks)
                # polite pause
                time.sleep(0.1)
            except Exception:
                # skip if any playlist fetch fails
                continue

        if not all_tracks:
            rows.append({
                "city": city,
                "n_tracks": 0,
                "avg_mood": None,
                "median_mood": None
            })
            continue

        track_ids = [t['id'] for t in all_tracks if t.get('id')]
        audio_feats = sp.get_audio_features(track_ids)
        df_tracks = build_track_dataframe(all_tracks, audio_feats)
        df_mood = compute_mood_score(df_tracks)
        rows.append({
            "city": city,
            "n_tracks": int(len(df_mood)),
            "avg_mood": float(df_mood['mood_score'].mean()) if not df_mood['mood_score'].isna().all() else None,
            "median_mood": float(df_mood['mood_score'].median()) if not df_mood['mood_score'].isna().all() else None
        })

    return pd.DataFrame(rows)
