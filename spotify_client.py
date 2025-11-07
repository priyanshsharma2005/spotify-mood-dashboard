# spotify_client.py
import os
import time
from typing import List, Dict, Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyClient:
    def __init__(self, client_id: Optional[str]=None, client_secret: Optional[str]=None):
        client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError("Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET as environment variables or pass them in.")

        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10, retries=3)

    def get_playlist_tracks(self, playlist_id: str, limit: int = 100) -> List[Dict]:
        results = []
        offset = 0
        while True:
            chunk = self.sp.playlist_items(playlist_id, offset=offset, additional_types=["track"])
            items = chunk.get("items", [])
            for it in items:
                track = it.get("track")
                if track:
                    results.append(track)
            if len(items) == 0 or chunk.get("next") is None or len(results) >= limit:
                break
            offset += len(items)
            time.sleep(0.1)
        return results[:limit]

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        out = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            feats = self.sp.audio_features(batch)
            out.extend(feats)
        return out

    def search_playlists(self, query: str, limit: int=20) -> List[Dict]:
        res = self.sp.search(q=query, type='playlist', limit=limit)
        return res.get("playlists", {}).get("items", [])
