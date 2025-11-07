import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from spotify_client import SpotifyClient
from data_processing import build_track_dataframe, compute_mood_score, aggregate_for_region
from utils import ensure_nltk_data, sample_mock_tracks

# Basic logging setup so debug messages appear in Streamlit logs
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Spotify Mood Analytics (India)", layout="wide")
ensure_nltk_data()

st.title("🎧 Spotify Mood Analytics — India (Prototype)")

# Sidebar controls
st.sidebar.header("Connection & Data")
use_mock = st.sidebar.checkbox("Use mock data (no Spotify API)", value=False)
client_id = st.sidebar.text_input("Spotify Client ID", value=os.getenv("SPOTIPY_CLIENT_ID", ""))
client_secret = st.sidebar.text_input("Spotify Client Secret", value=os.getenv("SPOTIPY_CLIENT_SECRET", ""), type="password")
playlist_id = st.sidebar.text_input("Playlist ID to analyze (paste URL or ID)", value="")
limit = st.sidebar.number_input("Max tracks to pull", min_value=10, max_value=1000, value=200, step=10)

st.sidebar.header("Mood Scoring")
val_w = st.sidebar.slider("Valence weight", 0.0, 1.0, 0.7)
eng_w = st.sidebar.slider("Energy weight", 0.0, 1.0, 0.3)

st.sidebar.header("Regional options")
auto_regional = st.sidebar.checkbox("Auto-detect regional playlists (city-level)", value=False)
cities_text = st.sidebar.text_area("Optional comma-separated city list (leave blank to use defaults)", value="")

st.sidebar.header("Display")
show_table = st.sidebar.checkbox("Show track table", value=True)
show_choropleth = st.sidebar.checkbox("Show India choropleth (requires geojson)", value=False)

# Connect / Load data
if use_mock:
    st.info("Using mock data for testing.")
    df_tracks = sample_mock_tracks(200)
    df_mood = compute_mood_score(df_tracks, valence_weight=val_w, energy_weight=eng_w)
    sp = None
else:
    # If sidebar fields blank, they already defaulted from env variables above.
    if not client_id or not client_secret:
        st.warning("Please provide Spotify Client ID & Secret in the sidebar OR enable 'Use mock data'.")
        st.stop()

    try:
        sp = SpotifyClient(client_id=client_id, client_secret=client_secret)
    except Exception as e:
        st.error(f"Spotify auth error: {e}")
        st.stop()

    # --- DEBUG: log full HTTP error details for playlist fetch ---
    def _debug_playlist_fetch(sp_client, playlist_id_sample):
        """
        Attempt a tiny fetch and log HTTP details on failure to App logs.
        Call this from the sidebar button.
        """
        try:
            pid = playlist_id_sample.split("playlist/")[-1].split("?")[0] if "playlist/" in playlist_id_sample else playlist_id_sample
            # do a tiny fetch to force token + request
            tracks = sp_client.get_playlist_tracks(pid, limit=2)
            st.sidebar.success(f"Debug fetch OK — {len(tracks)} tracks")
            logging.info("DEBUG: playlist fetch OK for %s", pid)
            return True
        except Exception as e:
            # Log full traceback and attempt to extract any http response attributes Spotipy might attach
            logging.exception("DEBUG: playlist fetch failed")
            try:
                resp = getattr(e, "http_response", None) or getattr(e, "response", None)
                if resp is not None:
                    try:
                        status = getattr(resp, "status", getattr(resp, "status_code", "N/A"))
                        text = getattr(resp, "text", str(resp))
                        logging.error("DEBUG: http response status: %s", status)
                        logging.error("DEBUG: http response text: %s", text)
                    except Exception:
                        logging.exception("DEBUG: failed to log http response fields")
            except Exception:
                logging.exception("DEBUG: exception while extracting http response")
            st.sidebar.error("Spotify debug fetch failed — see Manage app → App logs for full details")
            return False

    # Debug button in sidebar
    if st.sidebar.button("Run Spotify debug fetch"):
        if not playlist_id:
            st.sidebar.info("Please paste a playlist URL or ID in the Playlist field then click this button.")
        else:
            _debug_playlist_fetch(sp, playlist_id)

    if not playlist_id and not auto_regional:
        st.info("Enter a playlist ID in the sidebar (e.g. Top 50 India), or enable auto regional detection.")
        st.stop()

    # If a playlist is provided, analyze it
    if playlist_id:
        with st.spinner("Fetching playlist tracks and audio features..."):
            if "playlist/" in playlist_id:
                try:
                    playlist_id = playlist_id.split("playlist/")[1].split("?")[0]
                except:
                    pass
            try:
                tracks = sp.get_playlist_tracks(playlist_id=playlist_id, limit=int(limit))
            except Exception as e:
                # If it fails here, instruct user to run the debug button and check logs
                logging.exception("Runtime error fetching playlist in main flow")
                st.error("Failed to fetch playlist — please run 'Run Spotify debug fetch' in the sidebar and check App logs.")
                st.stop()

            track_ids = [t['id'] for t in tracks if t.get('id')]
            audio_features = sp.get_audio_features(track_ids)
            df_tracks = build_track_dataframe(tracks, audio_features)
            df_mood = compute_mood_score(df_tracks, valence_weight=val_w, energy_weight=eng_w)
    else:
        df_tracks = pd.DataFrame()
        df_mood = pd.DataFrame()

# If regional option selected, run aggregator
regional_df = None
if not use_mock and auto_regional:
    cities = [c.strip() for c in cities_text.split(",")] if cities_text.strip() else None
    with st.spinner("Searching and aggregating regional playlists... (may take a minute)"):
        from regional_aggregator import search_regional_playlists, aggregate_playlists_by_region
        playlist_map = search_regional_playlists(sp, cities=cities, per_city_limit=1, playlist_fetch_limit=100)
        regional_df = aggregate_playlists_by_region(sp, playlist_map, tracks_per_playlist=50)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracks analyzed", int(len(df_mood)) if not df_mood.empty else 0)
col2.metric("Avg mood (0-1)", round(float(df_mood['mood_score'].mean()), 3) if not df_mood.empty else "N/A")
col3.metric("Median valence", round(float(df_mood['valence'].median()),3) if not df_mood.empty else "N/A")
col4.metric("Median energy", round(float(df_mood['energy'].median()),3) if not df_mood.empty else "N/A")

st.subheader("Mood category distribution")
if not df_mood.empty:
    dist = df_mood['mood_category'].value_counts(normalize=True).reindex(["Low","Medium","High"]).fillna(0)
    st.bar_chart(dist)
else:
    st.write("No track data to show. Provide a playlist or enable auto regional detection.")

if show_table and not df_mood.empty:
    st.subheader("Top tracks (sample)")
    st.dataframe(df_mood[["track_name","artist_names","valence","energy","mood_score","mood_category","popularity"]].sort_values("mood_score", ascending=False).head(200))

if regional_df is not None:
    st.subheader("Regional mood estimates (city-level)")
    st.dataframe(regional_df.sort_values("avg_mood", ascending=False).reset_index(drop=True))

st.markdown("---")
st.caption("Prototype built using public Spotify audio features and playlist-based regional proxies.")

