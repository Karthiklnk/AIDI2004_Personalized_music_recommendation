import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from loguru import logger
from functools import lru_cache
from flask import Flask, request, jsonify, render_template

# Configuration and setup
pd.set_option("display.max_columns", None)
load_dotenv()
logger.add(sys.stderr, level="INFO")

# Define the scope needed for our application
scope = "user-library-read user-top-read playlist-read-private user-read-recently-played user-read-email playlist-modify-private"

# Clean cache and initialize the Spotify client
if os.path.exists(".spotifycache"):
    os.remove(".spotifycache")

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_path=".spotifycache",
    )
)

# Initialize Flask app
app = Flask(__name__)

# Define available moods and time periods
AVAILABLE_MOODS = ["Happy", "Sad", "Energetic", "Calm", "Focused", "Relaxed"]
TIME_PERIODS = ["Morning", "Afternoon", "Evening", "Night"]

# --- User Data Collection Functions ---


@lru_cache(maxsize=32)
def get_users_top_tracks(limit=50, time_range="medium_term"):
    """
    Get the top tracks of the user (with caching)
    :param limit: number of tracks to return
    :param time_range: short_term (4 weeks), medium_term (6 months), long_term (years)
    :return: dataframe of the top tracks
    """
    logger.info(f"Fetching user's top {limit} tracks for {time_range}")
    try:
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range=time_range)

        if not top_tracks["items"]:
            logger.warning("No top tracks found")
            return pd.DataFrame()

        # Process tracks into dataframe format
        tracks_data = []
        for track in top_tracks["items"]:
            tracks_data.append(
                {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "popularity": track["popularity"],
                    "album_name": track["album"]["name"],
                    "album_image": track["album"]["images"][0]["url"]
                    if track["album"]["images"]
                    else None,
                    "uri": track["uri"],
                }
            )

        df = pd.DataFrame(tracks_data)

        # Sort by popularity
        df.sort_values(by="popularity", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error fetching top tracks: {e}")
        return pd.DataFrame()


def get_users_recently_played(limit=50):
    """
    Get the recently played tracks of the user
    :param limit: number of tracks to return
    :return: dataframe of the recently played tracks
    """
    logger.info(f"Fetching user's {limit} recently played tracks")
    try:
        recently_played = sp.current_user_recently_played(limit=limit)

        if not recently_played["items"]:
            logger.warning("No recently played tracks found")
            return pd.DataFrame()

        # Process tracks into dataframe format
        tracks_data = []
        for item in recently_played["items"]:
            track = item["track"]
            played_at = pd.to_datetime(item["played_at"])

            tracks_data.append(
                {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album_name": track["album"]["name"],
                    "album_image": track["album"]["images"][0]["url"]
                    if track["album"]["images"]
                    else None,
                    "played_at": played_at,
                    "day_of_week": played_at.day_name(),
                    "hour_of_day": played_at.hour,
                    "popularity": track["popularity"],
                    "uri": track["uri"],
                }
            )

        return pd.DataFrame(tracks_data)
    except Exception as e:
        logger.error(f"Error fetching recently played tracks: {e}")
        return pd.DataFrame()


def get_user_playlists():
    """
    Get all playlists of the user
    :return: list of dictionaries containing playlist information
    """
    logger.info("Fetching user's playlists")
    try:
        playlists = sp.current_user_playlists()

        playlist_list = []
        for playlist in playlists["items"]:
            playlist_list.append(
                {
                    "name": playlist["name"],
                    "id": playlist["id"],
                    "description": playlist["description"],
                    "image": playlist["images"][0]["url"]
                    if playlist["images"]
                    else None,
                    "tracks_total": playlist["tracks"]["total"],
                }
            )

        logger.info(f"Found {len(playlist_list)} playlists")
        return playlist_list
    except Exception as e:
        logger.error(f"Error fetching playlists: {e}")
        return []


# --- Playlist and Track Functions ---


def get_playlist_tracks(playlist_id):
    """
    Get all tracks from a specific playlist
    :param playlist_id: ID of the playlist
    :return: dataframe of tracks
    """
    logger.info(f"Fetching tracks from playlist {playlist_id}")
    try:
        # Get all tracks (handling pagination)
        tracks_data = []
        results = sp.playlist_tracks(playlist_id)

        for item in results["items"]:
            track = item["track"]
            if not track:
                continue

            tracks_data.append(
                {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"]
                    if track["artists"]
                    else "Unknown Artist",
                    "album_name": track["album"]["name"]
                    if "album" in track
                    else "Unknown Album",
                    "album_image": track["album"]["images"][0]["url"]
                    if "album" in track and track["album"]["images"]
                    else None,
                    "popularity": track["popularity"] if "popularity" in track else 0,
                    "uri": track["uri"],
                }
            )

        # Handle pagination
        while results["next"]:
            results = sp.next(results)
            for item in results["items"]:
                track = item["track"]
                if not track:
                    continue

                tracks_data.append(
                    {
                        "id": track["id"],
                        "name": track["name"],
                        "artist": track["artists"][0]["name"]
                        if track["artists"]
                        else "Unknown Artist",
                        "album_name": track["album"]["name"]
                        if "album" in track
                        else "Unknown Album",
                        "album_image": track["album"]["images"][0]["url"]
                        if "album" in track and track["album"]["images"]
                        else None,
                        "popularity": track["popularity"]
                        if "popularity" in track
                        else 0,
                        "uri": track["uri"],
                    }
                )

        if not tracks_data:
            logger.warning(f"No tracks found in playlist {playlist_id}")
            return pd.DataFrame()

        return pd.DataFrame(tracks_data)
    except Exception as e:
        logger.error(f"Error fetching playlist tracks: {e}")
        return pd.DataFrame()


def get_playlist_tracks_by_name(playlist_name):
    """
    Get all tracks from a specific playlist by name
    :param playlist_name: Name of the playlist
    :return: dataframe of tracks
    """
    logger.info(f"Fetching tracks from playlist {playlist_name}")
    try:
        results = sp.search(q=playlist_name, type="playlist", limit=1)

        if not results["playlists"]["items"]:
            logger.warning(f"No playlists found with name {playlist_name}")
            return pd.DataFrame()

        playlist_id = results["playlists"]["items"][0]["id"]
        return get_playlist_tracks(playlist_id)
    except Exception as e:
        logger.error(f"Error fetching playlist tracks by name: {e}")
        return pd.DataFrame()


def playlist_tracks_to_df(playlists):
    """
    Get all tracks from a list of playlists
    :param playlists: list of playlist dictionaries with id and name
    :return: dataframe of tracks
    """
    all_tracks = pd.DataFrame()
    for playlist in playlists:
        logger.info(f"Getting tracks from playlist {playlist['name']}")
        tracks = get_playlist_tracks(playlist["id"])
        if not tracks.empty:
            tracks["playlist_name"] = playlist["name"]
            all_tracks = pd.concat([all_tracks, tracks], ignore_index=True)

    # Remove duplicates
    if not all_tracks.empty:
        all_tracks.drop_duplicates(subset=["id"], inplace=True)

    return all_tracks


# --- Recommendation Functions ---


def search_mood_playlists(mood):
    """
    Search for playlists matching only the given mood (fallback)
    :param mood: mood to search for
    :return: dataframe of top tracks from mood playlists
    """
    logger.info(f"Searching for playlists with just mood: {mood}")
    try:
        results = sp.search(q=f"playlist:{mood}", type="playlist", limit=3)

        if (
            not results
            or not results.get("playlists")
            or not results["playlists"].get("items")
        ):
            logger.warning(f"No playlists found for mood: {mood}")
            return pd.DataFrame()

        # Get tracks from each playlist
        all_tracks = pd.DataFrame()
        for playlist in results["playlists"]["items"]:
            playlist_id = playlist["id"]
            playlist_name = playlist["name"]

            tracks = get_playlist_tracks(playlist_id)
            if not tracks.empty:
                tracks["playlist_name"] = playlist_name
                tracks["search_mood"] = mood
                all_tracks = pd.concat([all_tracks, tracks], ignore_index=True)

        # Get top 5 tracks by popularity from each playlist
        if not all_tracks.empty:
            top_tracks = pd.DataFrame()
            for playlist_name, group in all_tracks.groupby("playlist_name"):
                sorted_group = group.sort_values("popularity", ascending=False)
                top_5 = sorted_group.head(5)
                top_tracks = pd.concat([top_tracks, top_5], ignore_index=True)

            # Remove duplicates
            top_tracks.drop_duplicates(subset=["id"], inplace=True)

            logger.info(
                f"Found {len(top_tracks)} unique tracks from mood-only playlists"
            )
            return top_tracks

        return all_tracks

    except Exception as e:
        logger.error(f"Error in search_mood_playlists: {e}")
        return pd.DataFrame()


def get_mood_playlists(mood):
    """
    Search for playlists matching the given mood and top artists
    :param mood: mood to search for
    :return: dataframe of top tracks from mood-artist playlists
    """
    logger.info(f"Searching for {mood} playlists with top artists")

    all_tracks = pd.DataFrame()

    try:
        # Get user's top artists (limit to 3 for performance)
        top_artists_results = sp.current_user_top_artists(limit=3)

        if not top_artists_results or not top_artists_results.get("items"):
            logger.warning("No top artists found")
            # Fallback to just mood-based search
            return search_mood_playlists(mood)

        # For each artist, search for mood-related playlists
        for artist in top_artists_results["items"]:
            artist_name = artist["name"]
            logger.info(f"Searching for {mood} playlists related to {artist_name}")

            # Create search query combining artist and mood
            search_query = f"{artist_name} {mood}"

            try:
                # Search for playlists
                results = sp.search(q=search_query, type="playlist", limit=3)

                if (
                    not results
                    or not results.get("playlists")
                    or not results["playlists"].get("items")
                ):
                    logger.warning(f"No playlists found for {search_query}")
                    continue

                # Process each playlist
                for playlist in results["playlists"]["items"]:
                    playlist_id = playlist["id"]
                    playlist_name = playlist["name"]
                    logger.info(f"Getting tracks from playlist: {playlist_name}")

                    # Get tracks from the playlist
                    tracks = get_playlist_tracks(playlist_id)

                    if tracks.empty:
                        continue

                    # Add artist and mood information
                    tracks["search_artist"] = artist_name
                    tracks["search_mood"] = mood
                    tracks["playlist_name"] = playlist_name

                    # Add to our collection
                    all_tracks = pd.concat([all_tracks, tracks], ignore_index=True)

            except Exception as e:
                logger.error(
                    f"Error searching for {artist_name} {mood} playlists: {str(e)}"
                )
                continue

        # If we didn't find any tracks, try just with the mood
        if all_tracks.empty:
            logger.warning(
                f"No tracks found for artists with {mood}, falling back to mood-only search"
            )
            return search_mood_playlists(mood)

        # Sort by popularity and get top 5 tracks from each playlist
        if not all_tracks.empty:
            # Group by playlist name and get top 5 tracks by popularity from each
            top_tracks = pd.DataFrame()
            for playlist_name, group in all_tracks.groupby("playlist_name"):
                sorted_group = group.sort_values("popularity", ascending=False)
                top_5 = sorted_group.head(5)
                top_tracks = pd.concat([top_tracks, top_5], ignore_index=True)

            # Remove duplicates
            top_tracks.drop_duplicates(subset=["id"], inplace=True)

            logger.info(
                f"Found {len(top_tracks)} unique tracks from artist-mood playlists"
            )
            return top_tracks

        return all_tracks

    except Exception as e:
        logger.error(f"Error in get_mood_playlists: {str(e)}")
        return pd.DataFrame()


def create_playlist_for_user(track_df, mood, time_of_day):
    """
    Create a new playlist for the user using the provided tracks

    :param track_df: DataFrame containing tracks to add to the playlist
    :param mood: Mood used for the playlist name
    :param time_of_day: Time of day used for the playlist name
    :return: Dictionary with playlist information or error
    """
    if track_df.empty:
        logger.warning("No tracks provided to create playlist")
        return {"error": "No tracks available for playlist creation"}

    try:
        # Get user information
        user_info = sp.current_user()
        user_id = user_info["id"]

        # Create playlist name and description
        timestamp = datetime.now().strftime("%m%d_%H%M")
        playlist_name = f"mix_take_{mood}_{time_of_day}_{timestamp}"
        playlist_description = f"Custom mix for {mood} mood during {time_of_day}. Created on {datetime.now().strftime('%Y-%m-%d')}"

        logger.info(f"Creating playlist '{playlist_name}' for user {user_id}")

        # Create the playlist
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=playlist_description,
        )

        # Extract track URIs
        track_uris = track_df["uri"].tolist()

        # Add tracks to playlist
        if track_uris:
            sp.playlist_add_items(playlist["id"], track_uris)
            logger.info(f"Added {len(track_uris)} tracks to playlist {playlist_name}")

        # Return playlist information
        return {
            "success": True,
            "playlist_id": playlist["id"],
            "playlist_name": playlist_name,
            "playlist_url": playlist["external_urls"]["spotify"],
            "track_count": len(track_uris),
        }

    except Exception as e:
        logger.error(f"Error creating playlist: {str(e)}")
        return {"error": f"Failed to create playlist: {str(e)}"}


# --- Flask Routes ---


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html", moods=AVAILABLE_MOODS, times=TIME_PERIODS)


@app.route("/api/create_playlist", methods=["POST"])
def api_create_playlist():
    """Create a playlist with recommended tracks based on mood and time"""
    data = request.json
    mood = data.get("mood", "Happy")
    time_of_day = data.get("time_of_day", "Afternoon")
    limit = int(data.get("limit", 10))

    if mood not in AVAILABLE_MOODS:
        return jsonify(
            {"error": f'Invalid mood. Choose from: {", ".join(AVAILABLE_MOODS)}'}
        ), 400

    if time_of_day not in TIME_PERIODS:
        return jsonify(
            {"error": f'Invalid time period. Choose from: {", ".join(TIME_PERIODS)}'}
        ), 400

    # Get mood playlist tracks
    tracks_df = get_mood_playlists(mood)

    # Sort by popularity and get top tracks
    tracks_df.sort_values(by="popularity", ascending=False, inplace=True)
    tracks_df.reset_index(drop=True, inplace=True)
    final_tracks = tracks_df.head(limit)

    # Create the playlist
    playlist_result = create_playlist_for_user(final_tracks, mood, time_of_day)

    # Return the result
    if "error" in playlist_result:
        return jsonify({"success": False, "error": playlist_result["error"]}), 500

    return jsonify(
        {
            "success": True,
            "mood": mood,
            "time_of_day": time_of_day,
            "playlist": {
                "id": playlist_result["playlist_id"],
                "name": playlist_result["playlist_name"],
                "url": playlist_result["playlist_url"],
                "track_count": playlist_result["track_count"],
            },
            "tracks": final_tracks.to_dict("records"),
        }
    )


@app.route("/api/user_history")
def api_user_history():
    """Return the user's recently played tracks"""
    limit = int(request.args.get("limit", 10))
    recent_tracks = get_users_recently_played(limit=limit)

    if recent_tracks.empty:
        return jsonify({"tracks": []})

    return jsonify({"tracks": recent_tracks.to_dict("records")})


@app.route("/api/moods")
def api_moods():
    """Return the available moods"""
    return jsonify({"moods": AVAILABLE_MOODS})


@app.route("/api/times")
def api_times():
    """Return the available time periods"""
    return jsonify({"times": TIME_PERIODS})


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
