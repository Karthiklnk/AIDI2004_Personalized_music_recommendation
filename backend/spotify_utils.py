import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

def create_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="playlist-modify-private user-read-private"
    ))

def get_user_id(sp):
    return sp.current_user()["id"]

def create_playlist(sp, user_id, name="MusicGPT Playlist"):
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False
    )
    return playlist["id"]

def add_tracks_to_playlist(sp, playlist_id, track_uris):
    sp.playlist_add_items(playlist_id, track_uris)

def get_recommendations(sp, genres, valence=0.5, energy=0.5, limit=10):
    recommendations = sp.recommendations(
        seed_genres=genres[:2],
        limit=limit,
        target_valence=valence,
        target_energy=energy
    )
    return recommendations["tracks"]

if __name__ == "__main__":
    # Example usage
    try:
        sp = create_spotify_client()
        user_id = get_user_id(sp)
        print(f"Connected to Spotify as user: {user_id}")
    except Exception as e:
        print(f"Error connecting to Spotify: {e}")