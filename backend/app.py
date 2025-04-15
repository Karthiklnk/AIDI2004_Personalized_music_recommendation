from flask import Flask, request, redirect, jsonify
from spotify_utils import (
    create_spotify_client, get_user_id, create_playlist, 
    add_tracks_to_playlist, get_recommendations
)
from groq_utils import extract_music_features
import json

app = Flask(__name__)
spotify_client = create_spotify_client()

@app.route("/")
def home():
    # OAuth flow is handled by Spotipy
    return "Please authenticate with Spotify"

@app.route("/callback")
def callback():
    return "Authentication successful!"

@app.route("/generate_playlist", methods=["POST"])
def generate_playlist():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "No prompt provided"}, 400

    features_json = extract_music_features(prompt)
    try:
        features = json.loads(features_json)
    except Exception:
        return {"error": "LLM extraction failed"}, 500

    genres = features.get("genres", ["pop"])
    valence = float(features.get("valence", 0.5))
    energy = float(features.get("energy", 0.5))

    # Use Spotipy client for all Spotify API calls
    tracks = get_recommendations(
        sp=spotify_client,
        genres=genres,
        valence=valence,
        energy=energy
    )
    track_uris = [track["uri"] for track in tracks]

    user_id = get_user_id(spotify_client)
    playlist_id = create_playlist(
        sp=spotify_client,
        user_id=user_id,
        name=f"🎶 MusicGPT: {prompt[:30]}"
    )
    add_tracks_to_playlist(spotify_client, playlist_id, track_uris)

    return {"playlist_id": playlist_id, "tracks": track_uris}

if __name__ == "__main__":
    app.run(debug=True, port=5000)