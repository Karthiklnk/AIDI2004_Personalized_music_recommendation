from flask import Flask, request, redirect, jsonify
from spotify_utils import (
    create_spotify_client, get_user_id, create_playlist, 
    add_tracks_to_playlist, get_recommendations
)
from groq_utils import extract_music_features
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
spotify_client = create_spotify_client()

@app.route("/")
def home():
    """Landing page with API documentation"""
    return jsonify({
        "name": "MusicGPT API",
        "version": "1.0",
        "description": "AI-powered music playlist generator using Spotify",
        "endpoints": {
            "POST /generate_playlist": {
                "description": "Generate a Spotify playlist based on text description",
                "request_format": {
                    "prompt": "string - Description of the desired music"
                },
                "response_format": {
                    "playlist_id": "string - Spotify playlist ID",
                    "tracks": "array - List of Spotify track URIs"
                },
                "example_request": {
                    "prompt": "Happy summer beach party music"
                }
            }
        }
    })

@app.route("/generate_playlist", methods=["POST"])
def generate_playlist():
    try:
        data = request.json
        if not data:
            return {"error": "No JSON data received"}, 400
            
        prompt = data.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}, 400

        logger.debug(f"Received prompt: {prompt}")

        # Extract features with better error handling
        try:
            features_json = extract_music_features(prompt)
            features = json.loads(features_json)
            logger.debug(f"Extracted features: {features}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"error": f"Failed to parse music features: {str(e)}"}, 500
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {"error": f"Failed to extract music features: {str(e)}"}, 500

        # Validate required fields
        if not isinstance(features.get("genres"), list):
            return {"error": "Invalid genres format"}, 500

        genres = features.get("genres", ["pop"])
        try:
            valence = float(features.get("valence", 0.5))
            energy = float(features.get("energy", 0.5))
        except (TypeError, ValueError) as e:
            logger.error(f"Value conversion error: {e}")
            return {"error": "Invalid valence or energy values"}, 500

        logger.debug(f"Using genres: {genres}, valence: {valence}, energy: {energy}")

        # Get recommendations
        try:
            tracks = get_recommendations(
                sp=spotify_client,
                genres=genres,
                valence=valence,
                energy=energy
            )
            if not tracks:
                return {"error": "No tracks found for given criteria"}, 500
                
            track_uris = [track["uri"] for track in tracks]
            logger.debug(f"Found {len(track_uris)} tracks")
            
        except Exception as e:
            logger.error(f"Spotify recommendation error: {e}")
            return {"error": f"Failed to get track recommendations: {str(e)}"}, 500

        # Create playlist
        try:
            user_id = get_user_id(spotify_client)
            playlist_id = create_playlist(
                sp=spotify_client,
                user_id=user_id,
                name=f"🎶 MusicGPT: {prompt[:30]}"
            )
            add_tracks_to_playlist(spotify_client, playlist_id, track_uris)
        except Exception as e:
            logger.error(f"Playlist creation error: {e}")
            return {"error": "Failed to create playlist"}, 500

        return {"playlist_id": playlist_id, "tracks": track_uris}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True, port=5000) 