import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
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
    try:
        # Get available genres first
        available_genres = sp.recommendation_genres()
        logger.debug(f"Available genres: {available_genres}")

        # Filter for valid genres
        valid_genres = [g for g in genres if g in available_genres]
        if not valid_genres:
            logger.warning(f"No valid genres found in {genres}, using default 'pop'")
            valid_genres = ['pop']

        logger.debug(f"Using genres: {valid_genres}")
        
        recommendations = sp.recommendations(
            seed_genres=valid_genres[:2],  # Spotify accepts max 5 seed values
            limit=limit,
            target_valence=valence,
            target_energy=energy
        )
        
        if not recommendations['tracks']:
            raise Exception("No tracks found for given parameters")
            
        return recommendations['tracks']

    except Exception as e:
        logger.error(f"Spotify recommendation error: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing Spotify API integration...")
        
        # 1. Test client creation and authentication
        print("\n1. Creating Spotify client...")
        sp = create_spotify_client()
        user_id = get_user_id(sp)
        print(f"✓ Connected as user: {user_id}")
        
        # 2. Test genre recommendations
        print("\n2. Testing recommendations...")
        test_genres = ["pop", "rock"]
        tracks = get_recommendations(sp, test_genres, valence=0.8, energy=0.8)
        print(f"✓ Found {len(tracks)} tracks")
        print("Sample tracks:")
        for track in tracks[:3]:
            print(f"  - {track['name']} by {track['artists'][0]['name']}")
            
        # 3. Test playlist creation
        print("\n3. Testing playlist creation...")
        playlist_id = create_playlist(sp, user_id, "Test MusicGPT Playlist")
        print(f"✓ Created playlist: {playlist_id}")
        
        # 4. Test adding tracks
        print("\n4. Adding tracks to playlist...")
        track_uris = [track['uri'] for track in tracks[:5]]
        add_tracks_to_playlist(sp, playlist_id, track_uris)
        print(f"✓ Added {len(track_uris)} tracks to playlist")
        
        print("\n✨ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise