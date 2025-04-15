"""
Personalized music recommendation web app for mobile
Input: song playlist history, likes and dislikes of the user
Output: Song recommendations based on user interest, mood and timing

"""
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

scope = "user-library-read user-top-read playlist-read-private user-read-recently-played"

# Create Spotify client with proper authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope=scope,
    cache_path=".spotifycache"
))




def get_playlists():
    """
    Get all playlists of the user
    """
    playlists = sp.current_user_playlists()
    playlist_list = []
    for playlist in playlists['items']:
        playlist_list.append({
            'name': playlist['name'],
            'id': playlist['id'],
            'description': playlist['description'],
            'image': playlist['images'][0]['url'] if playlist['images'] else None
        })
    return playlist_list



if __name__ == "__main__":
    playlists = get_playlists()
    print("Playlists:")
    for playlist in playlists:
        print(f"Name: {playlist['name']}, ID: {playlist['id']}, Description: {playlist['description']}, Image: {playlist['image']}")