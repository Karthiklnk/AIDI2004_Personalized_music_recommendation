import streamlit as st
import requests
from requests.exceptions import JSONDecodeError, ConnectionError

st.title("🎧 MusicGPT – AI Playlist Generator")

# Add debug expander
debug_expander = st.expander("Debug Info", expanded=False)

# Add description and instructions
st.markdown("""
### How it works:
1. Enter a description of the music you want (mood, activity, genre, etc.)
2. Click 'Generate Playlist' to create a custom Spotify playlist
3. Click the generated link to open your playlist in Spotify
""")

prompt = st.text_input("Describe your vibe (e.g. 'Late night drive under rain')")

if st.button("Generate Playlist"):
    if not prompt:
        st.error("Please enter a description first!")
    else:
        with st.spinner("🎵 Creating your playlist..."):
            try:
                # Make the API request
                res = requests.post(
                    "http://localhost:5000/generate_playlist", 
                    json={"prompt": prompt},
                    headers={"Content-Type": "application/json"}
                )
                
                # Show debug information in expander
                debug_info = {
                    "Status Code": res.status_code,
                    "Response Headers": dict(res.headers),
                    "Raw Response": res.text
                }
                debug_expander.json(debug_info)

                # Handle different response status codes
                if res.status_code == 200:
                    response_data = res.json()
                    st.success("✅ Playlist created successfully!")
                    playlist_id = response_data["playlist_id"]
                    st.markdown(f"""
                    ### 🎉 Your playlist is ready!
                    [Open in Spotify](https://open.spotify.com/playlist/{playlist_id})
                    """)
                elif res.status_code == 500:
                    st.error("❌ The server encountered an error while processing your request.")
                    st.info("💡 Try simplifying your description or try again in a moment.")
                else:
                    st.warning(f"⚠️ Unexpected response (Status {res.status_code})")
                    
            except ConnectionError:
                st.error("🔌 Cannot connect to the server. Is it running?")
                st.info("💡 Make sure the Flask backend is running on http://localhost:5000")
            except Exception as e:
                st.error(f"💥 An unexpected error occurred: {str(e)}")