import streamlit as st
import requests

st.title("🎧 MusicGPT – AI Playlist Generator")

prompt = st.text_input("Describe your vibe (e.g. 'Late night drive under rain')")

if st.button("Generate Playlist"):
    with st.spinner("Grooving..."):
        res = requests.post("http://localhost:5000/generate_playlist", 
                          json={"prompt": prompt})
        if res.status_code == 200:
            st.success("✅ Playlist created!")
            playlist_id = res.json()["playlist_id"]
            st.markdown(f"[Open Playlist](https://open.spotify.com/playlist/{playlist_id})")
        else:
            st.error(f"Error: {res.json().get('error', 'Something went wrong.')}")