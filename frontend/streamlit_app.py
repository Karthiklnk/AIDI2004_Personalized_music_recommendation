import streamlit as st
import requests

st.title("🎧 MusicGPT – AI Playlist Generator")

prompt = st.text_input("Describe your vibe (e.g. 'Late night drive under rain')")
access_token = st.text_input("Paste your Spotify access token")

if st.button("Generate Playlist"):
    with st.spinner("Grooving..."):
        res = requests.post("http://localhost:5000/generate_playlist", json={"prompt": prompt, "access_token": access_token})
        if res.status_code == 200:
            st.success("✅ Playlist created!")
            playlist_id = res.json()["playlist_id"]
            st.markdown(f"[Open Playlist](https://open.spotify.com/playlist/{playlist_id})")
        else:
            st.error("Something went wrong.")
