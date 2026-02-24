from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List, Dict
import uuid
import os
import json
import numpy as np
from pathlib import Path
import base64
import hashlib
import secrets
import time
import requests
from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from collections import defaultdict
from collections import Counter
from typing import List, Optional 
import re
from fastapi.staticfiles import StaticFiles






from bs4 import BeautifulSoup
from .vibe_v5 import (
    compute_composite_score_v5,
    build_playlist_queries_v5,
    parse_year,
    analyze_lyrics,
    analyze_album_art,
    WEIGHTS_V5,
    lastfm_get_artist_tags,
    lastfm_get_similar_artists,
)
#lyrics caching flow to hopefully optimize search
from .lyrics_cache import get_cached_lyrics, cache_lyrics, get_cache_stats



load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SESSION_SECRET = os.getenv("SESSION_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_REDIRECT_URI or not SESSION_SECRET:
    raise RuntimeError("Missing env vars.")



INDEX_PATH = "index.json"

###Helper Functions

def spotify_get(request: Request, url: str, params: dict | None = None):
    token = request.session.get("spotify_access_token")
    if not token:
        return None, {"error": "Not logged in. Go to /auth/login first."}

    headers = {"Authorization": f"Bearer {token}"}

    print("SPOTIFY GET:", url, "params:", params)

    r = requests.get(url, headers=headers, params=params, timeout=20)

    # If token expired, you'll see 401 — we'll add refresh later
    if r.status_code != 200:
        return None, {"error": "Spotify API error", "status": r.status_code, "body": r.text}

    return r.json(), None

def spotify_get_json(request: Request, path: str, params: dict | None = None):
    base = "https://api.spotify.com/v1"
    return spotify_get(request, f"{base}{path}", params=params)

def apply_diversity_caps(results, max_per_artist=5, max_from_seed_album=5):
    artist_counts = defaultdict(int)
    seed_album_count = 0
    out = []

    for r in results:
        # cap seed album spam
        if r.get("source") == "album":
            if seed_album_count >= max_from_seed_album:
                continue
            seed_album_count += 1

        # cap per-artist spam (use first artist as primary)
        primary_artist = (r.get("artists") or [""])[0]
        if primary_artist:
            if artist_counts[primary_artist] >= max_per_artist:
                continue
            artist_counts[primary_artist] += 1

        out.append(r)

    return out


def dedupe_tracks(tracks):
    seen = set()
    out = []
    for t in tracks:
        tid = t.get("id")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(t)
    return out

def get_artist_genres(request: Request, artist_ids: list[str]):
    if not artist_ids:
        return []
    ids = ",".join(artist_ids[:50])
    data, err = spotify_get_json(request, "/artists", params={"ids": ids})
    if err:
        return []
    genres = []
    for a in data.get("artists", []):
        genres.extend(a.get("genres", []))
    # keep top unique genres
    seen = set()
    out = []
    for g in genres:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out[:5]

BAD_WORDS = [
    "tribute", "cover", "karaoke", "piano", "lullaby", "renditions", "version",
    "kids", "kidz", "rockabye", "instrumental", "acoustic", "edit", "remix",
    "nightcore", "sped up", "slowed", "8d", "string quartet", "vsq", "live", "remastered", "deluxe", "anniversary", "radio edit"
]

def looks_like_cover(item: dict) -> bool:
    name = (item.get("name") or "").lower()
    album = (item.get("album") or "").lower()
    artists = " ".join(item.get("artists") or []).lower()

    text = f"{name} {album} {artists}"
    return any(w in text for w in BAD_WORDS)

BAD_SEASONAL = [
    "christmas", "xmas", "holiday", "holidays", "santa", "noel",
    "winter", "snow", "jingle", "carol", "hanukkah"
]

def looks_like_seasonal(item: dict) -> bool:
    name = (item.get("name") or "").lower()
    album = (item.get("album") or "").lower()
    artists = " ".join(item.get("artists") or []).lower()
    text = f"{name} {album} {artists}"
    return any(w in text for w in BAD_SEASONAL)


def normalize_for_filtering(candidate: dict) -> dict:
    # make sure we always have name/artists/album keys for filtering
    return {
        "id": candidate.get("id"),
        "name": candidate.get("name", ""),
        "artists": candidate.get("artists", []),
        "album": candidate.get("album", ""),   # may be empty until hydration
        "source": candidate.get("source", ""),
        "artist_ids": candidate.get("artist_ids", []),
    }


def add_playlist_tracks_as_candidates(request: Request, query: str, candidates: list, max_playlists: int = 1, per_playlist: int = 8):

    # make sure we don't double-append "playlist"
    q = query
    if "playlist" not in q.lower():
        q = f"{q} playlist"

    pdata, err = spotify_get_json(
        request,
        "/search",
        params={
            "q": q,
            "type": "playlist",
            "limit": max_playlists,
            "market": "US"
        }
    )

    if err:
        return

    playlists = pdata.get("playlists", {}).get("items", [])

    def is_good_playlist(p: dict) -> bool:
        name = (p.get("name") or "").lower()
        desc = (p.get("description") or "").lower()
        owner = ((p.get("owner") or {}).get("display_name") or "").lower()

        text = f"{name} {desc} {owner}"

        BAD_PLAYLIST_HINTS = [
            "top hits", "today", "viral", "throwback", "all out", "mix",
            "best of", "ultimate", "100", "00s", "90s", "80s", "70s",
            "party", "wedding", "karaoke", "family", "kids"
        ]

        return not any(x in text for x in BAD_PLAYLIST_HINTS)

    for p in playlists:
        if not p:
            continue
        if not is_good_playlist(p):
            continue

        pid = p.get("id")

        tracks_data, err = spotify_get_json(
            request,
            f"/playlists/{pid}/tracks",
            params={"limit": per_playlist, "market": "US"}
        )

        if err:
            continue

        for item in tracks_data.get("items", []):
            if not item:
                continue

            t = item.get("track")
            if not t or not t.get("id"):
                continue

            candidates.append({
                "id": t["id"],
                "name": t.get("name", ""),
                "artists": [a["name"] for a in t.get("artists", [])],
                "artist_ids": [a["id"] for a in t.get("artists", []) if a.get("id")],
                "album": (t.get("album", {}) or {}).get("name", ""),
                "source": "playlist"
            })




def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)         # remove (...) like "(Remastered 2011)"
    s = re.sub(r"\[.*?\]", "", s)         # remove [...] 
    s = re.sub(r"[^a-z0-9\s]", "", s)     # remove punctuation
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_year(date_str: str | None) -> int | None:
    # Spotify release_date can be "YYYY", "YYYY-MM-DD", or "YYYY-MM"
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except Exception:
        return None






    
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

def make_code_verifier() -> str:
    return _b64url(secrets.token_bytes(64))

def make_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _b64url(digest)




app = FastAPI(title="Vibe Finder")

app.mount("/static", StaticFiles(directory="static"), name="static") # for clouds

@app.get("/routes")
def routes():
    return [r.path for r in app.routes]


app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vibe Finder</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/color-thief/2.3.0/color-thief.umd.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        :root{
            --color-1: #1a1a2e;
            --color-2: #16213e;
            --color-3: #0f3460;
            --color-4: #533483;
            --color-5: #2c3e50;
            --cloud-tint: rgba(255, 255, 255, 0.9);
            --rain-intensity: 0;
            --sun-intensity: 0;
            --mist-intensity: 0;
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.15);
            --text-primary: rgba(255, 255, 255, 0.95);
            --text-secondary: rgba(255, 255, 255, 0.6);
            
            transition: --color-1 4s ease-in-out,
                        --color-2 4s ease-in-out,
                        --color-3 4s ease-in-out,
                        --color-4 4s ease-in-out;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
        }
        
        /* ============== SKY GRADIENT (Album Colors) ============== */
        .sky {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -10;
            background: linear-gradient(
                180deg,
                var(--color-1) 0%,
                var(--color-2) 30%,
                var(--color-3) 60%,
                var(--color-4) 100%
            );
            transition: all 4s ease;
        }
        
    
        /* ============== TOP SKY CLOUD LAYER ============== */
        .clouds-container{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 55vh;        /* Only top portion */
        overflow: hidden;
        pointer-events: none;
        z-index: -8;
        }

        /* Base cloud style */
        .cloud{
            position: absolute;position: absolute;
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
            will-change: transform;
            filter: blur(0.4px);
        }



       /* subtle movement (not drifting across the whole screen) */
        @keyframes floaty {
        0%, 100% { transform: translateY(0) translateX(0); }
        50%      { transform: translateY(10px) translateX(12px); }
        }
        @keyframes floaty2 {
        0%, 100% { transform: translateY(0) translateX(0); }
        50%      { transform: translateY(12px) translateX(-10px); }
        }

        /* LEFT cloud */
        .cloud-1{
        top: -120px;
        left: -180px;
        width: 1300px;
        height: 650px;
        background-image: url('/static/clouds/cloud1.png');
        opacity: 0.65;
        animation: floaty 22s ease-in-out infinite;
        }

        /* MID-LEFT cloud */
        .cloud-2{
        top: -160px;
        left: 18vw;
        width: 1200px;
        height: 600px;
        background-image: url('/static/clouds/cloud2.png');
        opacity: 0.92;
        animation: floaty2 26s ease-in-out infinite;
        }

        /* MID-RIGHT cloud */
        .cloud-3{
        top: -200px;
        left: 52vw;
        width: 1250px;
        height: 620px;
        background-image: url('/static/clouds/cloud1.png');
        opacity: 0.60;
        animation: floaty 30s ease-in-out infinite;
        }

        /* RIGHT cloud */
        .cloud-4{
        top: -120px;
        right: -260px;
        width: 1400px;
        height: 700px;
        background-image: url('/static/clouds/cloud2.png');
        opacity: 0.90;
        animation: floaty2 28s ease-in-out infinite;
        }

        /* SMALL filler puff (optional, helps “bridge” gaps) */
        .cloud-5{
        top: 40px;
        left: 40vw;
        width: 700px;
        height: 350px;
        background-image: url('/static/clouds/cloud2.png');
        opacity: 0.80;
        animation: floaty 34s ease-in-out infinite;
        }

        .cloud-6{
            top: -8px;
            left: 6%;
            width: 900px;
            height: 420px;
            opacity: 0.50;
            background-image: url('/static/clouds/cloud1.png'); /* <-- change to your new image file */
            animation: floaty 34s ease-in-out infinite;
        }

        .cloud-7{
            top: -8px;
            left: -190%;
            width: 900px;
            height: 420px;
            opacity: 0.50;
            background-image: url('/static/clouds/cloud4.png'); /* <-- change to your new image file */
            animation: floaty 34s ease-in-out infinite;
        }

        






        
        /* ============== RAIN ============== */
        .rain-container {
            position: fixed;
            top: 12vh;
            left: 0;
            width: 100%;
            height: 62vh;
            z-index: -7;
            pointer-events: none;
            opacity: 0;
            transition: opacity 1.5s ease;
            overflow: hidden;
        }

        .rain-container.active {
            opacity: var(--rain-intensity, 0.5);
        }

        .rain-drop {
            position: absolute;
            width: 2px;
            height: 20px;
            background: linear-gradient(transparent, rgba(200, 220, 255, 0.7));
            border-radius: 2px;
            transform: rotate(12deg);
            animation: rainFall linear infinite;
        }

        @keyframes rainFall {
            0% { transform: translateY(-5px) rotate(12deg); }
            100% { transform: translateY(62vh) rotate(12deg); }
        }

        /* ============== SUN RAYS ============== */
      
        .sun-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -7;
            pointer-events: none;
            opacity: 0;
            transition: opacity 1.5s ease;
            background: radial-gradient(
                circle at 85% 12%,
                rgba(255, 255, 240, 0.95) 0%,
                rgba(255, 250, 220, 0.7) 2%,
                rgba(255, 245, 180, 0.4) 5%,
                rgba(255, 240, 150, 0.2) 10%,
                rgba(255, 230, 120, 0.1) 15%,
                transparent 25%
            );
        }

        .sun-container.active {
            opacity: var(--sun-intensity, 0.5);
        }

        .sun, .sun-rays {
            display: none;
        }
       
        /* ============== MIST/FOG ============== */
        .mist-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 50%;
            z-index: -6;
            pointer-events: none;
            opacity: 0;
            transition: opacity 2s ease;
        }

        .mist-container.active {
            opacity: 1;
        }

        .mist-layer {
            position: absolute;
            bottom: 0;
            left: -50%;
            width: 200%;
            height: 100%;
            background: linear-gradient(0deg, 
                rgba(255, 255, 255, 0.4) 0%,
                rgba(255, 255, 255, 0.2) 30%,
                rgba(255, 255, 255, 0.05) 60%,
                transparent 100%
            );
            animation: mistDrift 60s ease-in-out infinite;
        }

        .mist-layer:nth-child(2) {
            animation-delay: -25s;
            animation-duration: 45s;
            opacity: 0.7;
            height: 70%;
        }
        
        @keyframes mistDrift {
            0%, 100% { transform: translateX(-25%); }
            50% { transform: translateX(0%); }
        }
        
        /* ============== STARS ============== */
        .stars-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -5;
            pointer-events: none;
            opacity: 0;
            transition: opacity 2s ease;
        }
        
        .stars-container.active {
            opacity: 0.8;
        }
        
        .star {
            position: absolute;
            width: 2px;
            height: 2px;
            background: white;
            border-radius: 50%;
            animation: twinkle 4s ease-in-out infinite;
        }
        
        @keyframes twinkle {
            0%, 100% { opacity: 0.2; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.3); }
        }

        /* ============== SHOOTING STARS ============== */
        .shooting-star {
            position: fixed;
            width: 4px;
            height: 4px;
            background: white;
            border-radius: 50%;
            box-shadow: 
                0 0 6px 2px rgba(255, 255, 255, 0.9),
                0 0 12px 4px rgba(200, 220, 255, 0.6);
            opacity: 0;
            z-index: -4;
            pointer-events: none;
        }

        .shooting-star::after {
            content: '';
            position: absolute;
            top: 50%;
            right: 4px;
            width: 80px;
            height: 2px;
            background: linear-gradient(to left, rgba(255, 255, 255, 0.8), transparent);
            transform: translateY(-50%);
            border-radius: 2px;
        }

        .shooting-star.active {
            animation: shootingStar 1.2s ease-out forwards;
        }

        @keyframes shootingStar {
            0% {
                opacity: 1;
                transform: translate(0, 0) rotate(-35deg);
            }
            70% {
                opacity: 1;
            }
            100% {
                opacity: 0;
                transform: translate(-250px, 180px) rotate(-35deg);
            }
        }

        /* ============== TITLE SECTION ============== */
     
        .title-section {
            position: fixed;
            top: 6%;
            left: 0;
            width: 100%;
            z-index: 15;
            text-align: center;
            pointer-events: none;
        }

        .title-section h1 {
            font-size: 3.5rem;
            font-weight: 200;
            letter-spacing: 0.25em;
            margin-bottom: 8px;
            text-shadow: 0 0 60px rgba(255, 255, 255, 0.4);
        }

        .title-section .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
            letter-spacing: 0.08em;
            margin-bottom: 0;
        }
        /* ============== CONTENT ============== */
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 450px 20px 40px 20px;
            position: relative;
            z-index: 1;
        }
        
        .glass {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
        }
        
        h1 {
            text-align: center;
            font-size: 3.5rem;           /* slightly bigger */
            font-weight: 200;            /* lighter/more elegant */
            letter-spacing: 0.2em;       /* more spread out */
            margin-bottom: 8px;
            text-shadow: 0 0 60px rgba(255, 255, 255, 0.3);
        }
        
        .subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
            letter-spacing: 0.05em;
            margin-bottom: 40px;
        }
        
        .search-box {
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
        }
        
        .search-box input {
            flex: 1;
            padding: 18px 24px;
            font-size: 16px;
            border: none;
            border-radius: 16px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
            outline: none;
            transition: all 0.3s ease;
            font-weight: 300;
        }
        
        .search-box input::placeholder { color: var(--text-secondary); }
        
        .search-box input:focus {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        .search-box button {
            padding: 18px 32px;
            font-size: 16px;
            border: none;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-primary);
            cursor: pointer;
            font-weight: 400;
            letter-spacing: 0.05em;
            transition: all 0.3s ease;
        }
        
        .search-box button:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
        }
        
        .section-title {
            font-size: 0.9rem;
            font-weight: 400;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin: 40px 0 20px 0;
            color: var(--text-secondary);
        }
        
        .results-grid {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .track-card {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 12px;
        }
        
        .track-card:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        .track-card.selected { background: rgba(255, 255, 255, 0.15); }
        
        .track-card img {
            width: 60px;
            height: 60px;
            border-radius: 10px;
            object-fit: cover;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .track-info { flex: 1; }
        .track-name { font-weight: 500; font-size: 1.05rem; margin-bottom: 4px; }
        .track-artist { color: var(--text-secondary); font-size: 0.9rem; font-weight: 300; }
        .track-meta { font-size: 0.8rem; color: var(--text-secondary); }
        
        .seed-info {
            padding: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .seed-info img {
            width: 100px;
            height: 100px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }
        
        .seed-details h3 { font-size: 1.3rem; font-weight: 500; margin-bottom: 6px; }
        .seed-details p { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 12px; }
        
        .rec-themes { display: flex; gap: 8px; flex-wrap: wrap; }
        
        .theme-tag {
            padding: 5px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            font-size: 0.75rem;
        }
        
        .recommendation-card {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px;
            transition: all 0.3s ease;
            border-radius: 12px;
        }
        
        .recommendation-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(5px);
        }
        
        .recommendation-card img {
            width: 70px;
            height: 70px;
            border-radius: 10px;
            object-fit: cover;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .rec-info { flex: 1; }
        .rec-name { font-weight: 500; font-size: 1.05rem; margin-bottom: 4px; }
        .rec-name a { color: var(--text-primary); text-decoration: none; }
        .rec-name a:hover { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        .rec-artist { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 8px; }
        
        .rec-score { text-align: center; min-width: 70px; }
        .score-value { font-size: 1.8rem; font-weight: 300; text-shadow: 0 0 20px rgba(255, 255, 255, 0.3); }
        .score-label { font-size: 0.7rem; color: var(--text-secondary); letter-spacing: 0.1em; text-transform: uppercase; }
        
        .loading { text-align: center; padding: 50px; color: var(--text-secondary); }
        
        .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-top-color: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .error {
            background: rgba(255, 100, 100, 0.1);
            border: 1px solid rgba(255, 100, 100, 0.2);
            padding: 20px;
            border-radius: 16px;
            color: rgba(255, 150, 150, 0.9);
        }
        
        .login-prompt { text-align: center; padding: 50px; }
        
        .login-prompt a {
            display: inline-block;
            padding: 16px 32px;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-primary);
            text-decoration: none;
            border-radius: 30px;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        
        .login-prompt a:hover { background: rgba(255, 255, 255, 0.25); }
        
        #search-results, #recommendations { display: none; }
    </style>
</head>
<body>
    <!-- Sky gradient -->
    <div class="sky"></div>
    
    <!-- Clouds -->
    <div class="clouds-container">
        <div class="cloud cloud-1"></div>
        <div class="cloud cloud-2"></div>
        <div class="cloud cloud-3"></div>
        <div class="cloud cloud-4"></div>
        <div class="cloud cloud-5"></div>
        <div class="cloud cloud-6"></div>
        <div class="cloud cloud-7"></div>

    </div>
    
    <!-- Stars -->
    <div class="stars-container" id="stars-container"></div>
    
    <!-- Rain -->
    <div class="rain-container" id="rain-container"></div>
    
    <!-- Sun -->
    <div class="sun-container" id="sun-container"></div>
    
    <!-- Mist -->
    <div class="mist-container" id="mist-container">
        <div class="mist-layer"></div>
        <div class="mist-layer"></div>
    </div>

    <!-- Shooting Stars -->
    <div id="shooting-stars-container"></div>

    <!-- Title (centered on clouds) -->
    <div class="title-section">
        <h1>Soundscape</h1>
        <p class="subtitle">escape into music</p>
    </div>
    
    <!-- Content -->
    <div class="container">
        
        
        <div class="search-box">
            <input type="text" id="search-input" placeholder="Search for a song..." autocomplete="off">
            <button onclick="searchSongs()">Search</button>
        </div>
        
        <div id="login-prompt" class="login-prompt glass" style="display: none;">
            <p>Connect with Spotify to discover your vibe</p>
            <a href="/auth/login">Connect Spotify</a>
        </div>
        
        <div id="search-results">
            <h2 class="section-title">Select a song</h2>
            <div id="search-results-list" class="results-grid glass"></div>
        </div>
        
        <div id="recommendations">
            <h2 class="section-title">Seed Track</h2>
            <div id="seed-info" class="glass"></div>
            
            <h2 class="section-title">Recommended Tracks</h2>
            <div id="recommendations-list" class="results-grid glass"></div>
        </div>
    </div>
    
    <script>
        const colorThief = new ColorThief();
        const searchInput = document.getElementById('search-input');
        const searchResultsDiv = document.getElementById('search-results');
        const searchResultsList = document.getElementById('search-results-list');
        const recommendationsDiv = document.getElementById('recommendations');
        const recommendationsList = document.getElementById('recommendations-list');
        const seedInfoDiv = document.getElementById('seed-info');
        const loginPrompt = document.getElementById('login-prompt');
        
        const rainContainer = document.getElementById('rain-container');
        const sunContainer = document.getElementById('sun-container');
        const mistContainer = document.getElementById('mist-container');
        const starsContainer = document.getElementById('stars-container');

        function testWeather(type) {
        // Clear all first
        rainContainer.classList.remove('active');
        sunContainer.classList.remove('active');
        mistContainer.classList.remove('active');
        starsContainer.classList.remove('active');
        
        if (type === 'rain') {
            document.documentElement.style.setProperty('--rain-intensity', 0.7);
            generateRain(0.7);
            rainContainer.classList.add('active');
        }
        if (type === 'stars') {
            starsContainer.classList.add('active');
        }
        if (type === 'sun') {
            document.documentElement.style.setProperty('--sun-intensity', 0.8);
            sunContainer.classList.add('active');
        }
        if (type === 'mist') {
            document.documentElement.style.setProperty('--mist-intensity', 0.6);
            mistContainer.classList.add('active');
        }
    }
        
        // Generate rain
        function generateRain(intensity) {
            rainContainer.innerHTML = '';
            const dropCount = Math.floor(intensity * 120);
            
            for (let i = 0; i < dropCount; i++) {
                const drop = document.createElement('div');
                drop.className = 'rain-drop';
                drop.style.left = Math.random() * 100 + '%';
                
                // Random duration between 0.8s and 1.4s
                const duration = 0.8 + Math.random() * 0.6;
                drop.style.animationDuration = duration + 's';
                
                // KEY FIX: Random NEGATIVE delay so drops start at different Y positions
                // This spreads them out vertically instead of all starting at the same line
                drop.style.animationDelay = -(Math.random() * duration) + 's';
                
                drop.style.opacity = 0.3 + Math.random() * 0.4;
                rainContainer.appendChild(drop);
            }
        }
        
        // Generate stars
        function generateStars() {
            starsContainer.innerHTML = '';
            for (let i = 0; i < 80; i++) {
                const star = document.createElement('div');
                star.className = 'star';
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.animationDelay = Math.random() * 4 + 's';
                star.style.width = (1 + Math.random() * 2) + 'px';
                star.style.height = star.style.width;
                starsContainer.appendChild(star);
            }
        }
        generateStars();

        // Shooting stars
        function createShootingStar() {
            const container = document.getElementById('shooting-stars-container');
            const star = document.createElement('div');
            star.className = 'shooting-star';
            
            // Random position in upper-right area (above clouds)
            const startX = 50 + Math.random() * 45; // 50-95% from left
            const startY = 5 + Math.random() * 25;  // 5-30% from top
            
            star.style.left = startX + '%';
            star.style.top = startY + '%';
            
            container.appendChild(star);
            
            // Trigger animation
            requestAnimationFrame(() => {
                star.classList.add('active');
            });
            
            // Remove after animation
            setTimeout(() => {
                star.remove();
            }, 1500);
        }

        // Random shooting stars every 15-25 seconds
        function scheduleRandomShootingStar() {
            const delay = 15000 + Math.random() * 10000; // 15-25 seconds
            setTimeout(() => {
                createShootingStar();
                scheduleRandomShootingStar(); // Schedule next one
            }, delay);
        }

        // Start the random shooting stars
        scheduleRandomShootingStar();
        
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchSongs();
        });
        
        function updateColorsFromImage(imgElement) {
            try {
                if (imgElement.complete) extractAndApply(imgElement);
                else imgElement.addEventListener('load', () => extractAndApply(imgElement));
            } catch (e) { console.log('Color extraction error:', e); }
        }
        
        function extractAndApply(img) {
            try {
                const palette = colorThief.getPalette(img, 5);
                if (palette && palette.length >= 5) {
                    
                    function getBrightness(r, g, b) {
                        return (r * 299 + g * 587 + b * 114) / 1000;
                    }
                    
                    // Calculate target colors
                    const targetColors = palette.map((c, i) => {
                        let [r, g, b] = c;
                        const brightness = getBrightness(r, g, b);
                        
                        let darken;
                        if (i === 0) {
                            darken = brightness > 150 ? 0.25 : 0.4;
                        } else if (i === 1) {
                            darken = brightness > 150 ? 0.35 : 0.55;
                        } else {
                            darken = brightness > 150 ? 0.45 : 0.7;
                        }
                        
                        return {
                            r: Math.floor(r * darken),
                            g: Math.floor(g * darken),
                            b: Math.floor(b * darken)
                        };
                    });
                    
                    // Get current colors
                    const getCurrentColor = (varName) => {
                        const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
                        const match = value.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
                        if (match) {
                            return { r: parseInt(match[1]), g: parseInt(match[2]), b: parseInt(match[3]) };
                        }
                        return { r: 26, g: 26, b: 46 }; // default fallback
                    };
                    
                    const currentColors = [
                        getCurrentColor('--color-1'),
                        getCurrentColor('--color-2'),
                        getCurrentColor('--color-3'),
                        getCurrentColor('--color-4'),
                        getCurrentColor('--color-5')
                    ];
                    
                    // Animate over 3 seconds
                    const duration = 3000;
                    const startTime = performance.now();
                    
                    function animateColors(currentTime) {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Ease-in-out curve
                        const eased = progress < 0.5 
                            ? 2 * progress * progress 
                            : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                        
                        for (let i = 0; i < 5; i++) {
                            const r = Math.round(currentColors[i].r + (targetColors[i].r - currentColors[i].r) * eased);
                            const g = Math.round(currentColors[i].g + (targetColors[i].g - currentColors[i].g) * eased);
                            const b = Math.round(currentColors[i].b + (targetColors[i].b - currentColors[i].b) * eased);
                            document.documentElement.style.setProperty(`--color-${i + 1}`, `rgb(${r}, ${g}, ${b})`);
                        }
                        
                        if (progress < 1) {
                            requestAnimationFrame(animateColors);
                        }
                    }
                    
                    requestAnimationFrame(animateColors);
                    
                    // Tint clouds (this can stay instant, it's subtle)
                    const [r, g, b] = palette[2];
                    document.documentElement.style.setProperty('--cloud-tint',
                        `rgba(${Math.min(r + 40, 255)}, ${Math.min(g + 40, 255)}, ${Math.min(b + 40, 255)}, 0.9)`);
                }
            } catch (e) { console.log('Color apply error:', e); }
        }
        
        function updateWeather(themes) {
            rainContainer.classList.remove('active');
            sunContainer.classList.remove('active');
            mistContainer.classList.remove('active');
            starsContainer.classList.remove('active');
            
            if (!themes || themes.length === 0) return;
            
            const sadMoods = ['melancholy', 'heartbreak'];
            const happyMoods = ['party_fun', 'empowerment', 'hope_inspiration'];
            const calmMoods = ['nostalgia', 'love_romantic', 'sensual'];
            const deepMoods = ['existential'];
            
            let sadCount = 0, happyCount = 0, calmCount = 0, deepCount = 0;
            
            themes.forEach(theme => {
                if (sadMoods.includes(theme)) sadCount++;
                if (happyMoods.includes(theme)) happyCount++;
                if (calmMoods.includes(theme)) calmCount++;
                if (deepMoods.includes(theme)) deepCount++;
            });
            
            // Rain for sad moods
            if (sadCount > 0) {
                const intensity = Math.min(sadCount * 0.4, 0.8);
                document.documentElement.style.setProperty('--rain-intensity', intensity);
                generateRain(intensity);
                rainContainer.classList.add('active');
            }
            
            // Sun for happy moods (can combine with rain = bittersweet)
            if (happyCount > 0) {
                const intensity = Math.min(happyCount * 0.4, 0.8);
                document.documentElement.style.setProperty('--sun-intensity', intensity);
                sunContainer.classList.add('active');
            }
            
            // Mist for calm/romantic moods (can combine with others now!)
            if (calmCount > 0) {
                // Reduce intensity if other effects are active
                const baseIntensity = Math.min(calmCount * 0.3, 0.6);
                const intensity = (sadCount > 0 || happyCount > 0) ? baseIntensity * 0.6 : baseIntensity;
                document.documentElement.style.setProperty('--mist-intensity', intensity);
                mistContainer.classList.add('active');
            }
            
            // Stars for existential/deep moods (can combine with anything)
            if (deepCount > 0) {
                starsContainer.classList.add('active');
            }
            
            // Bonus: Add stars for very calm romantic nights too
            if (calmCount >= 2 && happyCount === 0) {
                starsContainer.classList.add('active');
            }
        }
                
        async function searchSongs() {
            const query = searchInput.value.trim();
            if (!query) return;
            
            searchResultsList.innerHTML = '<div class="loading"><div class="spinner"></div><p>Searching...</p></div>';
            searchResultsDiv.style.display = 'block';
            recommendationsDiv.style.display = 'none';
            loginPrompt.style.display = 'none';
            
            try {
                const response = await fetch(`/spotify/search?q=${encodeURIComponent(query)}&limit=6`);
                const data = await response.json();
                
                if (data.error) {
                    if (data.error === 'Not logged in. Go to /auth/login first.') {
                        searchResultsDiv.style.display = 'none';
                        loginPrompt.style.display = 'block';
                        return;
                    }
                    searchResultsList.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                if (!data.results || data.results.length === 0) {
                    searchResultsList.innerHTML = '<div class="error">No songs found.</div>';
                    return;
                }
                
                searchResultsList.innerHTML = data.results.map(track => `
                    <div class="track-card" onclick="getRecommendations('${track.id}', this)">
                        <img src="${track.image || 'https://via.placeholder.com/60'}" alt="" crossorigin="anonymous">
                        <div class="track-info">
                            <div class="track-name">${escapeHtml(track.name)}</div>
                            <div class="track-artist">${escapeHtml(track.artists.join(', '))}</div>
                        </div>
                        <div class="track-meta">${track.album || ''}</div>
                    </div>
                `).join('');
            } catch (err) {
                searchResultsList.innerHTML = `<div class="error">Error: ${err.message}</div>`;
            }
        }
        
        async function getRecommendations(trackId, element) {
            document.querySelectorAll('.track-card').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            
            const albumImg = element.querySelector('img');
            updateColorsFromImage(albumImg);
            
            recommendationsList.innerHTML = '<div class="loading"><div class="spinner"></div><p>Finding similar vibes...</p></div>';
            recommendationsDiv.style.display = 'block';
            
            try {
                const response = await fetch(`/spotify/vibe_v5?track_id=${trackId}&limit=10`);
                const data = await response.json();
                
                if (data.error) {
                    recommendationsList.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                const seed = data.seed;
                // Celebration shooting star!
                createShootingStar();   
                updateWeather(seed.detected_themes || []);
                
                seedInfoDiv.innerHTML = `
                <div class="seed-info">
                    <img src="${albumImg.src}" alt="" crossorigin="anonymous">
                    <div class="seed-details">
                    <h3>${escapeHtml(seed.name)}</h3>
                    <p>${escapeHtml(seed.artists.join(', '))} · ${seed.year || ''}</p>
                    </div>
                </div>
                `;

                
                if (!data.results || data.results.length === 0) {
                    recommendationsList.innerHTML = '<div class="error">No recommendations found.</div>';
                    return;
                }
                
                recommendationsList.innerHTML = data.results.map(track => `
                    <div class="recommendation-card">
                        <img src="${track.image || 'https://via.placeholder.com/70'}" alt="" crossorigin="anonymous">
                        <div class="rec-info">
                        <div class="rec-name">
                            <a href="${track.url}" target="_blank">${escapeHtml(track.name)}</a>
                        </div>
                        <div class="rec-artist">${escapeHtml(track.artists.join(', '))}</div>
                        </div>
                    </div>
                    `).join('');
                } catch (err) {
                    recommendationsList.innerHTML = `<div class="error">Error: ${err.message}</div>`;
                }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""







@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommend")
async def recommend(audio: UploadFile = File(...)) -> Dict:
    index = load_index()
    if index is None:
        return {"error": "index.json not found. Run POST /index first."}

    data = await audio.read()
    query_features = extract_features_from_bytes(data)

    scored = []
    for item in index:
        score = cosine_similarity(query_features, item["features"])
        scored.append({
            "filename": item["filename"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "VERSION": "REAL_SIMILARITY_V1",
        "query_filename": audio.filename,
        "results": scored[:10]
    }


@app.post("/index")
def index_music_folder():
    music_dir = Path("music")

    if not music_dir.exists():
        return {"error": "music/ folder not found"}

    index = []

    for file_path in music_dir.glob("*"):
        if file_path.suffix.lower() not in [".mp3", ".wav"]:
            continue

        print(f"Indexing {file_path.name}")

        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        features = extract_features_from_bytes(audio_bytes)

        index.append({
            "filename": file_path.name,
            "features": features
        })

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)

    return {"indexed_files": len(index)}

@app.get("/auth/login")
def auth_login(request: Request):
    verifier = make_code_verifier()
    challenge = make_code_challenge(verifier)
    state = secrets.token_urlsafe(16)

    request.session["pkce_verifier"] = verifier
    request.session["oauth_state"] = state

    from urllib.parse import urlencode

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": state,
    }

    authorize_url = "https://accounts.spotify.com/authorize"
    return RedirectResponse(f"{authorize_url}?{urlencode(params)}")


@app.get("/auth/callback")
def auth_callback(request: Request, code: str = "", state: str = ""):
    saved_state = request.session.get("oauth_state")
    verifier = request.session.get("pkce_verifier")

    if not code or state != saved_state:
        return {"error": "Invalid state"}

    token_url = "https://accounts.spotify.com/api/token"

    data = {
        "client_id": SPOTIFY_CLIENT_ID,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "code_verifier": verifier,
    }

    resp = requests.post(token_url, data=data)

    if resp.status_code != 200:
        return {"error": "Token exchange failed", "details": resp.text}

    token = resp.json()
    request.session["spotify_access_token"] = token["access_token"]

    return RedirectResponse("/")

@app.get("/spotify/search")
def spotify_search(request: Request, q: str, limit: int = 10, prefer_artist: str = ""):

    data, err = spotify_get(
        request,
        "https://api.spotify.com/v1/search",
        params={"q": q, "type": "track", "limit": 30, "market": "US"},
    )
    if err:
        return err

    items = data.get("tracks", {}).get("items", []) or []

    cleaned = []
    for t in items:
        candidate = {
            "id": t.get("id"),
            "name": t.get("name", ""),
            "artists": [a["name"] for a in (t.get("artists") or [])],
            "artist_ids": [a["id"] for a in (t.get("artists") or []) if a.get("id")],
            "album": (t.get("album") or {}).get("name", ""),
            "url": (t.get("external_urls") or {}).get("spotify"),
            "image": ((t.get("album") or {}).get("images") or [{}])[0].get("url"),
            "popularity": t.get("popularity", 0) or 0,
        }

        # filter junk/covers
        if looks_like_cover(candidate):
            continue
        if looks_like_seasonal(candidate):
            continue

        cleaned.append(candidate)

    # ranking: prefer exact-ish name match + higher popularity
    qnorm = norm(q)
    def score(c: dict) -> float:
        s = 0.0
        if norm(c["name"]) == qnorm:
            s += 3.0
        elif qnorm and qnorm in norm(c["name"]):
            s += 1.5
        s += (c.get("popularity", 0) / 100.0)  # small boost
        if prefer_artist and any(prefer_artist.lower() == a.lower() for a in c["artists"]):
            s += 2.5

        return s

    cleaned.sort(key=score, reverse=True)

    return {"query": q, "count": min(limit, len(cleaned)), "results": cleaned[:limit]}




# ============================================================
# GENIUS API HELPERS (add these before the endpoint)
# ============================================================

def genius_search(song_name: str, artist_name: str) -> dict | None:
    """
    Search Genius for a song. Returns song info with URL.
    """
    if not GENIUS_ACCESS_TOKEN:
        return None
    
    query = f"{song_name} {artist_name}"
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}"}
    params = {"q": query}
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            print(f"Genius search failed: {r.status_code}")
            return None
        
        data = r.json()
        hits = data.get("response", {}).get("hits", [])
        
        if not hits:
            return None
        
        # Return first hit
        return hits[0].get("result")
    except Exception as e:
        print(f"Genius search error: {e}")
        return None


def fetch_lyrics_from_genius(genius_url: str) -> str | None:
    """
    Scrape lyrics from a Genius song page.
    """
    try:
        r = requests.get(genius_url, timeout=10)
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.text, "lxml")
        
        # Genius stores lyrics in divs with data-lyrics-container="true"
        lyrics_containers = soup.find_all("div", {"data-lyrics-container": "true"})
        
        if not lyrics_containers:
            # Fallback: try older format
            lyrics_div = soup.find("div", class_="lyrics")
            if lyrics_div:
                return lyrics_div.get_text(separator="\n").strip()
            return None
        
        lyrics_parts = []
        for container in lyrics_containers:
            # Get text, replacing <br> with newlines
            text = container.get_text(separator="\n")
            lyrics_parts.append(text)
        
        return "\n".join(lyrics_parts).strip()
    except Exception as e:
        print(f"Lyrics fetch error: {e}")
        return None


def get_lyrics_for_track(song_name: str, artist_name: str) -> str | None:
    """
    Get lyrics for a track via Genius API + scraping.
    """
    # Check cache first
    cached = get_cached_lyrics(song_name, artist_name)
    if cached:
        return cached
    
    # Search for the song
    song_info = genius_search(song_name, artist_name)
    if not song_info:
        return None
    
    # Get the Genius URL
    genius_url = song_info.get("url")
    if not genius_url:
        return None
    
    # Fetch lyrics
    lyrics = fetch_lyrics_from_genius(genius_url)
    
    # Cache if successful
    if lyrics:
        cache_lyrics(song_name, artist_name, lyrics)
    
    return lyrics
    
    


# ============================================================
# ENDPOINT
# ============================================================

@app.get("/spotify/vibe_v5")
def spotify_vibe_v5(request: Request, track_id: str, limit: int = 20):
    """
    Playlist co-occurrence + Lyrics analysis + Last.fm recommendations.
    
    Combines:
    - Playlist co-occurrence (which songs humans group together)
    - Lyrics similarity (themes, sentiment, mood)
    - Last.fm tags (user-generated genre/mood tags)
    - Last.fm similar artists (community-driven similarity)
    - Era, genre, popularity signals
    """
    
    # =========================================
    # 1) FETCH SEED TRACK METADATA
    # =========================================
    
    seed, err = spotify_get_json(request, f"/tracks/{track_id}", params={"market": "US"})
    if err:
        return {"error": "Could not fetch seed track", "details": err}
    
    seed_name = seed.get("name", "")
    seed_artists = seed.get("artists", []) or []
    seed_artist_ids = set(a.get("id") for a in seed_artists if a.get("id"))
    seed_artist_name = seed_artists[0].get("name", "") if seed_artists else ""
    seed_album = seed.get("album", {}) or {}
    seed_year = parse_year(seed_album.get("release_date"))
    
    # Fetch genres
    seed_genres = []
    if seed_artist_ids:
        artist_ids_str = ",".join(list(seed_artist_ids)[:5])
        artists_data, err = spotify_get_json(request, "/artists", params={"ids": artist_ids_str})
        if not err:
            for a in artists_data.get("artists", []) or []:
                seed_genres.extend(a.get("genres", []))
    seed_genres = list(dict.fromkeys(seed_genres))[:10]
    
    # =========================================
    # 2) FETCH SEED LYRICS & ANALYZE
    # =========================================
    
    seed_lyrics = get_lyrics_for_track(seed_name, seed_artist_name)
    seed_lyrics_analysis = analyze_lyrics(seed_lyrics) if seed_lyrics else {}
    
    # =========================================
    # 2.5) FETCH SEED LAST.FM DATA
    # =========================================
    
    seed_lastfm_tags = lastfm_get_artist_tags(seed_artist_name)
    seed_similar_artists = lastfm_get_similar_artists(seed_artist_name)
    
    # =========================================
    # 2.75) ANALYZE SEED ALBUM ART
    # =========================================
    
    seed_album_image = (seed_album.get("images") or [{}])[0].get("url")
    seed_album_art_analysis = analyze_album_art(seed_album_image) if seed_album_image else None
    
    # =========================================
    # 3) FIND PLAYLISTS & TRACK CO-OCCURRENCE
    # =========================================
    
    candidate_playlist_count: dict[str, int] = defaultdict(int)
    candidate_data: dict[str, dict] = {}
    playlists_searched = 0
    playlists_with_seed = 0
    
    queries = build_playlist_queries_v5(seed_name, seed_artist_name)
    
    for query in queries[:4]:
        pdata, err = spotify_get_json(
            request,
            "/search",
            params={"q": query, "type": "playlist", "limit": 3, "market": "US"}
        )
        if err:
            continue
        
        playlists = pdata.get("playlists", {}).get("items", []) or []
        
        for playlist in playlists:
            if not playlist:
                continue
            
            pid = playlist.get("id")
            if not pid:
                continue
            
            playlists_searched += 1
            
            tracks_data, err = spotify_get_json(
                request,
                f"/playlists/{pid}/tracks",
                params={"limit": 50, "market": "US"}
            )
            if err:
                continue
            
            items = tracks_data.get("items", []) or []
            
            playlist_track_ids = set()
            for item in items:
                t = (item or {}).get("track")
                if t and t.get("id"):
                    playlist_track_ids.add(t["id"])
            
            seed_in_playlist = track_id in playlist_track_ids
            if seed_in_playlist:
                playlists_with_seed += 1
            
            for item in items:
                t = (item or {}).get("track")
                if not t or not t.get("id"):
                    continue
                
                tid = t["id"]
                if tid == track_id:
                    continue
                
                if seed_in_playlist:
                    candidate_playlist_count[tid] += 2
                else:
                    candidate_playlist_count[tid] += 1
                
                if tid not in candidate_data:
                    album = t.get("album", {}) or {}
                    candidate_data[tid] = {
                        "id": tid,
                        "name": t.get("name", ""),
                        "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                        "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                        "album": album.get("name", ""),
                        "release_date": album.get("release_date"),
                        "popularity": t.get("popularity", 0),
                        "image": (album.get("images") or [{}])[0].get("url"),
                        "url": (t.get("external_urls", {}) or {}).get("spotify"),
                        "source": "playlist",
                    }
    
    # =========================================
    # 3.5) SEARCH FOR TRACKS BY LAST.FM SIMILAR ARTISTS
    # =========================================
    
    lastfm_candidates_added = 0
    if seed_similar_artists:
        # Search for top tracks by similar artists (limit to top 5 artists, 3 tracks each)
        for similar_artist in seed_similar_artists[:5]:
            # Search Spotify for this artist's tracks
            search_data, err = spotify_get_json(
                request,
                "/search",
                params={
                    "q": f"artist:{similar_artist}",
                    "type": "track",
                    "limit": 5,
                    "market": "US"
                }
            )
            if err:
                continue
            
            tracks = search_data.get("tracks", {}).get("items", []) or []
            
            for t in tracks[:3]:  # top 3 per artist
                if not t or not t.get("id"):
                    continue
                
                tid = t["id"]
                if tid == track_id:
                    continue
                
                # Skip if already found via playlist
                if tid in candidate_data:
                    # But boost its score since it's also a Last.fm similar artist
                    candidate_playlist_count[tid] += 2
                    continue
                
                album = t.get("album", {}) or {}
                candidate_data[tid] = {
                    "id": tid,
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "album": album.get("name", ""),
                    "release_date": album.get("release_date"),
                    "popularity": t.get("popularity", 0),
                    "image": (album.get("images") or [{}])[0].get("url"),
                    "url": (t.get("external_urls", {}) or {}).get("spotify"),
                    "source": "lastfm_similar",
                }
                # Give them a baseline playlist count so they're competitive
                candidate_playlist_count[tid] = 2
                lastfm_candidates_added += 1
    
    if not candidate_data:
        return {
            "seed": {"id": track_id, "name": seed_name, "artists": [a.get("name") for a in seed_artists]},
            "count": 0,
            "results": [],
            "debug": {"error": "No candidates found in playlists"}
        }
    
    # =========================================
    # 4) FETCH GENRES FOR CANDIDATES
    # =========================================
    
    all_artist_ids = set()
    for c in candidate_data.values():
        all_artist_ids.update(c.get("artist_ids", []))
    
    genres_by_artist = {}
    artist_id_list = list(all_artist_ids)
    for i in range(0, len(artist_id_list), 50):
        batch = artist_id_list[i:i+50]
        ids_str = ",".join(batch)
        artists_data, err = spotify_get_json(request, "/artists", params={"ids": ids_str})
        if not err:
            for a in (artists_data.get("artists", []) or []):
                if a and a.get("id"):
                    genres_by_artist[a["id"]] = a.get("genres", [])
    
    # =========================================
    # 5) FETCH LYRICS + LAST.FM FOR TOP CANDIDATES & SCORE
    # =========================================
    
    # Separate Last.fm candidates from playlist candidates
    lastfm_candidates = [(tid, c) for tid, c in candidate_data.items() if c.get("source") == "lastfm_similar"]
    playlist_candidates = [(tid, c) for tid, c in candidate_data.items() if c.get("source") != "lastfm_similar"]
    
    # Sort each group by playlist count
    lastfm_candidates.sort(key=lambda x: candidate_playlist_count.get(x[0], 0), reverse=True)
    playlist_candidates.sort(key=lambda x: candidate_playlist_count.get(x[0], 0), reverse=True)
    
    # Take top from each: prioritize Last.fm candidates, fill rest with playlist candidates
    candidates_to_score = lastfm_candidates[:15] + playlist_candidates[:20]
    
    max_cooccurrence = max(candidate_playlist_count.values()) if candidate_playlist_count else 1
    scored_results = []
    lyrics_fetched = 0
    lastfm_tags_fetched = 0
    max_lyrics_fetch = 8  # limit API calls
    max_lastfm_fetch = 12  # slightly higher since it's faster
    
    # Cache for lyrics analysis
    lyrics_cache: dict[str, dict] = {}
    # Cache for Last.fm tags (by artist)
    candidate_tags_cache: dict[str, list[str]] = {}
    
    for tid, c in candidates_to_score:  # score mixed candidates
        # Build candidate genres
        candidate_genres = []
        for aid in c.get("artist_ids", []):
            candidate_genres.extend(genres_by_artist.get(aid, []))
        candidate_genres = list(dict.fromkeys(candidate_genres))
        
        # Filter junk
        filter_obj = {
            "name": c.get("name", ""),
            "album": c.get("album", ""),
            "artists": c.get("artists", []),
        }
        if looks_like_cover(filter_obj):
            continue
        if looks_like_seasonal(filter_obj):
            continue
        
        cand_name = c.get("name", "")
        cand_artist = c.get("artists", [""])[0] if c.get("artists") else ""
        
        # Fetch lyrics for top candidates (if we have seed lyrics)
        candidate_lyrics_analysis = {}
        if seed_lyrics_analysis and lyrics_fetched < max_lyrics_fetch:
            cache_key = f"{cand_name}|{cand_artist}".lower()
            
            if cache_key in lyrics_cache:
                candidate_lyrics_analysis = lyrics_cache[cache_key]
            else:
                cand_lyrics = get_lyrics_for_track(cand_name, cand_artist)
                if cand_lyrics:
                    candidate_lyrics_analysis = analyze_lyrics(cand_lyrics)
                    lyrics_cache[cache_key] = candidate_lyrics_analysis
                    lyrics_fetched += 1
        
        # Fetch Last.fm tags for candidate artist
        candidate_lastfm_tags = []
        if cand_artist:
            artist_lower = cand_artist.lower()
            if artist_lower in candidate_tags_cache:
                candidate_lastfm_tags = candidate_tags_cache[artist_lower]
            elif lastfm_tags_fetched < max_lastfm_fetch:
                candidate_lastfm_tags = lastfm_get_artist_tags(cand_artist)
                candidate_tags_cache[artist_lower] = candidate_lastfm_tags
                if candidate_lastfm_tags:  # only count if we got results
                    lastfm_tags_fetched += 1
        
        # Analyze candidate album art
        candidate_image = c.get("image")
        candidate_album_art_analysis = analyze_album_art(candidate_image) if candidate_image else None
        
        # Get co-occurrence count
        cooccur_count = candidate_playlist_count.get(tid, 0)
        
        # Score
        score_result = compute_composite_score_v5(
            seed_name=seed_name,
            seed_genres=seed_genres,
            seed_year=seed_year,
            seed_artist_ids=seed_artist_ids,
            seed_lyrics_analysis=seed_lyrics_analysis,
            seed_lastfm_tags=seed_lastfm_tags,
            seed_similar_artists=seed_similar_artists,
            seed_album_art_analysis=seed_album_art_analysis,
            candidate=c,
            candidate_genres=candidate_genres,
            candidate_lyrics_analysis=candidate_lyrics_analysis,
            candidate_lastfm_tags=candidate_lastfm_tags,
            candidate_album_art_analysis=candidate_album_art_analysis,
            cooccurrence_count=cooccur_count,
            max_cooccurrence=max_cooccurrence,
        )
        
        c["score"] = score_result["final_score"]
        c["score_components"] = score_result["components"]
        
        # Add lyrics info (for debugging, not the actual lyrics)
        if candidate_lyrics_analysis:
            top_themes = sorted(
                candidate_lyrics_analysis.get("themes", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            c["detected_themes"] = [t[0] for t in top_themes if t[1] > 0.3]
        
        scored_results.append(c)
    
    # Sort by score
    scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # =========================================
    # 6) DIVERSITY CAPS + OUTPUT
    # =========================================
    
    final_results = apply_diversity_caps(scored_results, max_per_artist=2, max_from_seed_album=2)
    final_results = final_results[:limit]
    
    # Clean up
    for r in final_results:
        r.pop("artist_ids", None)
    
    # Seed themes for debug
    seed_top_themes = []
    if seed_lyrics_analysis:
        top = sorted(
            seed_lyrics_analysis.get("themes", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        seed_top_themes = [t[0] for t in top if t[1] > 0.3]
    
    return {
        "seed": {
            "id": track_id,
            "name": seed_name,
            "artists": [a.get("name", "") for a in seed_artists],
            "album": seed_album.get("name"),
            "year": seed_year,
            "genres": seed_genres[:5],
            "url": (seed.get("external_urls", {}) or {}).get("spotify"),
            "detected_themes": seed_top_themes,
            "lyrics_found": bool(seed_lyrics),
            "lastfm_tags": seed_lastfm_tags[:5],
            "lastfm_similar_count": len(seed_similar_artists),
            "album_art_analysis": seed_album_art_analysis,
        },
        "weights": WEIGHTS_V5,
        "count": len(final_results),
        "results": final_results,
        "debug": {
            "playlists_searched": playlists_searched,
            "playlists_with_seed": playlists_with_seed,
            "unique_candidates": len(candidate_data),
            "candidates_scored": len(scored_results),
            "lyrics_fetched": lyrics_fetched,
            "lastfm_tags_fetched": lastfm_tags_fetched,
            "lastfm_candidates_added": lastfm_candidates_added,
            "max_cooccurrence": max_cooccurrence,
        }
    }