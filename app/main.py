from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List, Dict
import uuid
import os
import json
import numpy as np
import librosa
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




from bs4 import BeautifulSoup
from .vibe_v5 import (
    compute_composite_score_v5,
    build_playlist_queries_v5,
    parse_year,
    analyze_lyrics,
    WEIGHTS_V5,
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

    # If token expired, you’ll see 401 — we’ll add refresh later
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

    # make sure we don’t double-append "playlist"
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
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #1DB954, #1ed760);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .search-box input {
            flex: 1;
            padding: 14px 18px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            outline: none;
        }
        
        .search-box input::placeholder {
            color: #888;
        }
        
        .search-box input:focus {
            background: rgba(255,255,255,0.15);
        }
        
        .search-box button {
            padding: 14px 28px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: #1DB954;
            color: #fff;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        
        .search-box button:hover {
            background: #1ed760;
        }
        
        .section-title {
            font-size: 1.2rem;
            margin: 30px 0 15px 0;
            color: #ccc;
        }
        
        .results-grid {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .track-card {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .track-card:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .track-card.selected {
            background: rgba(29, 185, 84, 0.2);
            border: 1px solid #1DB954;
        }
        
        .track-card img {
            width: 56px;
            height: 56px;
            border-radius: 4px;
            object-fit: cover;
        }
        
        .track-info {
            flex: 1;
        }
        
        .track-name {
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        .track-artist {
            color: #888;
            font-size: 14px;
        }
        
        .track-meta {
            text-align: right;
            font-size: 12px;
            color: #666;
        }
        
        .recommendation-card {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            transition: background 0.2s;
        }
        
        .recommendation-card:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .recommendation-card img {
            width: 64px;
            height: 64px;
            border-radius: 4px;
            object-fit: cover;
        }
        
        .rec-info {
            flex: 1;
        }
        
        .rec-name {
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        .rec-name a {
            color: #fff;
            text-decoration: none;
        }
        
        .rec-name a:hover {
            text-decoration: underline;
        }
        
        .rec-artist {
            color: #888;
            font-size: 14px;
            margin-bottom: 6px;
        }
        
        .rec-themes {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        
        .theme-tag {
            padding: 3px 8px;
            background: rgba(29, 185, 84, 0.2);
            color: #1DB954;
            border-radius: 12px;
            font-size: 11px;
        }
        
        .rec-score {
            text-align: right;
            min-width: 60px;
        }
        
        .score-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1DB954;
        }
        
        .score-label {
            font-size: 11px;
            color: #666;
        }
        
        .seed-info {
            background: rgba(29, 185, 84, 0.1);
            border: 1px solid rgba(29, 185, 84, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .seed-info img {
            width: 80px;
            height: 80px;
            border-radius: 4px;
        }
        
        .seed-details h3 {
            margin-bottom: 4px;
        }
        
        .seed-details p {
            color: #888;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        
        .spinner {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: #1DB954;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .error {
            background: rgba(255, 100, 100, 0.1);
            border: 1px solid rgba(255, 100, 100, 0.3);
            padding: 15px;
            border-radius: 8px;
            color: #ff6b6b;
        }
        
        .login-prompt {
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        
        .login-prompt a {
            display: inline-block;
            margin-top: 15px;
            padding: 12px 24px;
            background: #1DB954;
            color: #fff;
            text-decoration: none;
            border-radius: 24px;
            font-weight: 600;
        }
        
        .login-prompt a:hover {
            background: #1ed760;
        }
        
        #search-results, #recommendations {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vibe Finder</h1>
        <p class="subtitle">Find songs that match your vibe</p>
        
        <div class="search-box">
            <input type="text" id="search-input" placeholder="Search for a song..." autocomplete="off">
            <button onclick="searchSongs()">Search</button>
        </div>
        
        <div id="login-prompt" class="login-prompt" style="display: none;">
            <p>Please log in with Spotify to search for songs</p>
            <a href="/auth/login">Log in with Spotify</a>
        </div>
        
        <div id="search-results">
            <h2 class="section-title">Select a song</h2>
            <div id="search-results-list" class="results-grid"></div>
        </div>
        
        <div id="recommendations">
            <h2 class="section-title">Seed Track</h2>
            <div id="seed-info"></div>
            
            <h2 class="section-title">Recommended Tracks</h2>
            <div id="recommendations-list" class="results-grid"></div>
        </div>
    </div>
    
    <script>
        const searchInput = document.getElementById('search-input');
        const searchResultsDiv = document.getElementById('search-results');
        const searchResultsList = document.getElementById('search-results-list');
        const recommendationsDiv = document.getElementById('recommendations');
        const recommendationsList = document.getElementById('recommendations-list');
        const seedInfoDiv = document.getElementById('seed-info');
        const loginPrompt = document.getElementById('login-prompt');
        
        // Search on Enter key
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchSongs();
        });
        
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
                    searchResultsList.innerHTML = '<div class="error">No songs found. Try a different search.</div>';
                    return;
                }
                
                searchResultsList.innerHTML = data.results.map(track => `
                    <div class="track-card" onclick="getRecommendations('${track.id}', this)">
                        <img src="${track.image || 'https://via.placeholder.com/56'}" alt="">
                        <div class="track-info">
                            <div class="track-name">${escapeHtml(track.name)}</div>
                            <div class="track-artist">${escapeHtml(track.artists.join(', '))}</div>
                        </div>
                        <div class="track-meta">${track.album || ''}</div>
                    </div>
                `).join('');
                
            } catch (err) {
                searchResultsList.innerHTML = `<div class="error">Error searching: ${err.message}</div>`;
            }
        }
        
        async function getRecommendations(trackId, element) {
            // Highlight selected
            document.querySelectorAll('.track-card').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            
            recommendationsList.innerHTML = '<div class="loading"><div class="spinner"></div><p>Finding similar vibes...</p></div>';
            recommendationsDiv.style.display = 'block';
            
            try {
                const response = await fetch(`/spotify/vibe_v5?track_id=${trackId}&limit=10`);
                const data = await response.json();
                
                if (data.error) {
                    recommendationsList.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                // Show seed info
                const seed = data.seed;
                seedInfoDiv.innerHTML = `
                    <div class="seed-info">
                        <img src="${element.querySelector('img').src}" alt="">
                        <div class="seed-details">
                            <h3>${escapeHtml(seed.name)}</h3>
                            <p>${escapeHtml(seed.artists.join(', '))} · ${seed.year || ''}</p>
                            <div class="rec-themes">
                                ${(seed.detected_themes || []).map(t => `<span class="theme-tag">${t}</span>`).join('')}
                                ${seed.lyrics_found ? '<span class="theme-tag">lyrics ✓</span>' : ''}
                            </div>
                        </div>
                    </div>
                `;
                
                // Show recommendations
                if (!data.results || data.results.length === 0) {
                    recommendationsList.innerHTML = '<div class="error">No recommendations found.</div>';
                    return;
                }
                
                recommendationsList.innerHTML = data.results.map(track => `
                    <div class="recommendation-card">
                        <img src="${track.image || 'https://via.placeholder.com/64'}" alt="">
                        <div class="rec-info">
                            <div class="rec-name">
                                <a href="${track.url}" target="_blank">${escapeHtml(track.name)}</a>
                            </div>
                            <div class="rec-artist">${escapeHtml(track.artists.join(', '))}</div>
                            <div class="rec-themes">
                                ${(track.detected_themes || []).map(t => `<span class="theme-tag">${t}</span>`).join('')}
                            </div>
                        </div>
                        <div class="rec-score">
                            <div class="score-value">${Math.round(track.score * 100)}</div>
                            <div class="score-label">match</div>
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
    Playlist co-occurrence + Lyrics analysis recommendations.
    
    Combines:
    - Playlist co-occurrence (which songs humans group together)
    - Lyrics similarity (themes, sentiment, mood)
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
                    }
    
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
    # 5) FETCH LYRICS FOR TOP CANDIDATES & SCORE
    # =========================================
    
    # Sort candidates by playlist count to prioritize lyrics fetching
    sorted_candidates = sorted(
        candidate_data.items(),
        key=lambda x: candidate_playlist_count.get(x[0], 0),
        reverse=True
    )
    
    max_cooccurrence = max(candidate_playlist_count.values()) if candidate_playlist_count else 1
    scored_results = []
    lyrics_fetched = 0
    max_lyrics_fetch = 8  # limit API calls
    
    # Cache for lyrics analysis
    lyrics_cache: dict[str, dict] = {}
    
    for tid, c in sorted_candidates[:30]:  # limit to top 30 candidates for optimization
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
        
        # Fetch lyrics for top candidates (if we have seed lyrics)
        candidate_lyrics_analysis = {}
        if seed_lyrics_analysis and lyrics_fetched < max_lyrics_fetch:
            cand_name = c.get("name", "")
            cand_artist = c.get("artists", [""])[0] if c.get("artists") else ""
            
            cache_key = f"{cand_name}|{cand_artist}".lower()
            
            if cache_key in lyrics_cache:
                candidate_lyrics_analysis = lyrics_cache[cache_key]
            else:
                cand_lyrics = get_lyrics_for_track(cand_name, cand_artist)
                if cand_lyrics:
                    candidate_lyrics_analysis = analyze_lyrics(cand_lyrics)
                    lyrics_cache[cache_key] = candidate_lyrics_analysis
                    lyrics_fetched += 1
        
        # Get co-occurrence count
        cooccur_count = candidate_playlist_count.get(tid, 0)
        
        # Score
        score_result = compute_composite_score_v5(
            seed_name=seed_name,
            seed_genres=seed_genres,
            seed_year=seed_year,
            seed_artist_ids=seed_artist_ids,
            seed_lyrics_analysis=seed_lyrics_analysis,
            candidate=c,
            candidate_genres=candidate_genres,
            candidate_lyrics_analysis=candidate_lyrics_analysis,
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
            "max_cooccurrence": max_cooccurrence,
        }
    }




















