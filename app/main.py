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
from typing import List
import re
from .vibe_v2 import (
    compute_composite_score,
    build_genre_search_queries,
    parse_year,
    WEIGHTS,
)

from .vibe_v3 import (
    compute_composite_score_v3,
    build_search_queries_v3,
    parse_year,
    WEIGHTS_V3,
)

from .vibe_v4 import (
     compute_composite_score_v4,
     build_playlist_queries_v4,
     parse_year,
     WEIGHTS_V4,
)



load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SESSION_SECRET = os.getenv("SESSION_SECRET")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_REDIRECT_URI or not SESSION_SECRET:
    raise RuntimeError("Missing env vars.")



INDEX_PATH = "index.json"

###Helper Functions

def extract_features_from_bytes(audio_bytes: bytes):
    import io

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=30)

    # MFCC = timbre fingerprint (core of production feel)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # average across time to get fixed-size vector
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean.tolist()

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))




def load_index():
    if not Path(INDEX_PATH).exists():
        return None
    with open(INDEX_PATH, "r") as f:
        return json.load(f)

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

def cap_seed_artist(results: list[dict], seed_artist_names: list[str], max_seed_tracks: int = 2) -> list[dict]:
    seed_set = set((a or "").lower() for a in seed_artist_names)
    count = 0
    out = []
    for r in results:
        artists = [(x or "").lower() for x in (r.get("artists") or [])]
        is_seed = any(a in seed_set for a in artists)
        if is_seed:
            if count >= max_seed_tracks:
                continue
            count += 1
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

def find_best_local_match(index: list[dict], seed_name: str, seed_artists: list[str]) -> dict | None:
    """
    index entries look like: {"filename": "...", "features": [...]}
    We try to match by filename containing artist + title tokens.
    """
    target_title = norm(seed_name)
    target_artist = norm(seed_artists[0] if seed_artists else "")

    best = None
    best_score = 0

    for item in index:
        fname = norm(item.get("filename", ""))

        # simple token overlap score
        score = 0
        if target_title and target_title in fname:
            score += 3
        if target_artist and target_artist in fname:
            score += 2

        # partial overlap bonus
        title_tokens = set(target_title.split())
        fname_tokens = set(fname.split())
        score += len(title_tokens & fname_tokens) * 0.2

        if score > best_score:
            best_score = score
            best = item

    # require minimum confidence
    if best_score >= 2.5:
        return best
    return None


def parse_year(date_str: str | None) -> int | None:
    # Spotify release_date can be "YYYY", "YYYY-MM-DD", or "YYYY-MM"
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except Exception:
        return None

def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def get_related_artists(request: Request, artist_id: str) -> list[dict]:
    data, err = spotify_get_json(request, f"/artists/{artist_id}/related-artists")
    if err:
        return []
    return data.get("artists", []) or []

def add_artist_album_tracks(request: Request, artist_id: str, candidates: list, max_albums: int = 3, tracks_per_album: int = 10):
    # get artist albums (albums + singles)
    albums, err = spotify_get_json(
        request,
        f"/artists/{artist_id}/albums",
        params={"include_groups": "album,single", "limit": 20, "market": "US"},
    )
    if err:
        return

    items = albums.get("items", []) or []
    # de-dupe by album id
    seen_albums = set()
    picked = []
    for a in items:
        aid = a.get("id")
        if not aid or aid in seen_albums:
            continue
        seen_albums.add(aid)
        picked.append(aid)
        if len(picked) >= max_albums:
            break

    for alb_id in picked:
        tracks, err = spotify_get_json(
            request,
            f"/albums/{alb_id}/tracks",
            params={"limit": tracks_per_album, "market": "US"},
        )
        if err:
            continue
        for t in tracks.get("items", []):
            if not t or not t.get("id"):
                continue
            candidates.append({
                "id": t["id"],
                "name": t.get("name", ""),
                "artists": [ar.get("name","") for ar in t.get("artists", [])],
                "artist_ids": [ar.get("id") for ar in t.get("artists", []) if ar.get("id")],
                "source": "artist_catalog",
            })






    
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
    <html>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Vibe Finder (Prototype)</h1>

        <p>
          <a href="/auth/login" style="font-weight:bold;">
            Log in with Spotify
          </a>
        </p>

        <h3>1) Health Check</h3>
        <p>Open <a href="/health">/health</a></p>

        <h3>2) Upload audio (real recommend)</h3>
        <form action="/recommend" enctype="multipart/form-data" method="post">
          <input name="audio" type="file" accept="audio/*" />
          <button type="submit" style="margin-left: 8px;">Find similar</button>
        </form>

        <p style="margin-top: 20px; color: #666;">
          Now supports Spotify login.
        </p>
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


@app.get("/spotify/vibe")
def spotify_vibe(request: Request, track_id: str, limit: int = 20):
    """
    "Similar songs" without downloading/uploading music.
    Strategy:
      1) Fetch seed track + seed audio-features
      2) Build a candidate pool from:
         - related artists (top tracks via artist search/playlists you already do)
         - playlists found via search queries
         - seed album tracks (small contribution, but capped)
      3) Hydrate candidates (track metadata + audio-features)
      4) Score by audio-feature similarity (NOT same artist)
      5) Penalize same-artist, filter covers/seasonal, apply diversity caps
    """

    print("spotify_vibe called with track_id =", track_id)

    # -------------------------
    # 1) Seed track + audio features
    # -------------------------
    seed, err = spotify_get_json(request, f"/tracks/{track_id}", params={"market": "US"})
    if err:
        return err

    seed_artists = seed.get("artists", []) or []
    seed_artist_ids = [a.get("id") for a in seed_artists if a.get("id")]
    seed_artist_name = (seed_artists[0].get("name") if seed_artists else "") or ""
    seed_album_id = (seed.get("album", {}) or {}).get("id")
    seed_track_name = (seed.get("name") or "") or ""
    seed_artist_set = set(seed_artist_ids)

    seed_af, err = spotify_get_json(request, f"/audio-features/{track_id}")
    # audio-features sometimes fails for local/unavailable tracks; handle gracefully
    if err:
        seed_af = None

    # -------------------------
    # 2) Build candidates pool (no /recommendations)
    # -------------------------
    candidates: list[dict] = []

    # 2a) small amount from seed album (helps, but will be capped later)
    if seed_album_id:
        album_tracks, err = spotify_get_json(
            request,
            f"/albums/{seed_album_id}/tracks",
            params={"limit": 20, "market": "US"},
        )
        if not err:
            for t in (album_tracks.get("items", []) or []):
                tid = t.get("id")
                if not tid or tid == track_id:
                    continue
                candidates.append({
                    "id": tid,
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "source": "album",
                })

    # 2b) related artists (strong signal for "similar", but we will still re-rank by audio)
    related_artist_ids: list[str] = []
    if seed_artist_ids:
        rel = get_related_artists(request, seed_artist_ids[0])
        for a in (rel or [])[:12]:
            rid = a.get("id")
            if rid:
                related_artist_ids.append(rid)

    # Pull some tracks from playlists based on seed artist + seed track.
    # (This avoids "classic rock playlist" generic noise.)
    queries = []
    if seed_artist_name:
        queries += [f"{seed_artist_name} similar", f"{seed_artist_name} radio"]
    if seed_track_name:
        queries += [f"{seed_track_name} similar", f"songs like {seed_track_name}"]

    for q in queries:
        add_playlist_tracks_as_candidates(request, q, candidates, max_playlists=1, per_playlist=10)

    # Optionally: also sample playlists based on related artist names (adds breadth)
    # We do this lightly to avoid random drift.
    if related_artist_ids:
        # fetch a few related artist objects to get their names for queries
        ids = ",".join(related_artist_ids[:5])
        related_data, err = spotify_get_json(request, "/artists", params={"ids": ids})
        if not err:
            for a in (related_data.get("artists", []) or [])[:5]:
                nm = (a.get("name") or "").strip()
                if nm:
                    add_playlist_tracks_as_candidates(request, f"{nm} radio", candidates, max_playlists=1, per_playlist=6)

    # remove seed + dedupe
    candidates = [c for c in candidates if c.get("id") and c["id"] != track_id]
    candidates = dedupe_tracks(candidates)

    if not candidates:
        return {
            "seed": {
                "id": track_id,
                "name": seed_track_name,
                "artists": [a.get("name", "") for a in seed_artists],
                "url": (seed.get("external_urls", {}) or {}).get("spotify"),
            },
            "count": 0,
            "results": [],
            "debug": {"note": "No candidates found from playlists/album/related sources."}
        }

    # -------------------------
    # 3) Hydrate track metadata (album/image/url/popularity/release_date)
    # -------------------------
    hydrate_ids = ",".join([c["id"] for c in candidates[:50]])
    by_id = {}

    if hydrate_ids:
        full, err = spotify_get_json(request, "/tracks", params={"ids": hydrate_ids, "market": "US"})
        if not err:
            by_id = {t["id"]: t for t in (full.get("tracks", []) or []) if t and t.get("id")}

    hydrated: list[dict] = []
    for c in candidates[:50]:
        t = by_id.get(c["id"])
        if not t:
            continue
        alb = (t.get("album", {}) or {})
        hydrated.append({
            **c,
            "name": t.get("name", c.get("name", "")),
            "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
            "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
            "url": (t.get("external_urls", {}) or {}).get("spotify"),
            "album": alb.get("name", ""),
            "image": (alb.get("images")[0]["url"] if alb.get("images") else None),
            "popularity": t.get("popularity", 0),
            "release_date": alb.get("release_date"),
        })

    # -------------------------
    # 4) Filter junk early (covers/seasonal/etc.)
    # -------------------------
    filtered = []
    for c in hydrated:
        tmp = normalize_for_filtering(c)
        tmp["album"] = c.get("album", tmp["album"])

        if looks_like_cover(tmp):
            continue
        if looks_like_seasonal(tmp):
            continue

        filtered.append(c)

    if not filtered:
        return {
            "seed": {
                "id": track_id,
                "name": seed_track_name,
                "artists": [a.get("name", "") for a in seed_artists],
                "url": (seed.get("external_urls", {}) or {}).get("spotify"),
            },
            "count": 0,
            "results": [],
            "debug": {"note": "All candidates filtered out as cover/seasonal/junk."}
        }

    # -------------------------
    # 5) Fetch audio-features for candidates in batch + score similarity
    # -------------------------
    def _norm_tempo(x):
        # tempo can be ~60-200+. normalize into ~0-1-ish
        try:
            return float(x) / 200.0
        except Exception:
            return None

    def _vec_from_af(af: dict):
        # Pick stable core dimensions for "vibe"
        # (All are 0-1 except tempo.)
        return [
            af.get("danceability"),
            af.get("energy"),
            af.get("speechiness"),
            af.get("acousticness"),
            af.get("instrumentalness"),
            af.get("liveness"),
            af.get("valence"),
            _norm_tempo(af.get("tempo")),
        ]

    def _dist(a, b, w):
        # weighted euclidean on shared dims
        s = 0.0
        used = 0
        for i in range(len(w)):
            if a[i] is None or b[i] is None:
                continue
            d = (float(a[i]) - float(b[i]))
            s += w[i] * (d * d)
            used += 1
        if used == 0:
            return None
        return s ** 0.5

    # If we don't have seed audio-features, we can't do real similarity
    # so we fall back to popularity + playlist sources (still penalize same-artist).
    seed_vec = _vec_from_af(seed_af) if seed_af else None

    # Batch fetch audio-features for candidate ids
    af_ids = ",".join([c["id"] for c in filtered[:50]])
    af_by_id = {}
    if af_ids:
        af_data, err = spotify_get_json(request, "/audio-features", params={"ids": af_ids})
        if not err:
            for af in (af_data.get("audio_features", []) or []):
                if af and af.get("id"):
                    af_by_id[af["id"]] = af

    # weights (tweakable)
    # energy/valence/danceability dominate "feel"
    weights = [1.2, 1.4, 0.4, 0.9, 0.5, 0.5, 1.3, 0.8]

    scored = []
    for c in filtered:
        cid = c["id"]
        caf = af_by_id.get(cid)
        same_artist = any(aid in seed_artist_set for aid in (c.get("artist_ids") or []))

        # Base score: audio similarity if possible
        audio_score = None
        if seed_vec and caf:
            cvec = _vec_from_af(caf)
            d = _dist(seed_vec, cvec, weights)
            if d is not None:
                # convert distance to similarity-ish (higher is better)
                audio_score = 1.0 / (1.0 + d)

        # Build final score:
        # - if audio_score exists: heavily use it
        # - else: fall back to popularity + source weight
        source_weight = {"playlist": 0.15, "album": 0.10, "artist_top": 0.12, "artist_catalog": 0.10}
        base = (audio_score * 10.0) if audio_score is not None else 0.0
        base += (float(c.get("popularity", 0) or 0) / 100.0) * 1.0
        base += source_weight.get(c.get("source", ""), 0.05)

        # BIG IMPORTANT PART: penalize same-artist so it doesn't become "just Queen"
        if same_artist:
            base -= 0.75

        c["audio_score"] = round(audio_score, 4) if audio_score is not None else None
        c["final_score"] = round(base, 4)
        scored.append(c)

    scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # -------------------------
    # 6) Diversity caps + limit
    # -------------------------
    top = apply_diversity_caps(scored, max_per_artist=2, max_from_seed_album=3)
    top = top[:limit]

    source_counts = Counter([c.get("source") for c in candidates])

    return {
        "debug": {
            "seed_audio_features_ok": bool(seed_af),
            "candidate_source_counts": dict(source_counts),
            "same_artist_penalty_enabled": True,
            "audio_scored_count": sum(1 for r in top if r.get("audio_score") is not None),
        },
        "seed": {
            "id": track_id,
            "name": seed_track_name,
            "artists": [a.get("name", "") for a in seed_artists],
            "url": (seed.get("external_urls", {}) or {}).get("spotify"),
        },
        "count": len(top),
        "results": top,
    }

   

@app.get("/spotify/track")
def spotify_track(request: Request, track_id: str):
    data, err = spotify_get_json(request, f"/tracks/{track_id}", params={"market": "US"})
    if err:
        return err
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "artists": [a["name"] for a in data.get("artists", [])],
        "album": (data.get("album", {}) or {}).get("name"),
        "url": (data.get("external_urls", {}) or {}).get("spotify"),
    }

@app.get("/spotify/vibe_local")
def spotify_vibe_local(request: Request, track_id: str, limit: int = 10):
    index = load_index()
    if index is None:
        return {"error": "index.json not found. Run POST /index first."}

    # get seed track metadata from spotify
    seed, err = spotify_get_json(request, f"/tracks/{track_id}", params={"market": "US"})
    if err:
        return err

    seed_name = seed.get("name", "")
    seed_artists = [a.get("name", "") for a in seed.get("artists", [])]

    # find a matching local file
    match = find_best_local_match(index, seed_name, seed_artists)
    if not match:
        return {
            "error": "No matching local file found for this Spotify track.",
            "seed": {"id": track_id, "name": seed_name, "artists": seed_artists},
            "tip": "Add the song to your music/ folder (filename should include artist + title), then POST /index again."
        }

    # compute similarity against local index
    query_features = match["features"]
    scored = []
    for item in index:
        score = cosine_similarity(query_features, item["features"])
        scored.append({"filename": item["filename"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "seed": {
            "id": track_id,
            "name": seed_name,
            "artists": seed_artists,
            "matched_local_file": match["filename"],
        },
        "count": min(limit, len(scored)),
        "results": scored[:limit],
    }

"""
Add this endpoint to your main.py

This is the refactored /spotify/vibe_v2 endpoint that uses multi-signal scoring.
"""

# Add this import at the top of main.py:
# from vibe_v2 import (
#     compute_composite_score,
#     build_genre_search_queries,
#     parse_year,
#     WEIGHTS,
# )


@app.get("/spotify/vibe_v2")
def spotify_vibe_v2(request: Request, track_id: str, limit: int = 20):
    """
    Multi-signal "more like this" recommendations.
    
    Combines:
    - Audio features (energy, valence, tempo, key, etc.)
    - Genre overlap
    - Era/decade proximity
    - Related artist graph
    - Key/mode compatibility
    """
    
    # =========================================
    # 1) FETCH SEED TRACK + ALL METADATA
    # =========================================
    
    # Basic track info
    seed, err = spotify_get_json(request, f"/tracks/{track_id}", params={"market": "US"})
    if err:
        return {"error": "Could not fetch seed track", "details": err}
    
    seed_name = seed.get("name", "")
    seed_artists = seed.get("artists", []) or []
    seed_artist_ids = set(a.get("id") for a in seed_artists if a.get("id"))
    seed_artist_name = seed_artists[0].get("name", "") if seed_artists else ""
    seed_album = seed.get("album", {}) or {}
    seed_album_id = seed_album.get("id")
    seed_year = parse_year(seed_album.get("release_date"))
    
    # Audio features for seed
    seed_af, err = spotify_get_json(request, f"/audio-features/{track_id}")
    if err:
        seed_af = {}
        print(f"Warning: Could not fetch audio features for seed: {err}")
    
    # Genres from seed artist(s)
    seed_genres = []
    if seed_artist_ids:
        artist_ids_str = ",".join(list(seed_artist_ids)[:5])
        artists_data, err = spotify_get_json(request, "/artists", params={"ids": artist_ids_str})
        if not err:
            for a in artists_data.get("artists", []) or []:
                seed_genres.extend(a.get("genres", []))
    seed_genres = list(dict.fromkeys(seed_genres))[:10]  # dedupe, cap at 10
    
    # Related artists
    related_artist_ids = set()
    if seed_artist_ids:
        primary_artist_id = list(seed_artist_ids)[0]
        rel_data, err = spotify_get_json(request, f"/artists/{primary_artist_id}/related-artists")
        if not err:
            for a in (rel_data.get("artists", []) or [])[:20]:
                if a.get("id"):
                    related_artist_ids.add(a["id"])
    
    # =========================================
    # 2) BUILD CANDIDATE POOL
    # =========================================
    
    candidates: list[dict] = []
    
    # 2a) Related artists' top tracks
    for artist_id in list(related_artist_ids)[:10]:
        top_tracks, err = spotify_get_json(
            request,
            f"/artists/{artist_id}/top-tracks",
            params={"market": "US"}
        )
        if err:
            continue
        for t in (top_tracks.get("tracks", []) or [])[:5]:
            if not t or not t.get("id") or t["id"] == track_id:
                continue
            candidates.append({
                "id": t["id"],
                "name": t.get("name", ""),
                "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                "album": (t.get("album", {}) or {}).get("name", ""),
                "release_date": (t.get("album", {}) or {}).get("release_date"),
                "popularity": t.get("popularity", 0),
                "image": ((t.get("album", {}) or {}).get("images") or [{}])[0].get("url"),
                "url": (t.get("external_urls", {}) or {}).get("spotify"),
                "source": "related_artist_top",
            })
    
    # 2b) Seed album tracks (capped)
    if seed_album_id:
        album_tracks, err = spotify_get_json(
            request,
            f"/albums/{seed_album_id}/tracks",
            params={"limit": 15, "market": "US"}
        )
        if not err:
            for t in (album_tracks.get("items", []) or []):
                tid = t.get("id")
                if not tid or tid == track_id:
                    continue
                candidates.append({
                    "id": tid,
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "source": "seed_album",
                })
    
    # 2c) Genre-based playlist search
    search_queries = build_genre_search_queries(seed_genres, seed_name, seed_artist_name)
    
    for query in search_queries[:6]:
        add_playlist_tracks_as_candidates(
            request, query, candidates,
            max_playlists=1, per_playlist=8
        )
    
    # 2d) Direct track search for similar terms
    if seed_genres:
        genre_query = f"{seed_genres[0]} {seed_year or ''}"
        search_data, err = spotify_get_json(
            request, "/search",
            params={"q": genre_query, "type": "track", "limit": 20, "market": "US"}
        )
        if not err:
            for t in (search_data.get("tracks", {}).get("items", []) or []):
                if not t or not t.get("id") or t["id"] == track_id:
                    continue
                candidates.append({
                    "id": t["id"],
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "album": (t.get("album", {}) or {}).get("name", ""),
                    "release_date": (t.get("album", {}) or {}).get("release_date"),
                    "popularity": t.get("popularity", 0),
                    "image": ((t.get("album", {}) or {}).get("images") or [{}])[0].get("url"),
                    "url": (t.get("external_urls", {}) or {}).get("spotify"),
                    "source": "genre_search",
                })
    
    # Deduplicate
    candidates = dedupe_tracks(candidates)
    candidates = [c for c in candidates if c.get("id") != track_id]
    
    if not candidates:
        return {
            "seed": {"id": track_id, "name": seed_name, "artists": [a.get("name") for a in seed_artists]},
            "count": 0,
            "results": [],
            "debug": {"error": "No candidates found"}
        }
    
    # =========================================
    # 3) HYDRATE CANDIDATES (metadata + audio features + genres)
    # =========================================
    
    # Batch fetch track metadata
    candidate_ids = [c["id"] for c in candidates[:100]]
    tracks_by_id = {}
    
    for i in range(0, len(candidate_ids), 50):
        batch = candidate_ids[i:i+50]
        ids_str = ",".join(batch)
        tracks_data, err = spotify_get_json(request, "/tracks", params={"ids": ids_str, "market": "US"})
        if not err:
            for t in (tracks_data.get("tracks", []) or []):
                if t and t.get("id"):
                    tracks_by_id[t["id"]] = t
    
    # Batch fetch audio features
    af_by_id = {}
    for i in range(0, len(candidate_ids), 100):
        batch = candidate_ids[i:i+100]
        ids_str = ",".join(batch)
        af_data, err = spotify_get_json(request, "/audio-features", params={"ids": ids_str})
        if not err:
            for af in (af_data.get("audio_features", []) or []):
                if af and af.get("id"):
                    af_by_id[af["id"]] = af
    
    # Collect unique artist IDs for genre lookup
    all_artist_ids = set()
    for c in candidates[:100]:
        all_artist_ids.update(c.get("artist_ids", []))
    
    # Batch fetch artist genres
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
    # 4) FILTER + SCORE CANDIDATES
    # =========================================
    
    scored_results = []
    
    for c in candidates[:100]:
        cid = c["id"]
        
        # Get hydrated data
        track_data = tracks_by_id.get(cid, {})
        candidate_af = af_by_id.get(cid, {})
        
        # Build candidate genres from all artists
        candidate_genres = []
        for aid in c.get("artist_ids", []):
            candidate_genres.extend(genres_by_artist.get(aid, []))
        candidate_genres = list(dict.fromkeys(candidate_genres))
        
        # Update candidate with hydrated metadata
        album = track_data.get("album", {}) or {}
        c.update({
            "name": track_data.get("name", c.get("name", "")),
            "artists": [a.get("name", "") for a in (track_data.get("artists", []) or c.get("artists", []))],
            "album": album.get("name", c.get("album", "")),
            "release_date": album.get("release_date", c.get("release_date")),
            "popularity": track_data.get("popularity", c.get("popularity", 0)),
            "image": (album.get("images") or [{}])[0].get("url", c.get("image")),
            "url": (track_data.get("external_urls", {}) or {}).get("spotify", c.get("url")),
            "genres": candidate_genres,
        })
        
        # Filter: covers/seasonal/junk
        filter_obj = {
            "name": c.get("name", ""),
            "album": c.get("album", ""),
            "artists": c.get("artists", []),
        }
        if looks_like_cover(filter_obj):
            continue
        if looks_like_seasonal(filter_obj):
            continue
        
        # Compute composite score
        score_result = compute_composite_score(
            seed_af=seed_af,
            seed_genres=seed_genres,
            seed_year=seed_year,
            seed_artist_ids=seed_artist_ids,
            related_artist_ids=related_artist_ids,
            candidate=c,
            candidate_af=candidate_af,
            candidate_genres=candidate_genres,
        )
        
        c["score"] = score_result["final_score"]
        c["score_components"] = score_result["components"]
        
        scored_results.append(c)
    
    # Sort by final score
    scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # =========================================
    # 5) DIVERSITY CAPS + RETURN
    # =========================================
    
    final_results = apply_diversity_caps(scored_results, max_per_artist=2, max_from_seed_album=2)
    final_results = final_results[:limit]
    
    # Clean up output
    for r in final_results:
        # Remove internal fields
        r.pop("artist_ids", None)
        r.pop("source", None)
        r.pop("genres", None)
    
    return {
        "seed": {
            "id": track_id,
            "name": seed_name,
            "artists": [a.get("name", "") for a in seed_artists],
            "album": seed_album.get("name"),
            "year": seed_year,
            "genres": seed_genres[:5],
            "url": (seed.get("external_urls", {}) or {}).get("spotify"),
        },
        "weights": WEIGHTS,
        "count": len(final_results),
        "results": final_results,
        "debug": {
            "candidates_found": len(candidates),
            "candidates_scored": len(scored_results),
            "seed_audio_features": bool(seed_af),
            "related_artists_found": len(related_artist_ids),
        }
    }

"""
Add this endpoint to your main.py

/spotify/vibe_v3 - works without audio-features or related-artists APIs
"""

# Add this import at the top of main.py:
# from .vibe_v3 import (
#     compute_composite_score_v3,
#     build_search_queries_v3,
#     parse_year,
#     WEIGHTS_V3,
# )


@app.get("/spotify/vibe_v3")
def spotify_vibe_v3(request: Request, track_id: str, limit: int = 20):
    """
    "More like this" recommendations using only non-restricted Spotify APIs.
    
    Works with:
    - Genre overlap (artist metadata)
    - Era/year proximity (album metadata)
    - Playlist co-occurrence (search API)
    - Popularity signals
    
    Does NOT require:
    - audio-features API (restricted)
    - related-artists API (may be restricted)
    - recommendations API (deprecated)
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
    seed_album_id = seed_album.get("id")
    seed_year = parse_year(seed_album.get("release_date"))
    
    # Fetch genres from seed artist(s)
    seed_genres = []
    if seed_artist_ids:
        artist_ids_str = ",".join(list(seed_artist_ids)[:5])
        artists_data, err = spotify_get_json(request, "/artists", params={"ids": artist_ids_str})
        if not err:
            for a in artists_data.get("artists", []) or []:
                seed_genres.extend(a.get("genres", []))
    seed_genres = list(dict.fromkeys(seed_genres))[:10]  # dedupe, cap
    
    # =========================================
    # 2) BUILD CANDIDATE POOL
    # =========================================
    
    candidates: list[dict] = []
    
    # 2a) Seed album tracks
    if seed_album_id:
        album_tracks, err = spotify_get_json(
            request,
            f"/albums/{seed_album_id}/tracks",
            params={"limit": 20, "market": "US"}
        )
        if not err:
            for t in (album_tracks.get("items", []) or []):
                tid = t.get("id")
                if not tid or tid == track_id:
                    continue
                candidates.append({
                    "id": tid,
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "source": "seed_album",
                })
    
    # 2b) Search-based candidates (playlists and tracks)
    search_queries = build_search_queries_v3(seed_name, seed_artist_name, seed_genres, seed_year)
    
    for sq in search_queries:
        query = sq["query"]
        source = sq["source"]
        search_type = sq["type"]
        
        if search_type == "playlist":
            # Search for playlists, then get their tracks
            q = query if "playlist" in query.lower() else f"{query} playlist"
            
            pdata, err = spotify_get_json(
                request,
                "/search",
                params={"q": q, "type": "playlist", "limit": 2, "market": "US"}
            )
            if err:
                continue
            
            for playlist in (pdata.get("playlists", {}).get("items", []) or [])[:2]:
                if not playlist:
                    continue
                pid = playlist.get("id")
                if not pid:
                    continue
                
                # Get tracks from this playlist
                tracks_data, err = spotify_get_json(
                    request,
                    f"/playlists/{pid}/tracks",
                    params={"limit": 10, "market": "US"}
                )
                if err:
                    continue
                
                for item in (tracks_data.get("items", []) or []):
                    t = (item or {}).get("track")
                    if not t or not t.get("id") or t["id"] == track_id:
                        continue
                    
                    candidates.append({
                        "id": t["id"],
                        "name": t.get("name", ""),
                        "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                        "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                        "album": (t.get("album", {}) or {}).get("name", ""),
                        "release_date": (t.get("album", {}) or {}).get("release_date"),
                        "popularity": t.get("popularity", 0),
                        "image": ((t.get("album", {}) or {}).get("images") or [{}])[0].get("url"),
                        "url": (t.get("external_urls", {}) or {}).get("spotify"),
                        "source": source,
                    })
        
        elif search_type == "track":
            # Direct track search
            tdata, err = spotify_get_json(
                request,
                "/search",
                params={"q": query, "type": "track", "limit": 15, "market": "US"}
            )
            if err:
                continue
            
            for t in (tdata.get("tracks", {}).get("items", []) or []):
                if not t or not t.get("id") or t["id"] == track_id:
                    continue
                
                candidates.append({
                    "id": t["id"],
                    "name": t.get("name", ""),
                    "artists": [a.get("name", "") for a in (t.get("artists", []) or [])],
                    "artist_ids": [a.get("id") for a in (t.get("artists", []) or []) if a.get("id")],
                    "album": (t.get("album", {}) or {}).get("name", ""),
                    "release_date": (t.get("album", {}) or {}).get("release_date"),
                    "popularity": t.get("popularity", 0),
                    "image": ((t.get("album", {}) or {}).get("images") or [{}])[0].get("url"),
                    "url": (t.get("external_urls", {}) or {}).get("spotify"),
                    "source": source,
                })
    
    # Deduplicate
    candidates = dedupe_tracks(candidates)
    candidates = [c for c in candidates if c.get("id") != track_id]
    
    if not candidates:
        return {
            "seed": {"id": track_id, "name": seed_name, "artists": [a.get("name") for a in seed_artists]},
            "count": 0,
            "results": [],
            "debug": {"error": "No candidates found"}
        }
    
    # =========================================
    # 3) HYDRATE CANDIDATES (fill in missing metadata)
    # =========================================
    
    # Find candidates that need hydration
    needs_hydration = [c for c in candidates if not c.get("release_date") or not c.get("url")]
    hydrate_ids = [c["id"] for c in needs_hydration[:50]]
    
    if hydrate_ids:
        ids_str = ",".join(hydrate_ids)
        tracks_data, err = spotify_get_json(request, "/tracks", params={"ids": ids_str, "market": "US"})
        if not err:
            tracks_by_id = {t["id"]: t for t in (tracks_data.get("tracks", []) or []) if t and t.get("id")}
            
            for c in candidates:
                if c["id"] in tracks_by_id:
                    t = tracks_by_id[c["id"]]
                    album = t.get("album", {}) or {}
                    c.update({
                        "name": t.get("name", c.get("name", "")),
                        "album": album.get("name", c.get("album", "")),
                        "release_date": album.get("release_date", c.get("release_date")),
                        "popularity": t.get("popularity", c.get("popularity", 0)),
                        "image": (album.get("images") or [{}])[0].get("url", c.get("image")),
                        "url": (t.get("external_urls", {}) or {}).get("spotify", c.get("url")),
                    })
    
    # Fetch genres for all candidate artists
    all_artist_ids = set()
    for c in candidates:
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
    # 4) FILTER + SCORE
    # =========================================
    
    scored_results = []
    
    for c in candidates:
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
        
        # Score
        score_result = compute_composite_score_v3(
            seed_name=seed_name,
            seed_genres=seed_genres,
            seed_year=seed_year,
            seed_artist_ids=seed_artist_ids,
            candidate=c,
            candidate_genres=candidate_genres,
        )
        
        c["score"] = score_result["final_score"]
        c["score_components"] = score_result["components"]
        
        scored_results.append(c)
    
    # Sort by score
    scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # =========================================
    # 5) DIVERSITY CAPS + CLEAN OUTPUT
    # =========================================
    
    final_results = apply_diversity_caps(scored_results, max_per_artist=2, max_from_seed_album=2)
    final_results = final_results[:limit]
    
    # Clean up internal fields
    for r in final_results:
        r.pop("artist_ids", None)
        r.pop("source", None)
    
    return {
        "seed": {
            "id": track_id,
            "name": seed_name,
            "artists": [a.get("name", "") for a in seed_artists],
            "album": seed_album.get("name"),
            "year": seed_year,
            "genres": seed_genres[:5],
            "url": (seed.get("external_urls", {}) or {}).get("spotify"),
        },
        "weights": WEIGHTS_V3,
        "count": len(final_results),
        "results": final_results,
        "debug": {
            "candidates_found": len(candidates),
            "candidates_scored": len(scored_results),
            "queries_used": len(search_queries),
        }
    }

@app.get("/spotify/vibe_v4")
def spotify_vibe_v4(request: Request, track_id: str, limit: int = 20):
    """
    Playlist co-occurrence based recommendations.
    
    Logic: If humans put songs together in playlists, they probably vibe similarly.
    Songs appearing in multiple playlists with the seed = higher confidence.
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
    
    # Fetch genres from seed artist(s)
    seed_genres = []
    if seed_artist_ids:
        artist_ids_str = ",".join(list(seed_artist_ids)[:5])
        artists_data, err = spotify_get_json(request, "/artists", params={"ids": artist_ids_str})
        if not err:
            for a in artists_data.get("artists", []) or []:
                seed_genres.extend(a.get("genres", []))
    seed_genres = list(dict.fromkeys(seed_genres))[:10]
    
    # =========================================
    # 2) FIND PLAYLISTS & TRACK CO-OCCURRENCE
    # =========================================
    
    # Track how many playlists each candidate appears in
    candidate_playlist_count: dict[str, int] = defaultdict(int)
    candidate_data: dict[str, dict] = {}  # store track metadata
    playlists_searched = 0
    playlists_with_seed = 0
    
    # Build search queries
    queries = build_playlist_queries_v4(seed_name, seed_artist_name)
    
    for query in queries:
        # Search for playlists
        pdata, err = spotify_get_json(
            request,
            "/search",
            params={"q": query, "type": "playlist", "limit": 5, "market": "US"}
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
            
            # Get tracks from this playlist
            tracks_data, err = spotify_get_json(
                request,
                f"/playlists/{pid}/tracks",
                params={"limit": 50, "market": "US"}  # get more tracks per playlist
            )
            if err:
                continue
            
            items = tracks_data.get("items", []) or []
            
            # Check if seed track is in this playlist
            playlist_track_ids = set()
            for item in items:
                t = (item or {}).get("track")
                if t and t.get("id"):
                    playlist_track_ids.add(t["id"])
            
            seed_in_playlist = track_id in playlist_track_ids
            if seed_in_playlist:
                playlists_with_seed += 1
            
            # Add all tracks as candidates (with co-occurrence boost if seed is present)
            for item in items:
                t = (item or {}).get("track")
                if not t or not t.get("id"):
                    continue
                
                tid = t["id"]
                if tid == track_id:
                    continue  # skip seed itself
                
                # Count co-occurrence (stronger signal if seed is in same playlist)
                if seed_in_playlist:
                    candidate_playlist_count[tid] += 2  # double weight
                else:
                    candidate_playlist_count[tid] += 1
                
                # Store track data (keep first occurrence's data)
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
    # 3) FETCH GENRES FOR CANDIDATES
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
    # 4) SCORE CANDIDATES
    # =========================================
    
    max_cooccurrence = max(candidate_playlist_count.values()) if candidate_playlist_count else 1
    scored_results = []
    
    for tid, c in candidate_data.items():
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
        
        # Get co-occurrence count
        cooccur_count = candidate_playlist_count.get(tid, 0)
        
        # Score
        score_result = compute_composite_score_v4(
            seed_name=seed_name,
            seed_genres=seed_genres,
            seed_year=seed_year,
            seed_artist_ids=seed_artist_ids,
            candidate=c,
            candidate_genres=candidate_genres,
            cooccurrence_count=cooccur_count,
            max_cooccurrence=max_cooccurrence,
        )
        
        c["score"] = score_result["final_score"]
        c["score_components"] = score_result["components"]
        
        scored_results.append(c)
    
    # Sort by score
    scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # =========================================
    # 5) DIVERSITY CAPS + OUTPUT
    # =========================================
    
    final_results = apply_diversity_caps(scored_results, max_per_artist=2, max_from_seed_album=2)
    final_results = final_results[:limit]
    
    # Clean up
    for r in final_results:
        r.pop("artist_ids", None)
    
    return {
        "seed": {
            "id": track_id,
            "name": seed_name,
            "artists": [a.get("name", "") for a in seed_artists],
            "album": seed_album.get("name"),
            "year": seed_year,
            "genres": seed_genres[:5],
            "url": (seed.get("external_urls", {}) or {}).get("spotify"),
        },
        "weights": WEIGHTS_V4,
        "count": len(final_results),
        "results": final_results,
        "debug": {
            "playlists_searched": playlists_searched,
            "playlists_with_seed": playlists_with_seed,
            "unique_candidates": len(candidate_data),
            "candidates_scored": len(scored_results),
            "max_cooccurrence": max_cooccurrence,
        }
    }
