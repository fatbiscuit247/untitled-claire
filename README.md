# Soundscape 🎵
### *escape into music*

> A mood-responsive music discovery app that goes beyond Spotify's built-in recommendations — finding songs that match not just genre, but emotional tone and lyrical feel.

**[Live Demo](#)** · **[GitHub](https://github.com/fatbiscuit247/untitled-claire)**

---

## What is Soundscape?

Most music recommendation algorithms rely on audio features or listening history. Soundscape takes a different approach — it analyzes **how humans naturally group songs together** in playlists, combined with **lyrical theme detection**, to surface recommendations that truly match a song's vibe.

Search any song. Soundscape detects its emotional themes, finds musically and lyrically similar tracks, and renders a living, breathing sky that shifts to match the mood.

---

## How It Works

### Recommendation Engine
Soundscape uses a multi-signal scoring algorithm (V5) that combines:

- **Playlist co-occurrence** — searches Spotify playlists to find songs that humans naturally group together, treating this as a strong signal of musical similarity
- **Lyrics analysis** — fetches lyrics via the Genius API and detects emotional themes (melancholy, nostalgia, euphoria, heartbreak, existential, and more)
- **Last.fm artist similarity** — leverages community-driven artist tags and similarity graphs
- **Album art color analysis** — extracts dominant colors from album artwork using PIL
- **Era & genre matching** — soft-weights by release year and Spotify genre tags

### Dynamic Visual System
The UI responds in real time to the detected mood of the seed track:

| Detected Theme | Visual Effect |
|---|---|
| Melancholy / Heartbreak | 🌧️ Rain |
| Party / Empowerment / Hope | ☀️ Sun rays |
| Nostalgia / Romance | 🌫️ Mist + stars |
| Existential | ✨ Stars |
| Album art | 🎨 Sky gradient shifts to match album colors |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Music Data | Spotify Web API |
| Lyrics | Genius API + BeautifulSoup scraping |
| Artist Similarity | Last.fm API |
| Color Extraction | ColorThief.js (frontend), PIL (backend) |
| Frontend | Vanilla JS, CSS animations |
| Deployment | Railway |

---

## Features

- 🔍 Search any song on Spotify
- 🎯 Multi-signal recommendation algorithm (playlist co-occurrence + lyrics NLP + Last.fm)
- 🌈 Dynamic sky gradient extracted from album artwork
- ⛅ Animated cloud layer with mood-responsive weather effects
- 🌠 Shooting stars on recommendation load
- ⚡ Lyrics caching system to minimize API calls and improve response time

---

## Running Locally

### Prerequisites
- Python 3.10+
- Spotify Developer account ([create app here](https://developer.spotify.com/dashboard))
- Genius API token ([get one here](https://genius.com/api-clients))
- Last.fm API key ([get one here](https://www.last.fm/api/account/create))

### Setup

```bash
# Clone the repo
git clone https://github.com/fatbiscuit247/untitled-claire.git
cd untitled-claire

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Fill in your API keys

# Run the app
uvicorn app.main:app --reload --port 8000
```

### Environment Variables

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_REDIRECT_URI=http://localhost:8000/auth/callback
SESSION_SECRET=any_random_secret_string
GENIUS_ACCESS_TOKEN=your_genius_token
LASTFM_API_KEY=your_lastfm_key
```

---

## Why Not Just Use Spotify's Recommendation API?

Spotify's `/recommendations` endpoint requires elevated API access that isn't available to new developer accounts. More importantly, audio feature-based recommendations (tempo, energy, danceability) often miss the *emotional* quality of a song.

Soundscape's playlist co-occurrence approach captures something different: **the human intuition behind "these songs belong together"** — which turns out to be a surprisingly powerful signal.

---

## License

MIT