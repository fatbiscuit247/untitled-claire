"""
Vibe V5: Playlist Co-occurrence + Lyrics Analysis + Last.fm Integration

Combines V4's playlist co-occurrence with:
- Lyrical similarity (sentiment, themes, mood)
- Last.fm user-generated tags (better genre coverage for indie artists)
- Last.fm similar artists (community-driven similarity)

Requires: GENIUS_ACCESS_TOKEN and LASTFM_API_KEY in .env
"""

from collections import defaultdict, Counter
from typing import Optional
import re
import os
import requests

from .lastfm_cache import (
    get_cached_tags,
    cache_tags,
    get_cached_similar,
    cache_similar,
)

from .album_art_cache import (
    get_cached_album_art,
    cache_album_art,
)


# ============================================================
# CONFIGURATION
# ============================================================

def get_lastfm_api_key():
    """Get Last.fm API key (read at call time, not import time)."""
    return os.getenv("LASTFM_API_KEY")

WEIGHTS_V5 = {
    "cooccurrence": 0.30,      # playlist co-occurrence (was 0.32)
    "lyrics": 0.18,            # lyrical/thematic similarity
    "era": 0.10,               # temporal proximity (was 0.12)
    "genre": 0.06,             # spotify genres
    "popularity": 0.10,        # quality signal
    "name_similarity": 0.08,   # penalize covers/same song
    "lastfm_tags": 0.08,       # user-generated tags
    "lastfm_similar": 0.06,    # community similar artists
    "album_art": 0.04,         # NEW: visual aesthetic similarity
}


# ============================================================
# LAST.FM API HELPERS
# ============================================================

def lastfm_get_artist_tags(artist_name: str) -> list[str]:
    """
    Fetch top tags for an artist from Last.fm.
    Returns list of tag names (lowercase), or empty list on failure.
    """
    # Check cache first
    cached = get_cached_tags(artist_name)
    if cached is not None:
        return cached
    
    api_key = get_lastfm_api_key()
    if not api_key:
        print("Last.fm API key not found")
        return []
    
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getTopTags",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
    }
    
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        
        data = r.json()
        
        # Check for error response
        if "error" in data:
            cache_tags(artist_name, [])  # cache empty to avoid re-fetching
            return []
        
        tags_data = data.get("toptags", {}).get("tag", [])
        
        # Extract tag names, filter low-count ones
        tags = []
        for t in tags_data[:15]:  # top 15 tags
            name = t.get("name", "").lower().strip()
            count = int(t.get("count", 0))
            if name and count >= 10:  # only tags with decent usage
                tags.append(name)
        
        cache_tags(artist_name, tags)
        return tags
        
    except Exception as e:
        print(f"Last.fm tags error for {artist_name}: {e}")
        return []


def lastfm_get_similar_artists(artist_name: str, limit: int = 20) -> list[str]:
    """
    Fetch similar artists from Last.fm.
    Returns list of artist names (lowercase), or empty list on failure.
    """
    # Check cache first
    cached = get_cached_similar(artist_name)
    if cached is not None:
        return cached
    
    api_key = get_lastfm_api_key()
    if not api_key:
        print("Last.fm API key not found")
        return []
    
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getSimilar",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        
        data = r.json()
        
        if "error" in data:
            cache_similar(artist_name, [])
            return []
        
        similar_data = data.get("similarartists", {}).get("artist", [])
        
        similar = []
        for a in similar_data:
            name = a.get("name", "").lower().strip()
            if name:
                similar.append(name)
        
        cache_similar(artist_name, similar)
        return similar
        
    except Exception as e:
        print(f"Last.fm similar error for {artist_name}: {e}")
        return []


# ============================================================
# LAST.FM SCORING FUNCTIONS
# ============================================================

def score_lastfm_tags(seed_tags: list[str], candidate_tags: list[str]) -> float:
    """
    Compare Last.fm tags between seed and candidate.
    Uses Jaccard similarity with partial match bonus.
    """
    if not seed_tags or not candidate_tags:
        return 0.5  # neutral if either is missing
    
    seed_set = set(seed_tags)
    cand_set = set(candidate_tags)
    
    # Exact matches
    intersection = len(seed_set & cand_set)
    union = len(seed_set | cand_set)
    jaccard = intersection / union if union > 0 else 0
    
    # Partial matches (e.g., "indie rock" matches "indie")
    partial_bonus = 0
    for st in seed_set:
        for ct in cand_set:
            if st != ct:
                # Check if one contains the other
                if st in ct or ct in st:
                    partial_bonus += 0.05
                # Check word overlap
                st_words = set(st.split())
                ct_words = set(ct.split())
                if st_words & ct_words:
                    partial_bonus += 0.03
    
    return min(1.0, jaccard + partial_bonus)


def score_lastfm_similar(
    seed_artist: str,
    candidate_artists: list[str],
    similar_artists: list[str]
) -> float:
    """
    Check if candidate's artists appear in seed's Last.fm similar artists.
    Returns 1.0 if direct match, scaled score based on position, 0.3 if no match.
    """
    if not similar_artists or not candidate_artists:
        return 0.5  # neutral if data missing
    
    # Normalize for comparison
    similar_lower = [a.lower() for a in similar_artists]
    
    for cand_artist in candidate_artists:
        cand_lower = cand_artist.lower()
        
        if cand_lower in similar_lower:
            # Found! Score based on position (earlier = more similar)
            position = similar_lower.index(cand_lower)
            # Position 0 = 1.0, position 19 = 0.6
            return 1.0 - (position * 0.02)
    
    return 0.3  # no match


# ============================================================
# ALBUM ART ANALYSIS
# ============================================================

def analyze_album_art(image_url: str) -> dict | None:
    """
    Analyze album art for visual characteristics.
    Returns dict with brightness (0-1), saturation (0-1), warmth (-1 to 1).
    """
    if not image_url:
        return None
    
    # Check cache first
    cached = get_cached_album_art(image_url)
    if cached is not None:
        return cached
    
    try:
        from PIL import Image
        from io import BytesIO
        
        # Fetch image
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            return None
        
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
        
        # Resize to speed up analysis (64x64 is enough for color stats)
        img = img.resize((64, 64))
        
        pixels = list(img.getdata())
        
        total_brightness = 0
        total_saturation = 0
        total_warmth = 0
        
        for r, g, b in pixels:
            # Brightness: average of RGB, normalized to 0-1
            brightness = (r + g + b) / (3 * 255)
            total_brightness += brightness
            
            # Saturation: difference between max and min channel
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            if max_c > 0:
                saturation = (max_c - min_c) / max_c
            else:
                saturation = 0
            total_saturation += saturation
            
            # Warmth: red/yellow vs blue
            # Positive = warm (more red/yellow), negative = cool (more blue)
            warmth = ((r - b) / 255)  # Simple: red minus blue
            total_warmth += warmth
        
        num_pixels = len(pixels)
        
        analysis = {
            "brightness": round(total_brightness / num_pixels, 3),
            "saturation": round(total_saturation / num_pixels, 3),
            "warmth": round(total_warmth / num_pixels, 3),
        }
        
        cache_album_art(image_url, analysis)
        return analysis
        
    except Exception as e:
        print(f"Album art analysis error: {e}")
        return None


def score_album_art_similarity(seed_analysis: dict | None, candidate_analysis: dict | None) -> float:
    """
    Compare album art visual characteristics.
    Returns similarity score 0-1.
    """
    if not seed_analysis or not candidate_analysis:
        return 0.5  # neutral if either is missing
    
    # Compare each dimension
    brightness_diff = abs(seed_analysis.get("brightness", 0.5) - candidate_analysis.get("brightness", 0.5))
    saturation_diff = abs(seed_analysis.get("saturation", 0.5) - candidate_analysis.get("saturation", 0.5))
    warmth_diff = abs(seed_analysis.get("warmth", 0) - candidate_analysis.get("warmth", 0))
    
    # Convert differences to similarity scores (0 diff = 1.0, max diff = 0.0)
    brightness_sim = 1.0 - brightness_diff
    saturation_sim = 1.0 - saturation_diff
    warmth_sim = 1.0 - (warmth_diff / 2)  # warmth ranges -1 to 1, so max diff is 2
    
    # Weighted combination (brightness matters most for "mood")
    combined = (
        brightness_sim * 0.5 +
        saturation_sim * 0.3 +
        warmth_sim * 0.2
    )
    
    return round(combined, 3)


# ============================================================
# LYRICS THEME DETECTION
# ============================================================

# Theme keyword dictionaries (words that indicate each theme)
THEME_KEYWORDS = {
    "love_romantic": [
        "love", "heart", "kiss", "hold", "touch", "forever", "baby", 
        "darling", "beautiful", "eyes", "fall for", "adore", "desire",
        "passion", "romantic", "embrace", "sweetheart", "devotion"
    ],
    "heartbreak": [
        "cry", "tears", "pain", "hurt", "gone", "leave", "lost", "miss",
        "broken", "alone", "goodbye", "forget", "regret", "sorry", "end",
        "over", "without you", "let go", "walking away", "memories"
    ],
    "party_fun": [
        "party", "dance", "night", "club", "drink", "fun", "crazy", 
        "wild", "celebrate", "friday", "weekend", "hands up", "move",
        "groove", "beat", "dj", "floor", "lights", "turn up"
    ],
    "empowerment": [
        "strong", "power", "fight", "stand", "rise", "believe", "dream",
        "winner", "champion", "unstoppable", "fearless", "brave", "conquer",
        "overcome", "strength", "confident", "proud", "warrior"
    ],
    "melancholy": [
        "sad", "lonely", "empty", "dark", "rain", "cold", "shadow",
        "fade", "silence", "hollow", "numb", "drown", "weight", "heavy",
        "grey", "bleak", "sorrow", "despair", "hopeless"
    ],
    "rebellion": [
        "fight", "rebel", "break", "free", "rules", "system", "against",
        "revolution", "riot", "scream", "rage", "anger", "burn", "destroy",
        "anarchy", "resist", "defy", "middle finger"
    ],
    "nostalgia": [
        "remember", "memories", "past", "young", "days", "time", "back",
        "childhood", "old", "years", "used to", "way back", "once",
        "those days", "looking back", "reminisce"
    ],
    "hope_inspiration": [
        "hope", "light", "tomorrow", "believe", "dream", "faith", "new",
        "begin", "start", "change", "better", "someday", "possible",
        "bright", "future", "wish", "stars", "sky"
    ],
    "existential": [
        "life", "death", "meaning", "why", "world", "time", "universe",
        "soul", "exist", "purpose", "truth", "reality", "question",
        "wonder", "infinite", "eternity", "beyond"
    ],
    "sensual": [
        "body", "skin", "touch", "feel", "close", "heat", "sweat",
        "tonight", "bed", "lips", "taste", "want", "need", "desire",
        "seduce", "tempt", "pleasure"
    ],
}

# Sentiment word lists
POSITIVE_WORDS = [
    "love", "happy", "joy", "beautiful", "amazing", "wonderful", "great",
    "good", "best", "smile", "laugh", "sun", "light", "hope", "dream",
    "alive", "free", "peace", "perfect", "paradise", "heaven", "blessed"
]

NEGATIVE_WORDS = [
    "hate", "sad", "pain", "hurt", "cry", "tears", "dark", "death",
    "kill", "die", "suffer", "broken", "lost", "alone", "fear", "angry",
    "rage", "hell", "devil", "damn", "curse", "nightmare", "despair"
]


def analyze_lyrics(lyrics: str) -> dict:
    """
    Analyze lyrics and return theme scores and sentiment.
    Returns dict with theme scores (0-1) and overall sentiment (-1 to 1).
    """
    if not lyrics:
        return {"themes": {}, "sentiment": 0.0, "word_count": 0}
    
    # Normalize text
    text = lyrics.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {"themes": {}, "sentiment": 0.0, "word_count": 0}
    
    # Count theme matches
    theme_scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        matches = sum(1 for w in words if any(kw in w for kw in keywords))
        # Also check for multi-word phrases
        for kw in keywords:
            if " " in kw and kw in text:
                matches += 2  # bonus for phrase match
        
        # Normalize by word count (avoid division by zero)
        theme_scores[theme] = min(1.0, matches / (word_count * 0.05))
    
    # Calculate sentiment
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        sentiment = (positive_count - negative_count) / total_sentiment_words
    else:
        sentiment = 0.0
    
    return {
        "themes": theme_scores,
        "sentiment": round(sentiment, 3),
        "word_count": word_count,
    }


def score_lyrics_similarity(seed_analysis: dict, candidate_analysis: dict) -> float:
    """
    Compare two lyrics analyses and return similarity score (0-1).
    """
    if not seed_analysis.get("themes") or not candidate_analysis.get("themes"):
        return 0.5  # neutral if either is missing
    
    seed_themes = seed_analysis["themes"]
    cand_themes = candidate_analysis["themes"]
    
    # Compare theme profiles (cosine-like similarity)
    dot_product = 0.0
    seed_magnitude = 0.0
    cand_magnitude = 0.0
    
    all_themes = set(seed_themes.keys()) | set(cand_themes.keys())
    
    for theme in all_themes:
        s = seed_themes.get(theme, 0)
        c = cand_themes.get(theme, 0)
        dot_product += s * c
        seed_magnitude += s * s
        cand_magnitude += c * c
    
    if seed_magnitude == 0 or cand_magnitude == 0:
        theme_similarity = 0.5
    else:
        theme_similarity = dot_product / ((seed_magnitude ** 0.5) * (cand_magnitude ** 0.5))
    
    # Compare sentiment (closer = better)
    seed_sentiment = seed_analysis.get("sentiment", 0)
    cand_sentiment = candidate_analysis.get("sentiment", 0)
    sentiment_diff = abs(seed_sentiment - cand_sentiment)
    sentiment_similarity = 1.0 - (sentiment_diff / 2.0)  # normalize to 0-1
    
    # Combine (theme matters more than sentiment)
    combined = (theme_similarity * 0.7) + (sentiment_similarity * 0.3)
    
    return round(combined, 3)


# ============================================================
# SCORING FUNCTIONS (same as V4, adjusted weights)
# ============================================================

def score_cooccurrence(count: int, max_count: int) -> float:
    if max_count == 0:
        return 0.5
    if count >= 3:
        return 1.0
    elif count == 2:
        return 0.85
    elif count == 1:
        return 0.6
    else:
        return 0.3


def score_era_proximity(seed_year: Optional[int], candidate_year: Optional[int]) -> float:
    if seed_year is None or candidate_year is None:
        return 0.5
    
    diff = abs(seed_year - candidate_year)
    
    if diff <= 2:
        return 1.0
    elif diff <= 5:
        return 0.85
    elif diff <= 10:
        return 0.7
    elif diff <= 15:
        return 0.5
    elif diff <= 25:
        return 0.3
    else:
        return 0.1


def score_genre_overlap(seed_genres: list[str], candidate_genres: list[str]) -> float:
    if not seed_genres or not candidate_genres:
        return 0.5
    
    seed_set = set(g.lower().strip() for g in seed_genres)
    cand_set = set(g.lower().strip() for g in candidate_genres)
    
    if not seed_set or not cand_set:
        return 0.5
    
    intersection = len(seed_set & cand_set)
    union = len(seed_set | cand_set)
    jaccard = intersection / union if union > 0 else 0
    
    partial_bonus = 0
    for sg in seed_set:
        for cg in cand_set:
            if sg != cg and (sg in cg or cg in sg):
                partial_bonus += 0.1
    
    return min(1.0, jaccard + partial_bonus)


def score_popularity(popularity: int) -> float:
    p = max(0, min(100, popularity or 0))
    return (p / 100) ** 0.7


def score_name_similarity(seed_name: str, candidate_name: str) -> float:
    def normalize(s):
        s = (s or "").lower()
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    seed_norm = normalize(seed_name)
    cand_norm = normalize(candidate_name)
    
    if not seed_norm or not cand_norm:
        return 0.5
    
    if seed_norm == cand_norm:
        return 0.0
    
    seed_words = set(seed_norm.split())
    cand_words = set(cand_norm.split())
    
    if seed_words and cand_words:
        overlap = len(seed_words & cand_words) / len(seed_words | cand_words)
        if overlap > 0.7:
            return 0.2
        elif overlap > 0.4:
            return 0.5
    
    return 0.8


def compute_composite_score_v5(
    seed_name: str,
    seed_genres: list[str],
    seed_year: Optional[int],
    seed_artist_ids: set[str],
    seed_lyrics_analysis: dict,
    seed_lastfm_tags: list[str],
    seed_similar_artists: list[str],
    seed_album_art_analysis: dict | None,
    candidate: dict,
    candidate_genres: list[str],
    candidate_lyrics_analysis: dict,
    candidate_lastfm_tags: list[str],
    candidate_album_art_analysis: dict | None,
    cooccurrence_count: int,
    max_cooccurrence: int,
) -> dict:
    """
    Compute final score with co-occurrence + lyrics + Last.fm + album art + other signals.
    """
    
    # Parse candidate year
    cand_year = None
    release_date = candidate.get("release_date")
    if release_date:
        try:
            cand_year = int(release_date[:4])
        except:
            pass
    
    # Same artist check
    cand_artist_ids = set(candidate.get("artist_ids", []))
    is_same_artist = bool(cand_artist_ids & seed_artist_ids)
    
    # Get candidate artists for similar artist check
    candidate_artists = candidate.get("artists", [])
    
    # Compute all scores
    cooccur_score = score_cooccurrence(cooccurrence_count, max_cooccurrence)
    lyrics_score = score_lyrics_similarity(seed_lyrics_analysis, candidate_lyrics_analysis)
    era_score = score_era_proximity(seed_year, cand_year)
    genre_score = score_genre_overlap(seed_genres, candidate_genres)
    pop_score = score_popularity(candidate.get("popularity", 0))
    name_score = score_name_similarity(seed_name, candidate.get("name", ""))
    
    # Last.fm scores
    lastfm_tags_score = score_lastfm_tags(seed_lastfm_tags, candidate_lastfm_tags)
    lastfm_similar_score = score_lastfm_similar(
        seed_name,  # not actually used in current impl
        candidate_artists,
        seed_similar_artists
    )
    
    # Album art score
    album_art_score = score_album_art_similarity(seed_album_art_analysis, candidate_album_art_analysis)
    
    components = {
        "cooccurrence": round(cooccur_score, 3),
        "lyrics": round(lyrics_score, 3),
        "era": round(era_score, 3),
        "genre": round(genre_score, 3),
        "popularity": round(pop_score, 3),
        "name_similarity": round(name_score, 3),
        "lastfm_tags": round(lastfm_tags_score, 3),
        "lastfm_similar": round(lastfm_similar_score, 3),
        "album_art": round(album_art_score, 3),
        "playlist_count": cooccurrence_count,
    }
    
    # Weighted sum
    final_score = (
        WEIGHTS_V5["cooccurrence"] * cooccur_score +
        WEIGHTS_V5["lyrics"] * lyrics_score +
        WEIGHTS_V5["era"] * era_score +
        WEIGHTS_V5["genre"] * genre_score +
        WEIGHTS_V5["popularity"] * pop_score +
        WEIGHTS_V5["name_similarity"] * name_score +
        WEIGHTS_V5["lastfm_tags"] * lastfm_tags_score +
        WEIGHTS_V5["lastfm_similar"] * lastfm_similar_score +
        WEIGHTS_V5["album_art"] * album_art_score
    )
    
    # Same-artist penalty
    if is_same_artist:
        final_score *= 0.5
        components["same_artist_penalty"] = True
    
    return {
        "final_score": round(final_score, 4),
        "components": components,
    }


def build_playlist_queries_v5(seed_name: str, seed_artist: str) -> list[str]:
    """Same as V4"""
    queries = []
    
    if seed_name and seed_artist:
        queries.append(f"{seed_name} {seed_artist}")
        queries.append(f"{seed_artist} {seed_name}")
    
    if seed_name:
        queries.append(seed_name)
        queries.append(f"{seed_name} mix")
        queries.append(f"songs like {seed_name}")
    
    if seed_artist:
        queries.append(f"{seed_artist} mix")
        queries.append(f"{seed_artist} radio")
        queries.append(f"{seed_artist} essentials")
    
    return queries[:8]


def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except:
        return None