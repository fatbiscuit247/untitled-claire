"""
Last.fm API response caching.

Caches artist tags and similar artists to avoid repeated API calls.
Simple JSON file approach matching lyrics_cache.py pattern.
"""

import json
import re
from pathlib import Path


TAGS_CACHE_PATH = "lastfm_tags_cache.json"
SIMILAR_CACHE_PATH = "lastfm_similar_cache.json"


def _normalize_artist_key(artist_name: str) -> str:
    """Create a normalized cache key from artist name."""
    s = (artist_name or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# TAGS CACHE
# ============================================================

def load_tags_cache() -> dict:
    """Load the tags cache from disk."""
    if not Path(TAGS_CACHE_PATH).exists():
        return {}
    try:
        with open(TAGS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tags cache: {e}")
        return {}


def save_tags_cache(cache: dict) -> None:
    """Save the tags cache to disk."""
    try:
        with open(TAGS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving tags cache: {e}")


def get_cached_tags(artist_name: str) -> list[str] | None:
    """Get cached tags for an artist. Returns list of tag names or None."""
    cache = load_tags_cache()
    key = _normalize_artist_key(artist_name)
    return cache.get(key)


def cache_tags(artist_name: str, tags: list[str]) -> None:
    """Store tags for an artist in cache."""
    cache = load_tags_cache()
    key = _normalize_artist_key(artist_name)
    cache[key] = tags
    save_tags_cache(cache)


# ============================================================
# SIMILAR ARTISTS CACHE
# ============================================================

def load_similar_cache() -> dict:
    """Load the similar artists cache from disk."""
    if not Path(SIMILAR_CACHE_PATH).exists():
        return {}
    try:
        with open(SIMILAR_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading similar cache: {e}")
        return {}


def save_similar_cache(cache: dict) -> None:
    """Save the similar artists cache to disk."""
    try:
        with open(SIMILAR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving similar cache: {e}")


def get_cached_similar(artist_name: str) -> list[str] | None:
    """Get cached similar artists. Returns list of artist names or None."""
    cache = load_similar_cache()
    key = _normalize_artist_key(artist_name)
    return cache.get(key)


def cache_similar(artist_name: str, similar_artists: list[str]) -> None:
    """Store similar artists in cache."""
    cache = load_similar_cache()
    key = _normalize_artist_key(artist_name)
    cache[key] = similar_artists
    save_similar_cache(cache)


# ============================================================
# STATS
# ============================================================

def get_lastfm_cache_stats() -> dict:
    """Get cache statistics."""
    tags_cache = load_tags_cache()
    similar_cache = load_similar_cache()
    return {
        "tags_cached": len(tags_cache),
        "similar_cached": len(similar_cache),
    }