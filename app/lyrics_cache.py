"""
Lyrics Caching for Vibe V5

Simple file-based cache to avoid re-fetching lyrics from Genius.
Stores lyrics in a JSON file keyed by "artist|song" (normalized).
"""

import json
import os
import re
from pathlib import Path

LYRICS_CACHE_PATH = "lyrics_cache.json"


def _normalize_cache_key(song_name: str, artist_name: str) -> str:
    """Create a normalized cache key from song and artist."""
    def clean(s):
        s = (s or "").lower().strip()
        s = re.sub(r"\(.*?\)", "", s)  # remove (Remastered), etc.
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    return f"{clean(artist_name)}|{clean(song_name)}"


def load_lyrics_cache() -> dict:
    """Load the lyrics cache from disk."""
    if not Path(LYRICS_CACHE_PATH).exists():
        return {}
    
    try:
        with open(LYRICS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading lyrics cache: {e}")
        return {}


def save_lyrics_cache(cache: dict) -> None:
    """Save the lyrics cache to disk."""
    try:
        with open(LYRICS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving lyrics cache: {e}")


def get_cached_lyrics(song_name: str, artist_name: str) -> str | None:
    """Get lyrics from cache if available."""
    cache = load_lyrics_cache()
    key = _normalize_cache_key(song_name, artist_name)
    return cache.get(key)


def cache_lyrics(song_name: str, artist_name: str, lyrics: str) -> None:
    """Store lyrics in cache."""
    cache = load_lyrics_cache()
    key = _normalize_cache_key(song_name, artist_name)
    cache[key] = lyrics
    save_lyrics_cache(cache)


def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache = load_lyrics_cache()
    return {
        "total_cached": len(cache),
        "cache_file": LYRICS_CACHE_PATH,
    }