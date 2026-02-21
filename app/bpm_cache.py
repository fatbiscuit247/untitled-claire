"""
BPM Cache - Cache BPM/key data from GetSongBPM API to reduce API calls.
Same pattern as lyrics_cache.py
"""

import json
import os
from pathlib import Path

CACHE_FILE = "bpm_cache.json"


def _load_cache() -> dict:
    if not Path(CACHE_FILE).exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save BPM cache: {e}")


def _make_key(song_name: str, artist_name: str) -> str:
    """Create a cache key from song + artist."""
    return f"{song_name.lower().strip()}|{artist_name.lower().strip()}"


def get_cached_bpm(song_name: str, artist_name: str) -> dict | None:
    """
    Get cached BPM data for a song.
    Returns dict with 'bpm' and 'key' if found, None otherwise.
    """
    cache = _load_cache()
    key = _make_key(song_name, artist_name)
    return cache.get(key)


def cache_bpm(song_name: str, artist_name: str, bpm: int | None, key: str | None):
    """
    Cache BPM data for a song.
    """
    cache = _load_cache()
    cache_key = _make_key(song_name, artist_name)
    cache[cache_key] = {
        "bpm": bpm,
        "key": key,
    }
    _save_cache(cache)


def get_bpm_cache_stats() -> dict:
    """Return stats about the cache."""
    cache = _load_cache()
    return {
        "total_cached": len(cache),
        "with_bpm": sum(1 for v in cache.values() if v.get("bpm")),
        "with_key": sum(1 for v in cache.values() if v.get("key")),
    }