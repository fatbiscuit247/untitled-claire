"""
Album Art Analysis Caching.

Caches analyzed album art data (brightness, saturation, warmth) to avoid
re-downloading and re-processing images.
"""

import json
import re
from pathlib import Path


ALBUM_ART_CACHE_PATH = "album_art_cache.json"


def _normalize_url_key(image_url: str) -> str:
    """Create a normalized cache key from image URL."""
    # Extract the unique part of the Spotify image URL
    # e.g., "https://i.scdn.co/image/ab67616d0000b273..." -> "ab67616d0000b273..."
    if not image_url:
        return ""
    match = re.search(r'image/([a-f0-9]+)', image_url)
    if match:
        return match.group(1)
    # Fallback: hash the whole URL
    return str(hash(image_url))


def load_album_art_cache() -> dict:
    """Load the album art cache from disk."""
    if not Path(ALBUM_ART_CACHE_PATH).exists():
        return {}
    try:
        with open(ALBUM_ART_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading album art cache: {e}")
        return {}


def save_album_art_cache(cache: dict) -> None:
    """Save the album art cache to disk."""
    try:
        with open(ALBUM_ART_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving album art cache: {e}")


def get_cached_album_art(image_url: str) -> dict | None:
    """Get cached album art analysis. Returns dict with brightness/saturation/warmth or None."""
    if not image_url:
        return None
    cache = load_album_art_cache()
    key = _normalize_url_key(image_url)
    return cache.get(key)


def cache_album_art(image_url: str, analysis: dict) -> None:
    """Store album art analysis in cache."""
    if not image_url:
        return
    cache = load_album_art_cache()
    key = _normalize_url_key(image_url)
    cache[key] = analysis
    save_album_art_cache(cache)


def get_album_art_cache_stats() -> dict:
    """Get cache statistics."""
    cache = load_album_art_cache()
    return {
        "images_cached": len(cache),
    }