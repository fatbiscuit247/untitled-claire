"""
Vibe V3: "More like this" WITHOUT Spotify's restricted APIs

Works without:
- /audio-features (403 for new apps)
- /audio-analysis (403 for new apps)  
- /recommendations (deprecated)

Relies on:
- Genre overlap (from artist metadata - still works)
- Era/year proximity (from album metadata - still works)
- Playlist co-occurrence (search API - still works)
- Artist name similarity (search API - still works)
- Optional: local MFCC analysis for actual sonic similarity
"""

from collections import defaultdict, Counter
from typing import Optional
import re


# ============================================================
# CONFIGURATION - rebalanced for available signals
# ============================================================

WEIGHTS_V3 = {
    "genre": 0.40,            # primary signal now
    "era": 0.25,              # temporal proximity
    "playlist_source": 0.15,  # found in relevant playlists
    "popularity": 0.10,       # mild quality signal
    "name_similarity": 0.10,  # catches covers/versions, similar titles
}


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_genre_overlap(seed_genres: list[str], candidate_genres: list[str]) -> float:
    """
    Jaccard similarity between genre sets, with partial match bonus.
    Returns 0-1.
    """
    if not seed_genres or not candidate_genres:
        return 0.3  # neutral score if unknown
    
    # Normalize genres
    seed_set = set(g.lower().strip() for g in seed_genres)
    cand_set = set(g.lower().strip() for g in candidate_genres)
    
    if not seed_set or not cand_set:
        return 0.3
    
    # Exact Jaccard
    intersection = len(seed_set & cand_set)
    union = len(seed_set | cand_set)
    jaccard = intersection / union if union > 0 else 0
    
    # Partial match bonus (e.g., "classic rock" vs "rock")
    partial_bonus = 0
    for sg in seed_set:
        for cg in cand_set:
            # Check substring containment
            if sg != cg and (sg in cg or cg in sg):
                partial_bonus += 0.15
            # Check word overlap
            sg_words = set(sg.split())
            cg_words = set(cg.split())
            if sg_words & cg_words:
                partial_bonus += 0.1
    
    return min(1.0, jaccard + partial_bonus)


def score_era_proximity(seed_year: Optional[int], candidate_year: Optional[int]) -> float:
    """
    Score based on release year proximity.
    Returns 0-1.
    """
    if seed_year is None or candidate_year is None:
        return 0.5  # neutral if unknown
    
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


def score_playlist_source(source: str) -> float:
    """
    Score based on how the candidate was found.
    Playlist sources indicate human curation = higher confidence.
    """
    scores = {
        "genre_playlist": 1.0,      # found in genre-specific playlist
        "artist_playlist": 0.9,     # found in artist radio/similar playlist
        "song_playlist": 0.9,       # found in "songs like X" playlist
        "seed_album": 0.7,          # same album as seed
        "genre_search": 0.6,        # found via genre search
        "artist_catalog": 0.5,      # same artist's other work
    }
    return scores.get(source, 0.4)


def score_popularity(popularity: int) -> float:
    """
    Normalize popularity to 0-1.
    Slight curve to not over-reward mega-hits.
    """
    p = max(0, min(100, popularity or 0))
    # sqrt curve: 100 -> 1.0, 50 -> 0.71, 25 -> 0.5
    return (p / 100) ** 0.7


def score_name_similarity(seed_name: str, candidate_name: str) -> float:
    """
    Penalize if names are too similar (likely a cover/remix).
    Reward if completely different (true discovery).
    """
    def normalize(s):
        s = (s or "").lower()
        s = re.sub(r"\(.*?\)", "", s)  # remove (Remastered), (Live), etc.
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    seed_norm = normalize(seed_name)
    cand_norm = normalize(candidate_name)
    
    if not seed_norm or not cand_norm:
        return 0.5
    
    # Exact match = probably same song, skip
    if seed_norm == cand_norm:
        return 0.0
    
    # High overlap = might be cover/remix
    seed_words = set(seed_norm.split())
    cand_words = set(cand_norm.split())
    
    if seed_words and cand_words:
        overlap = len(seed_words & cand_words) / len(seed_words | cand_words)
        if overlap > 0.7:
            return 0.2  # penalize likely covers
        elif overlap > 0.4:
            return 0.5  # moderate similarity
    
    return 0.8  # different names = good discovery


def compute_composite_score_v3(
    seed_name: str,
    seed_genres: list[str],
    seed_year: Optional[int],
    seed_artist_ids: set[str],
    candidate: dict,
    candidate_genres: list[str],
) -> dict:
    """
    Compute the final composite score using available signals only.
    """
    
    # Parse candidate year
    cand_year = None
    release_date = candidate.get("release_date")
    if release_date:
        try:
            cand_year = int(release_date[:4])
        except:
            pass
    
    # Check if same artist (we'll penalize this)
    cand_artist_ids = set(candidate.get("artist_ids", []))
    is_same_artist = bool(cand_artist_ids & seed_artist_ids)
    
    # Compute individual scores
    genre_score = score_genre_overlap(seed_genres, candidate_genres)
    era_score = score_era_proximity(seed_year, cand_year)
    source_score = score_playlist_source(candidate.get("source", ""))
    pop_score = score_popularity(candidate.get("popularity", 0))
    name_score = score_name_similarity(seed_name, candidate.get("name", ""))
    
    components = {
        "genre": round(genre_score, 3),
        "era": round(era_score, 3),
        "playlist_source": round(source_score, 3),
        "popularity": round(pop_score, 3),
        "name_similarity": round(name_score, 3),
    }
    
    # Compute weighted sum
    final_score = 0.0
    for signal, score in components.items():
        w = WEIGHTS_V3.get(signal, 0)
        final_score += w * score
    
    # Same-artist penalty
    if is_same_artist:
        final_score *= 0.6  # 40% penalty
        components["same_artist_penalty"] = True
    
    return {
        "final_score": round(final_score, 4),
        "components": components,
    }


# ============================================================
# CANDIDATE SOURCING - optimized queries
# ============================================================

def build_search_queries_v3(
    seed_name: str,
    seed_artist: str, 
    seed_genres: list[str],
    seed_year: Optional[int]
) -> list[dict]:
    """
    Build targeted search queries with source labels.
    Returns list of {"query": str, "source": str, "type": "playlist"|"track"}
    """
    queries = []
    
    # Genre-based playlist searches (highest quality)
    for genre in seed_genres[:3]:
        queries.append({
            "query": f"{genre} essentials",
            "source": "genre_playlist",
            "type": "playlist"
        })
        queries.append({
            "query": f"best {genre} songs",
            "source": "genre_playlist", 
            "type": "playlist"
        })
    
    # Artist similarity playlists
    if seed_artist:
        queries.append({
            "query": f"{seed_artist} radio",
            "source": "artist_playlist",
            "type": "playlist"
        })
        queries.append({
            "query": f"artists like {seed_artist}",
            "source": "artist_playlist",
            "type": "playlist"
        })
    
    # Song similarity playlists
    if seed_name:
        queries.append({
            "query": f"songs like {seed_name}",
            "source": "song_playlist",
            "type": "playlist"
        })
    
    # Era + genre search (direct track search)
    if seed_year and seed_genres:
        decade = (seed_year // 10) * 10
        queries.append({
            "query": f"{seed_genres[0]} {decade}s",
            "source": "genre_search",
            "type": "track"
        })
    
    return queries


def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except:
        return None