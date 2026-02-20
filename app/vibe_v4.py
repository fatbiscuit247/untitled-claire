"""
Vibe V4: Playlist Co-occurrence + Future Lyrics Support

Core insight: If humans put songs together in playlists, they probably vibe similarly.

Strategy:
1. Search for playlists containing/related to the seed track
2. Extract other tracks from those playlists
3. Tracks appearing in MULTIPLE playlists = higher confidence
4. Score by co-occurrence frequency + era + popularity

Future: Add lyrics similarity via Genius/Musixmatch API
"""

from collections import defaultdict, Counter
from typing import Optional
import re


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHTS_V4 = {
    "cooccurrence": 0.45,     # appears in multiple playlists with seed
    "era": 0.20,              # temporal proximity
    "genre": 0.15,            # if available
    "popularity": 0.10,       # quality signal
    "name_similarity": 0.10,  # penalize covers/same song
    # Future:
    # "lyrics": 0.20,         # lyrical/thematic similarity
}


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_cooccurrence(count: int, max_count: int) -> float:
    """
    Score based on how many playlists contain both seed and candidate.
    More co-occurrences = stronger signal that songs "go together".
    """
    if max_count == 0:
        return 0.5
    
    # Normalize to 0-1, with diminishing returns after 3+ playlists
    normalized = min(count / 3.0, 1.0)
    
    # Bonus for appearing in multiple playlists
    if count >= 3:
        return 1.0
    elif count == 2:
        return 0.85
    elif count == 1:
        return 0.6
    else:
        return 0.3


def score_era_proximity(seed_year: Optional[int], candidate_year: Optional[int]) -> float:
    """Same as v3"""
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
    """Same as v3, but lower weight since we have co-occurrence now"""
    if not seed_genres or not candidate_genres:
        return 0.5  # neutral if unknown
    
    seed_set = set(g.lower().strip() for g in seed_genres)
    cand_set = set(g.lower().strip() for g in candidate_genres)
    
    if not seed_set or not cand_set:
        return 0.5
    
    intersection = len(seed_set & cand_set)
    union = len(seed_set | cand_set)
    jaccard = intersection / union if union > 0 else 0
    
    # Partial match bonus
    partial_bonus = 0
    for sg in seed_set:
        for cg in cand_set:
            if sg != cg and (sg in cg or cg in sg):
                partial_bonus += 0.1
    
    return min(1.0, jaccard + partial_bonus)


def score_popularity(popularity: int) -> float:
    """Same as v3"""
    p = max(0, min(100, popularity or 0))
    return (p / 100) ** 0.7


def score_name_similarity(seed_name: str, candidate_name: str) -> float:
    """Same as v3 - penalize likely covers/same song"""
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
        return 0.0  # same song
    
    seed_words = set(seed_norm.split())
    cand_words = set(cand_norm.split())
    
    if seed_words and cand_words:
        overlap = len(seed_words & cand_words) / len(seed_words | cand_words)
        if overlap > 0.7:
            return 0.2
        elif overlap > 0.4:
            return 0.5
    
    return 0.8


def compute_composite_score_v4(
    seed_name: str,
    seed_genres: list[str],
    seed_year: Optional[int],
    seed_artist_ids: set[str],
    candidate: dict,
    candidate_genres: list[str],
    cooccurrence_count: int,
    max_cooccurrence: int,
) -> dict:
    """
    Compute final score with co-occurrence as primary signal.
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
    
    # Compute scores
    cooccur_score = score_cooccurrence(cooccurrence_count, max_cooccurrence)
    era_score = score_era_proximity(seed_year, cand_year)
    genre_score = score_genre_overlap(seed_genres, candidate_genres)
    pop_score = score_popularity(candidate.get("popularity", 0))
    name_score = score_name_similarity(seed_name, candidate.get("name", ""))
    
    components = {
        "cooccurrence": round(cooccur_score, 3),
        "era": round(era_score, 3),
        "genre": round(genre_score, 3),
        "popularity": round(pop_score, 3),
        "name_similarity": round(name_score, 3),
        "playlist_count": cooccurrence_count,
    }
    
    # Weighted sum
    final_score = (
        WEIGHTS_V4["cooccurrence"] * cooccur_score +
        WEIGHTS_V4["era"] * era_score +
        WEIGHTS_V4["genre"] * genre_score +
        WEIGHTS_V4["popularity"] * pop_score +
        WEIGHTS_V4["name_similarity"] * name_score
    )
    
    # Same-artist penalty (we want discovery)
    if is_same_artist:
        final_score *= 0.5
        components["same_artist_penalty"] = True
    
    return {
        "final_score": round(final_score, 4),
        "components": components,
    }


# ============================================================
# PLAYLIST SEARCH QUERIES
# ============================================================

def build_playlist_queries_v4(seed_name: str, seed_artist: str) -> list[str]:
    """
    Build queries to find playlists likely containing the seed track.
    """
    queries = []
    
    # Direct song searches (most likely to contain the actual song)
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