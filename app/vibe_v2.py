"""
Vibe V2: Multi-signal "more like this" recommendations

Signals combined:
1. Audio features (tempo, energy, valence, acousticness, etc.)
2. Genre overlap (Jaccard similarity)
3. Era proximity (release year)
4. Key/mode compatibility
5. Related artist graph
6. Playlist co-occurrence (implicit via candidate sourcing)

Future: popularity filtering for niche discovery
"""

from collections import defaultdict, Counter
from typing import Optional
import re


# ============================================================
# CONFIGURATION (tune these)
# ============================================================

WEIGHTS = {
    "audio_features": 0.35,   # core sonic similarity
    "genre": 0.25,            # stylistic match
    "era": 0.15,              # temporal proximity
    "key_mode": 0.10,         # harmonic compatibility
    "related_artist": 0.10,   # graph proximity
    "popularity": 0.05,       # slight boost for known-good tracks
}

# Audio feature weights within the audio similarity calculation
AUDIO_FEATURE_WEIGHTS = {
    "energy": 1.5,
    "valence": 1.3,
    "danceability": 1.0,
    "acousticness": 1.2,
    "instrumentalness": 0.8,
    "speechiness": 0.6,
    "liveness": 0.4,
    "tempo": 1.0,  # normalized
}

# Filters
MAX_ERA_DISTANCE = 15  # years - candidates outside this get penalized heavily
TEMPO_TOLERANCE = 25   # BPM - candidates outside this get penalized


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_audio_features(seed_af: dict, candidate_af: dict) -> Optional[float]:
    """
    Compare audio features using weighted Euclidean distance.
    Returns similarity score 0-1 (higher = more similar).
    """
    if not seed_af or not candidate_af:
        return None
    
    def safe_float(v):
        try:
            return float(v) if v is not None else None
        except:
            return None
    
    # Normalize tempo to 0-1 range (assume 60-180 BPM range)
    def norm_tempo(t):
        t = safe_float(t)
        if t is None:
            return None
        return max(0, min(1, (t - 60) / 120))
    
    features = [
        ("energy", safe_float(seed_af.get("energy")), safe_float(candidate_af.get("energy"))),
        ("valence", safe_float(seed_af.get("valence")), safe_float(candidate_af.get("valence"))),
        ("danceability", safe_float(seed_af.get("danceability")), safe_float(candidate_af.get("danceability"))),
        ("acousticness", safe_float(seed_af.get("acousticness")), safe_float(candidate_af.get("acousticness"))),
        ("instrumentalness", safe_float(seed_af.get("instrumentalness")), safe_float(candidate_af.get("instrumentalness"))),
        ("speechiness", safe_float(seed_af.get("speechiness")), safe_float(candidate_af.get("speechiness"))),
        ("liveness", safe_float(seed_af.get("liveness")), safe_float(candidate_af.get("liveness"))),
        ("tempo", norm_tempo(seed_af.get("tempo")), norm_tempo(candidate_af.get("tempo"))),
    ]
    
    weighted_sq_diff = 0.0
    total_weight = 0.0
    
    for name, seed_val, cand_val in features:
        if seed_val is None or cand_val is None:
            continue
        w = AUDIO_FEATURE_WEIGHTS.get(name, 1.0)
        diff = seed_val - cand_val
        weighted_sq_diff += w * (diff ** 2)
        total_weight += w
    
    if total_weight == 0:
        return None
    
    # Convert distance to similarity (0-1)
    distance = (weighted_sq_diff / total_weight) ** 0.5
    similarity = 1.0 / (1.0 + distance * 3)  # scale factor for spread
    
    return similarity


def score_genre_overlap(seed_genres: list[str], candidate_genres: list[str]) -> float:
    """
    Jaccard similarity between genre sets.
    Returns 0-1.
    """
    if not seed_genres or not candidate_genres:
        return 0.3  # neutral score if unknown
    
    # Normalize genres
    seed_set = set(g.lower().strip() for g in seed_genres)
    cand_set = set(g.lower().strip() for g in candidate_genres)
    
    if not seed_set or not cand_set:
        return 0.3
    
    intersection = len(seed_set & cand_set)
    union = len(seed_set | cand_set)
    
    jaccard = intersection / union if union > 0 else 0
    
    # Also check for partial matches (e.g., "classic rock" vs "rock")
    partial_bonus = 0
    for sg in seed_set:
        for cg in cand_set:
            if sg in cg or cg in sg:
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
    
    if diff <= 3:
        return 1.0
    elif diff <= 7:
        return 0.8
    elif diff <= MAX_ERA_DISTANCE:
        return 0.6
    elif diff <= 25:
        return 0.3
    else:
        return 0.1


def score_key_compatibility(seed_af: dict, candidate_af: dict) -> float:
    """
    Score based on musical key/mode compatibility.
    Same key = 1.0, relative/parallel keys = 0.8, circle of fifths neighbors = 0.6
    Returns 0-1.
    """
    if not seed_af or not candidate_af:
        return 0.5  # neutral if unknown
    
    seed_key = seed_af.get("key")  # 0-11 (C, C#, D, ...)
    seed_mode = seed_af.get("mode")  # 0=minor, 1=major
    cand_key = candidate_af.get("key")
    cand_mode = candidate_af.get("mode")
    
    if seed_key is None or cand_key is None:
        return 0.5
    
    # Exact match
    if seed_key == cand_key and seed_mode == cand_mode:
        return 1.0
    
    # Same key, different mode (parallel major/minor)
    if seed_key == cand_key:
        return 0.85
    
    # Relative major/minor (3 semitones apart, opposite mode)
    if seed_mode != cand_mode:
        if seed_mode == 0:  # seed is minor
            relative_major = (seed_key + 3) % 12
            if cand_key == relative_major:
                return 0.85
        else:  # seed is major
            relative_minor = (seed_key - 3) % 12
            if cand_key == relative_minor:
                return 0.85
    
    # Circle of fifths neighbors (±1 fifth = ±7 semitones)
    fifth_up = (seed_key + 7) % 12
    fifth_down = (seed_key - 7) % 12
    if cand_key in (fifth_up, fifth_down):
        return 0.7
    
    # Same mode, different key
    if seed_mode == cand_mode:
        return 0.4
    
    return 0.3


def score_related_artist(candidate_artist_ids: list[str], related_artist_ids: set[str], seed_artist_ids: set[str]) -> float:
    """
    Score based on artist graph proximity.
    Returns 0-1.
    """
    if not candidate_artist_ids:
        return 0.3
    
    cand_set = set(candidate_artist_ids)
    
    # Same artist (penalize slightly - we want discovery)
    if cand_set & seed_artist_ids:
        return 0.4  # not zero, but not great
    
    # Related artist
    if cand_set & related_artist_ids:
        return 1.0
    
    return 0.3  # unknown relationship


def compute_composite_score(
    seed_af: dict,
    seed_genres: list[str],
    seed_year: Optional[int],
    seed_artist_ids: set[str],
    related_artist_ids: set[str],
    candidate: dict,
    candidate_af: dict,
    candidate_genres: list[str],
) -> dict:
    """
    Compute the final composite score combining all signals.
    Returns dict with component scores and final score.
    """
    
    # Parse candidate year
    cand_year = None
    release_date = candidate.get("release_date")
    if release_date:
        try:
            cand_year = int(release_date[:4])
        except:
            pass
    
    # Compute individual scores
    audio_score = score_audio_features(seed_af, candidate_af)
    genre_score = score_genre_overlap(seed_genres, candidate_genres)
    era_score = score_era_proximity(seed_year, cand_year)
    key_score = score_key_compatibility(seed_af, candidate_af)
    related_score = score_related_artist(
        candidate.get("artist_ids", []),
        related_artist_ids,
        seed_artist_ids
    )
    
    # Popularity as a minor signal (normalized to 0-1)
    popularity = (candidate.get("popularity") or 0) / 100.0
    
    # Compute weighted sum
    components = {
        "audio_features": audio_score,
        "genre": genre_score,
        "era": era_score,
        "key_mode": key_score,
        "related_artist": related_score,
        "popularity": popularity,
    }
    
    final_score = 0.0
    total_weight = 0.0
    
    for signal, score in components.items():
        if score is not None:
            w = WEIGHTS.get(signal, 0)
            final_score += w * score
            total_weight += w
    
    # Normalize if some signals were missing
    if total_weight > 0 and total_weight < 1.0:
        final_score = final_score / total_weight
    
    return {
        "final_score": round(final_score, 4),
        "components": {k: round(v, 3) if v is not None else None for k, v in components.items()},
    }


# ============================================================
# CANDIDATE SOURCING (improved)
# ============================================================

def build_genre_search_queries(genres: list[str], seed_name: str, seed_artist: str) -> list[str]:
    """
    Build targeted search queries based on genres.
    """
    queries = []
    
    # Genre-based queries
    for g in genres[:3]:
        queries.append(f"{g} essentials")
        queries.append(f"best {g}")
    
    # Artist similarity queries
    if seed_artist:
        queries.append(f"artists like {seed_artist}")
        queries.append(f"{seed_artist} similar")
    
    # Track similarity queries
    if seed_name:
        queries.append(f"songs like {seed_name}")
    
    return queries[:8]  # cap total queries


# ============================================================
# HELPER: Parse release year
# ============================================================

def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except:
        return None