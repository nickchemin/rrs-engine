"""
image_parser.py - Automatic Draft Screenshot OCR & Booster Parsing Module
Part of RRS Engine (nickchemin/rrs-engine)

Extracts player names and booster multipliers from Real Sports mobile app draft
screenshots and synchronizes them to boosters.txt and SQLite database.
"""

import os
import sys
import re
import shutil
import unicodedata
from typing import List, Dict, Tuple, Optional, Any, Set, Union, Callable

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = None
    ImageEnhance = None
    ImageFilter = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

import difflib

# Standard 32 NHL team abbreviations
NHL_TEAMS = {
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
}

# Common UI keywords from Real Sports app to ignore during name parsing
UI_IGNORE_WORDS = {
    "REAL", "SPORTS", "DRAFT", "BOOSTER", "BOOSTERS", "MULTIPLIER", "SLOT",
    "LOCK", "SALARY", "REMAINING", "FILTER", "SEARCH", "ROSTER", "AVG",
    "POINTS", "SCORE", "PROJECTED", "MATCHUP", "STATS", "GAME", "GAMES",
    "OPPONENT", "GOALIE", "STARTING", "CONFIRMED", "EXPECTED", "QUESTIONABLE",
    "DOUBTFUL", "OUT", "IR", "DTD", "INJURY", "INJURED", "TODAY", "YESTERDAY",
    "TOMORROW", "LINEUP", "LINEUPS", "SUBMIT", "CANCEL", "SAVE", "RESET",
    "CLEAR", "SELECT", "SELECTED", "UNSELECT", "SWAP", "REMOVE", "ADD",
    "POWER", "PLAY", "EVEN", "STRENGTH", "PENALTY", "KILL", "DEFENSE",
    "FORWARD", "CENTER", "WING", "LEFT", "RIGHT", "TIER", "TIERS"
}


# ==============================================================================
# OCR Backend Abstraction
# ==============================================================================

class OCRBackend:
    """Base class for image OCR extraction backends."""
    def extract_text(self, image_path: str) -> str:
        raise NotImplementedError


class TesseractOCR(OCRBackend):
    """
    Local OCR engine utilizing pytesseract and Google Tesseract-OCR binary.
    Automatically discovers standard Windows, macOS, and Linux binary paths.
    """
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or self._discover_tesseract_binary()
        if self.tesseract_cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    @staticmethod
    def _discover_tesseract_binary() -> Optional[str]:
        # 1. Check environment variable
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and os.path.isfile(env_cmd):
            return env_cmd

        # 2. Check PATH
        which_cmd = shutil.which("tesseract")
        if which_cmd:
            return which_cmd

        # 3. Check common installation directories
        candidate_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expanduser(r"~\scoop\apps\tesseract\current\tesseract.exe"),
            r"/usr/bin/tesseract",
            r"/usr/local/bin/tesseract",
            r"/opt/homebrew/bin/tesseract"
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                return p
        return None

    def is_available(self) -> bool:
        if not pytesseract:
            return False
        if not self.tesseract_cmd or not os.path.isfile(self.tesseract_cmd):
            return False
        return True

    def extract_text(self, image_path: str) -> str:
        if not pytesseract:
            raise RuntimeError("pytesseract is not installed. Install via: pip install pytesseract")
        if not self.is_available():
            raise RuntimeError(
                f"Tesseract binary not found at '{self.tesseract_cmd}'. "
                "Please install Tesseract-OCR or set the TESSERACT_CMD environment variable."
            )
        if not Image:
            raise RuntimeError("Pillow is not installed. Install via: pip install pillow")

        try:
            with Image.open(image_path) as img:
                # Preprocessing for mobile app dark theme: convert to grayscale and boost contrast
                gray = img.convert("L")
                if ImageEnhance:
                    enhancer = ImageEnhance.Contrast(gray)
                    gray = enhancer.enhance(1.8)
                text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
                if not text or len(text.strip()) < 10:
                    # Fallback to standard PSM 3 (fully automatic page segmentation)
                    text = pytesseract.image_to_string(gray, lang="eng", config="--psm 3")
                return text or ""
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed on image '{image_path}': {e}") from e


class MockOCR(OCRBackend):
    """
    Mock OCR engine for deterministic, offline unit testing.
    Can be configured with an explicit image-to-text mapping or a callable generator.
    """
    def __init__(
        self,
        text_mapping: Optional[Dict[str, str]] = None,
        default_text: str = "",
        generator: Optional[Callable[[str], str]] = None
    ):
        self.text_mapping = text_mapping or {}
        self.default_text = default_text
        self.generator = generator

    def extract_text(self, image_path: str) -> str:
        if self.generator:
            return self.generator(image_path)
        base = os.path.basename(image_path)
        if image_path in self.text_mapping:
            return self.text_mapping[image_path]
        if base in self.text_mapping:
            return self.text_mapping[base]
        return self.default_text


class VisionAPIOCR(OCRBackend):
    """
    HTTP-based Vision API OCR for cloud or local endpoints
    (configured via VISION_API_URL and optional VISION_API_KEY).
    """
    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.environ.get("VISION_API_URL")
        self.api_key = api_key or os.environ.get("VISION_API_KEY")

    def is_available(self) -> bool:
        return bool(self.endpoint_url)

    def extract_text(self, image_path: str) -> str:
        if not self.endpoint_url:
            raise RuntimeError("VISION_API_URL is not set.")
        import base64
        import requests

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            self.endpoint_url,
            json={"image_base64": b64_data, "filename": os.path.basename(image_path)},
            headers=headers,
            timeout=20.0
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text") or data.get("extracted_text") or ""


def get_default_ocr_backend() -> OCRBackend:
    """Returns the most appropriate available OCR backend."""
    tesseract = TesseractOCR()
    if tesseract.is_available():
        return tesseract
    vision = VisionAPIOCR()
    if vision.is_available():
        return vision
    # Return Tesseract backend which will raise a descriptive error when invoked
    return tesseract


# ==============================================================================
# Text Parsing & Booster Candidate Extraction
# ==============================================================================

def normalize_text_token(tok: str) -> str:
    """Strips accents, punctuation, and downcases."""
    if not tok:
        return ""
    nk = unicodedata.normalize("NFKD", str(tok))
    ascii_str = "".join(c for c in nk if not unicodedata.combining(c))
    return "".join(c for c in ascii_str.lower() if c.isalnum() or c in ("-", "'", "."))


def clean_line_text(line: str) -> str:
    """Cleans noisy non-ascii glyphs while preserving standard name and number characters."""
    line = line.replace("•", " - ").replace("|", " - ").replace("/", " - ")
    # Replace common OCR misreads for plus and x
    line = re.sub(r'[\u2010\u2013\u2014]', '-', line)
    return line.strip()


def parse_booster_multiplier_token(token: str) -> Optional[float]:
    """
    Parses a booster multiplier token such as '+0.4', '1.2x', '+20%', '0.8', '3.0'.
    Returns float value (e.g. 0.4 for +0.4 or +40%, 1.2 for 1.2x payout, 0.0 for 0).
    """
    token = token.strip().replace(",", ".")
    m = re.match(r'^\+?(\d+(?:\.\d+)?)\s*([x%])?$', token, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "%":
        return round(num / 100.0, 4)
    elif unit == "x":
        return num
    else:
        if num > 5.0:
            return round(num / 100.0, 4)
        return num


def is_name_like_token(s: str) -> bool:
    """Checks whether a token looks like part of a player name."""
    clean = re.sub(r"[.'\-]", "", s)
    if not clean.isalpha():
        return False
    if clean.upper() in UI_IGNORE_WORDS or clean.upper() in NHL_TEAMS:
        return False
    return len(clean) >= 2


def extract_booster_candidates_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses OCR raw text extracted from Real Sports mobile screenshots and returns
    a list of candidate player-booster dicts:
    [
        {
            "raw_name": "Leon Draisaitl",
            "booster_val": 0.4,
            "raw_booster": "+0.4",
            "team": "EDM",
            "pos": "C"
        },
        ...
    ]
    """
    candidates: List[Dict[str, Any]] = []
    lines = [clean_line_text(l) for l in raw_text.splitlines() if clean_line_text(l)]
    if not lines:
        return candidates

    # Regex patterns
    booster_re = re.compile(r'(?:booster:?\s*)?(?:\+)?(\d+(?:[.,]\d+)?)\s*([x%])?', re.IGNORECASE)
    team_pos_re = re.compile(
        r'\b([A-Z]{2,4})\b\s*[\-•\s]+\b(C|LW|RW|L|R|D|DEF|F|G)\b|\b(C|LW|RW|L|R|D|DEF|F|G)\b\s*[\-•\s]+\b([A-Z]{2,4})\b',
        re.IGNORECASE
    )

    # Strategy 1: Single-line format (e.g. "Leon Draisaitl +0.4" or "Matt Boldy = 0.8" or "Evan Bouchard (D) = 1.1")
    single_line_re = re.compile(
        r'^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.\'\- ]+?)\s*(?:\(([A-Za-z0-9, ]+)\))?\s*(?:[:=]|\s+)\s*(?:\+)?(\d+(?:[.,]\d+)?\s*[x%]?)\s*(?:#.*)?$',
        re.UNICODE
    )

    matched_line_indices: Set[int] = set()

    for idx, line in enumerate(lines):
        m = single_line_re.match(line)
        if m:
            cand_name = m.group(1).strip()
            paren_info = (m.group(2) or "").strip().upper()
            val_token = m.group(3).strip()
            b_val = parse_booster_multiplier_token(val_token)

            if b_val is not None:
                words = [w for w in cand_name.split() if is_name_like_token(w)]
                if 2 <= len(words) <= 4:
                    pos = None
                    team = None
                    if paren_info:
                        for p in ("DEF", "D", "C", "LW", "RW", "L", "R", "F"):
                            if p in paren_info.split():
                                pos = "D" if p in ("DEF", "D") else ("F" if p in ("F", "C", "LW", "RW", "L", "R") else p)
                                break
                        for t in NHL_TEAMS:
                            if t in paren_info:
                                team = t
                                break

                    candidates.append({
                        "raw_name": " ".join(words),
                        "booster_val": b_val,
                        "raw_booster": val_token,
                        "team": team,
                        "pos": pos,
                        "line_idx": idx
                    })
                    matched_line_indices.add(idx)

    # Strategy 2: Multi-line card layout (Common in Real Sports draft screen)
    # Line i: Player Name (e.g. "Connor McDavid")
    # Line i+1: Team & Pos (e.g. "EDM - C") or secondary info
    # Line i+2 (or i+1): Booster (e.g. "+0.4")
    for i in range(len(lines)):
        if i in matched_line_indices:
            continue
        line = lines[i]

        words = [w for w in line.split() if is_name_like_token(w)]
        if not (2 <= len(words) <= 4):
            continue

        cand_name = " ".join(words)
        # Look ahead up to 3 lines for team/pos and booster
        cand_team = None
        cand_pos = None
        cand_booster = None
        cand_booster_raw = None

        lookahead_max = min(i + 4, len(lines))
        for j in range(i + 1, lookahead_max):
            if j in matched_line_indices:
                continue
            sub_line = lines[j]

            # Check for team and position
            tp_match = team_pos_re.search(sub_line)
            if tp_match:
                t1, p1, p2, t2 = tp_match.groups()
                cand_team = (t1 or t2 or "").upper()
                if cand_team not in NHL_TEAMS:
                    cand_team = None
                p_raw = (p1 or p2 or "").upper()
                if p_raw in ("D", "DEF"):
                    cand_pos = "D"
                elif p_raw in ("C", "LW", "RW", "L", "R", "F"):
                    cand_pos = "F" if p_raw in ("L", "R", "F") else p_raw

            # Check for booster
            b_tokens = sub_line.split()
            for tok in b_tokens:
                val = parse_booster_multiplier_token(tok)
                if val is not None:
                    cand_booster = val
                    cand_booster_raw = tok
                    break

            if cand_booster is not None:
                break

        if cand_booster is not None:
            candidates.append({
                "raw_name": cand_name,
                "booster_val": cand_booster,
                "raw_booster": cand_booster_raw or f"{cand_booster:g}",
                "team": cand_team,
                "pos": cand_pos,
                "line_idx": i
            })

    return candidates


# ==============================================================================
# Fuzzy Matching & Namesake Disambiguation
# ==============================================================================

def compute_name_similarity(name_a: str, name_b: str) -> float:
    """Computes similarity percentage (0.0 to 100.0) between two player names."""
    norm_a = normalize_text_token(name_a)
    norm_b = normalize_text_token(name_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 100.0

    if fuzz:
        token_sort = fuzz.token_sort_ratio(norm_a, norm_b)
        ratio = fuzz.ratio(norm_a, norm_b)
        return max(token_sort, ratio)
    else:
        return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() * 100.0


def match_player_candidate(
    candidate: Dict[str, Any],
    known_players: List[Dict[str, Any]],
    threshold: float = 72.0
) -> Optional[Tuple[Dict[str, Any], str, float]]:
    """
    Fuzzy matches a parsed candidate against known slate players.
    Returns:
        (matched_player_dict, canonical_booster_key, confidence_score)
        or None if no confident match was found.
    """
    raw_name = candidate.get("raw_name", "")
    cand_team = (candidate.get("team") or "").upper()
    cand_pos = (candidate.get("pos") or "").upper()

    best_match: Optional[Dict[str, Any]] = None
    best_score: float = 0.0

    for p in known_players:
        p_name = p.get("name", "")
        p_team = (p.get("team") or "").upper()
        p_pos = (p.get("pos") or "").upper()

        score = compute_name_similarity(raw_name, p_name)

        # Team compatibility check
        if cand_team and p_team:
            if cand_team == p_team:
                score += 8.0  # Team confirmation bonus
            else:
                score -= 25.0  # Mismatched team penalty

        # Position compatibility check
        if cand_pos and p_pos:
            cand_group = "D" if cand_pos == "D" else "F"
            p_group = "D" if p_pos == "D" else "F"
            if cand_group == p_group:
                score += 5.0
            else:
                score -= 20.0

        if score > best_score:
            best_score = score
            best_match = p

    if not best_match or best_score < threshold:
        return None

    # Resolve canonical booster key with namesake disambiguation
    matched_name = best_match.get("name", "")
    is_duplicate = best_match.get("is_duplicate", False)
    matched_pos = best_match.get("pos", "")
    matched_team = best_match.get("team", "")

    # Check for known namesakes (e.g. Elias Pettersson, Sebastian Aho)
    from main import name_key, normalize_booster_name
    base_key = name_key(matched_name)

    # Disambiguate if duplicate or known namesakes
    if is_duplicate or "elias pettersson" in base_key or "sebastian aho" in base_key:
        if not cand_pos:
            print(f"⚠️ Warning: Ambiguous OCR match '{raw_name}' for duplicate player '{matched_name}' missing position in image. Skipping to prevent misattribution.")
            return None
        # Disambiguate using explicit candidate position and unified normalize_booster_name
        canonical_key = normalize_booster_name(f"{matched_name} ({cand_pos})")
    else:
        canonical_key = base_key

    return best_match, canonical_key, min(100.0, round(best_score, 1))


# ==============================================================================
# Screenshot Directory Batch Processing
# ==============================================================================

def scan_image_files(folder_path: str) -> List[str]:
    """Finds all supported image files in the target directory."""
    supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if not os.path.isdir(folder_path):
        return []
    files = []
    for f in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_exts:
            files.append(os.path.join(folder_path, f))
    return files


def parse_images_directory(
    folder_path: str,
    slate_players: List[Dict[str, Any]],
    ocr_backend: Optional[OCRBackend] = None,
    threshold: float = 72.0
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Processes all screenshot images in folder_path, extracting and fuzzy matching
    booster multipliers for active slate players.
    Returns:
        (detected_boosters_dict, detailed_match_records)
    """
    backend = ocr_backend or get_default_ocr_backend()
    image_files = scan_image_files(folder_path)

    detected_boosters: Dict[str, float] = {}
    detailed_matches: List[Dict[str, Any]] = []

    if not image_files:
        print(f"⚠️ Warning: No image files found in '{folder_path}'.")
        return detected_boosters, detailed_matches

    print(f"\nScanning {len(image_files)} draft screenshot(s) from: {folder_path}")

    # Track matches per canonical key to keep the highest confidence detection
    best_candidate_map: Dict[str, Dict[str, Any]] = {}

    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        sys.stdout.write(f"\rProcessing [{idx}/{len(image_files)}]: {filename}...")
        sys.stdout.flush()

        try:
            raw_text = backend.extract_text(img_path)
        except Exception as e:
            print(f"\n⚠️ Warning: Failed to extract text from '{filename}': {e}")
            continue

        candidates = extract_booster_candidates_from_text(raw_text)

        for cand in candidates:
            match_res = match_player_candidate(cand, slate_players, threshold=threshold)
            if not match_res:
                continue

            matched_player, canonical_key, confidence = match_res
            booster_val = cand["booster_val"]

            record = {
                "player_id": matched_player.get("id") or matched_player.get("player_id"),
                "name": matched_player.get("name"),
                "team": matched_player.get("team"),
                "pos": matched_player.get("pos"),
                "booster": booster_val,
                "canonical_key": canonical_key,
                "confidence": confidence,
                "source_image": filename
            }

            if canonical_key not in best_candidate_map or confidence > best_candidate_map[canonical_key]["confidence"]:
                best_candidate_map[canonical_key] = record

    sys.stdout.write(f"\rProcessing completed for {len(image_files)} screenshot(s).                  \n")
    sys.stdout.flush()

    for k, rec in best_candidate_map.items():
        detected_boosters[k] = rec["booster"]
        detailed_matches.append(rec)

    # Sort detailed matches alphabetically by player name
    detailed_matches.sort(key=lambda x: x["name"])
    return detected_boosters, detailed_matches


def print_ocr_summary_table(matches: List[Dict[str, Any]]):
    """Prints a structured summary table of extracted player boosters."""
    if not matches:
        print("\nNo player boosters matched from the provided screenshots.")
        return

    print("\n" + "=" * 80)
    print("REAL SPORTS OCR EXTRACTION RESULTS")
    print("=" * 80)
    header = f"{'Player Name':<26} {'Team':<6} {'Pos':<5} {'Booster':<10} {'Conf':<8} {'Source Image':<20}"
    print(header)
    print("-" * 80)

    for m in matches:
        b_str = f"+{m['booster']:g}" if m['booster'] > 0 else f"{m['booster']:g}"
        conf_str = f"{m['confidence']:.0f}%"
        src = m['source_image']
        if len(src) > 20:
            src = src[:17] + "..."
        print(f"{m['name']:<26} {m['team']:<6} {m['pos']:<5} {b_str:<10} {conf_str:<8} {src:<20}")

    print("-" * 80)
    print(f"Total matched players: {len(matches)}")
    print("=" * 80 + "\n")


# ==============================================================================
# CLI Orchestrator Entrypoint
# ==============================================================================

def handle_parse_images(
    folder_path: str,
    client: Any,
    db: Any,
    target_date: str = "",
    boosters_path: Optional[str] = None,
    ocr_backend: Optional[OCRBackend] = None
) -> Dict[str, float]:
    """
    Main orchestrator for --parse-images <path_to_folder> CLI flag.
    Scans screenshots, matches players to game-day roster, merges into boosters.txt,
    and synchronizes updated values to SQLite database.
    """
    from main import (
        BOOSTERS_FILE,
        load_boosters_file,
        save_boosters_file,
        sync_boosters_to_db,
        collect_slate_skaters,
        nhl_today
    )

    target_boosters_file = boosters_path or BOOSTERS_FILE

    if not os.path.exists(folder_path):
        print(f"❌ Error: Directory '{folder_path}' does not exist.")
        return {}

    image_files = scan_image_files(folder_path)
    if not image_files:
        print(f"❌ Error: No supported images (.png, .jpg, .webp) found in '{folder_path}'.")
        return {}

    # 1. Fetch upcoming schedule to resolve target date and matchups
    schedule = client.get_upcoming_schedule()
    if not schedule:
        print("Schedule unavailable; querying historical players from database.")
        all_players = db.get_all_players()
        slate_skaters = [
            {"name": p["name"], "team": p["team"], "pos": p["pos"], "id": p["player_id"], "is_duplicate": False}
            for p in all_players
        ]
        active_skaters = slate_skaters
        dtd_skaters = []
        depth_skaters = []
    else:
        if not target_date:
            today_str = nhl_today()
            target_day = next((d for d in schedule if d["date"] == today_str and d.get("games")), None)
            if not target_day:
                target_day = next((d for d in schedule if d["date"] >= today_str and d.get("games")), None)
        else:
            target_day = next((d for d in schedule if d["date"] == target_date and d.get("games")), None)

        if not target_day:
            print(f"No games found for date: {target_date or 'today'}. Using cached players.")
            all_players = db.get_all_players()
            slate_skaters = [
                {"name": p["name"], "team": p["team"], "pos": p["pos"], "id": p["player_id"], "is_duplicate": False}
                for p in all_players
            ]
            active_skaters = slate_skaters
            dtd_skaters = []
            depth_skaters = []
        else:
            game_date = target_day["date"]
            games = target_day.get("games", [])
            team_matchups = {}
            for g in games:
                home, away = g["homeTeam"], g["awayTeam"]
                team_matchups[home] = (away, True)
                team_matchups[away] = (home, False)

            injuries = client.get_injuries()
            goalies = client.get_starting_goalies(games=games)
            team_lines = client.prefetch_teams_lines(list(team_matchups.keys()))

            existing_boosters = load_boosters_file(target_boosters_file) if os.path.exists(target_boosters_file) else {}

            active_skaters, dtd_skaters, depth_skaters, _, _, _ = collect_slate_skaters(
                client, db, team_matchups, goalies, injuries, existing_boosters,
                team_lines=team_lines, game_date=game_date
            )
            slate_skaters = list(active_skaters) + list(dtd_skaters) + list(depth_skaters)

    # 2. Run OCR & fuzzy matching
    new_boosters, matches = parse_images_directory(
        folder_path, slate_skaters, ocr_backend=ocr_backend
    )

    if not new_boosters:
        print("No valid boosters were detected from the images.")
        return {}

    # Print formatted output table
    print_ocr_summary_table(matches)

    # 3. Load existing boosters from file, merge new ones, and persist
    current_boosters = load_boosters_file(target_boosters_file) if os.path.exists(target_boosters_file) else {}
    updated_count = 0
    added_count = 0

    for k, val in new_boosters.items():
        if k in current_boosters:
            if current_boosters[k] != val:
                current_boosters[k] = val
                updated_count += 1
        else:
            current_boosters[k] = val
            added_count += 1

    # Save to boosters.txt preserving team partition format
    if active_skaters:
        save_boosters_file(
            current_boosters,
            active_skaters,
            dtd_players=dtd_skaters,
            depth_players=depth_skaters,
            path=target_boosters_file
        )
    else:
        # Fallback append / write directly
        with open(target_boosters_file, "w", encoding="utf-8") as f:
            f.write("# Real Sports NHL - Player Boosters (OCR Export)\n")
            for k, val in sorted(current_boosters.items()):
                f.write(f"{k} = {val:g}\n")

    # Sync to SQLite database
    sync_boosters_to_db(db, current_boosters)

    print(
        f"✓ Successfully processed {len(matches)} boosters: "
        f"{updated_count} updated, {added_count} newly added into {os.path.basename(target_boosters_file)} "
        f"and synchronized to SQLite database."
    )
    return new_boosters
