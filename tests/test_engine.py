import io
import os
import sys
import unittest
import tempfile
import argparse
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from db import Database, season_from_date
from nhl_api import (
    NHLClient,
    nhl_now,
    nhl_today,
    current_season_start_year,
    compute_goalie_factor,
    TEAM_SLUG_MAP
)
from model import (
    PerformancePredictor,
    calculate_rrs_from_stats,
    calculate_xrrs_projection
)
from main import (
    parse_iso_date_arg,
    parse_history_arg,
    name_key,
    parse_booster_val,
    normalize_booster_name,
    resolve_booster,
    get_player_booster,
    load_boosters_file,
    save_boosters_file,
    sync_boosters_to_db,
    collect_slate_skaters,
    is_slate_started,
    build_skater_projection,
    match_injury,
    solve_optimal_lineup,
    solve_stacked_lineup,
    assign_slots,
    settle_past_lineups,
    print_quant_banner,
    _warned_ambiguous_names
)


class TestTimezoneAndDates(unittest.TestCase):
    def test_nhl_now_and_today(self):
        now = nhl_now()
        self.assertIsNotNone(now.tzinfo)
        today = nhl_today()
        self.assertRegex(today, r"^\d{4}-\d{2}-\d{2}$")

    def test_current_season_start_year(self):
        start_year = current_season_start_year()
        now = nhl_now()
        expected = now.year if now.month >= 8 else now.year - 1
        self.assertEqual(start_year, expected)

    def test_season_from_date(self):
        self.assertEqual(season_from_date("2024-10-15"), "20242025")
        self.assertEqual(season_from_date("2025-01-10"), "20242025")
        self.assertEqual(season_from_date("2025-08-01"), "20252026")

    def test_parse_iso_date_arg(self):
        self.assertEqual(parse_iso_date_arg(""), "")
        self.assertEqual(parse_iso_date_arg("2026-03-15"), "2026-03-15")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_iso_date_arg("15-03-2026")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_iso_date_arg("invalid-date")

    def test_parse_history_arg_validation(self):
        # Point 17: Validate --history argument (all or YYYY-MM-DD)
        self.assertEqual(parse_history_arg("all"), "all")
        self.assertEqual(parse_history_arg("ALL"), "all")
        self.assertEqual(parse_history_arg(""), "all")
        self.assertEqual(parse_history_arg("2026-10-18"), "2026-10-18")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_history_arg("not-a-date")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_history_arg("18-10-2026")


class TestBoosterManagement(unittest.TestCase):
    def setUp(self):
        _warned_ambiguous_names.clear()

    def test_name_key_normalization(self):
        self.assertEqual(name_key("Tim Stützle"), "tim stutzle")
        self.assertEqual(name_key("  Alexis   Lafrenière  "), "alexis lafreniere")
        self.assertEqual(name_key("Cale Makar"), "cale makar")

    def test_parse_booster_val(self):
        self.assertAlmostEqual(parse_booster_val("+10%"), 0.10)
        self.assertAlmostEqual(parse_booster_val("10%"), 0.10)
        self.assertAlmostEqual(parse_booster_val("1.4"), 1.4)
        self.assertAlmostEqual(parse_booster_val("+1.5x"), 1.5)
        self.assertAlmostEqual(parse_booster_val("80%"), 0.80)
        self.assertAlmostEqual(parse_booster_val("0.0"), 0.0)
        self.assertAlmostEqual(parse_booster_val(""), 0.0)

    def test_parse_booster_val_rules(self):
        # % always divided by 100
        self.assertAlmostEqual(parse_booster_val("10%"), 0.10)
        self.assertAlmostEqual(parse_booster_val("+150%"), 1.50)
        # Multipliers <= 5.0 as-is
        self.assertAlmostEqual(parse_booster_val("0.8"), 0.8)
        self.assertAlmostEqual(parse_booster_val("1.4x"), 1.4)
        self.assertAlmostEqual(parse_booster_val("+0.3"), 0.3)
        # Number without x or % > 5.0 divided by 100 with warning
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val80 = parse_booster_val("80")
        self.assertAlmostEqual(val80, 0.80)
        self.assertIn("0.8", buf.getvalue())
        # Multiplier with x > 5.0 keeps multiplier with warning
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val8x = parse_booster_val("8x")
        self.assertAlmostEqual(val8x, 8.0)
        self.assertIn("large booster multiplier 8x", buf.getvalue())
        # Comma replaced with dot and .5 supported
        self.assertAlmostEqual(parse_booster_val("1,5"), 1.5)
        self.assertAlmostEqual(parse_booster_val("0,8"), 0.8)
        self.assertAlmostEqual(parse_booster_val(".5"), 0.5)
        # Warning for 10.5 contains 0.105
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val105 = parse_booster_val("10.5")
        self.assertAlmostEqual(val105, 0.105)
        self.assertIn("0.105", buf.getvalue())
        # Unparsable prints warning with line number and returns 0.0
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val_inv = parse_booster_val("abc", line_num=42)
        self.assertAlmostEqual(val_inv, 0.0)
        self.assertIn("Line 42", buf.getvalue())

    def test_normalize_booster_name(self):
        self.assertEqual(normalize_booster_name("Elias Pettersson (D)"), "elias pettersson (D)")
        self.assertEqual(normalize_booster_name("Elias Pettersson #25"), "elias pettersson (D)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (C)"), "elias pettersson (C)")
        self.assertEqual(normalize_booster_name("Elias Pettersson"), "elias pettersson (C)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (F)"), "elias pettersson (F)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (D, VAN)"), "elias pettersson (D, VAN)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (C, VAN)"), "elias pettersson (C, VAN)")
        self.assertEqual(normalize_booster_name("[DTD] Cale Makar (DEF)"), "cale makar (D)")
        self.assertEqual(normalize_booster_name("[DEPTH] Matt Rempe (F)"), "matt rempe (F)")
        self.assertEqual(normalize_booster_name("Tim Stützle (C)"), "tim stutzle (C)")
        self.assertEqual(normalize_booster_name("Leo Carlsson"), "leo carlsson")

    def test_normalize_booster_name_and_resolve_with_team(self):
        # Point 16: Support (POS, TEAM) for namesakes with same position
        self.assertEqual(normalize_booster_name("Sebastian Aho (D, NYI)"), "sebastian aho (D, NYI)")
        self.assertEqual(normalize_booster_name("Sebastian Aho (C, CAR)"), "sebastian aho (C, CAR)")

        boosters = {
            "sebastian aho (C, CAR)": 1.2,
            "sebastian aho (D, NYI)": 0.5
        }
        val_car, key_car = resolve_booster(boosters, "Sebastian Aho", pos="C", team="CAR", is_duplicate=True)
        self.assertEqual(val_car, 1.2)
        self.assertEqual(key_car, "sebastian aho (C, CAR)")

        val_nyi, key_nyi = resolve_booster(boosters, "Sebastian Aho", pos="D", team="NYI", is_duplicate=True)
        self.assertEqual(val_nyi, 0.5)
        self.assertEqual(key_nyi, "sebastian aho (D, NYI)")

    def test_resolve_booster_duplicate_and_non_duplicate(self):
        boosters = {
            "elias pettersson (D)": 0.0,
            "elias pettersson (C)": 1.4,
            "cale makar (D)": 0.5,
            "tim stutzle": 1.2
        }

        # Duplicate: Elias Pettersson D vs C
        val_d, key_d = resolve_booster(boosters, "Elias Pettersson", pos="D", is_duplicate=True)
        self.assertEqual(val_d, 0.0)
        self.assertEqual(key_d, "elias pettersson (D)")

        val_c, key_c = resolve_booster(boosters, "Elias Pettersson", pos="C", is_duplicate=True)
        self.assertEqual(val_c, 1.4)
        self.assertEqual(key_c, "elias pettersson (C)")

        # Non-duplicate: Tim Stützle bare name
        val_s, key_s = resolve_booster(boosters, "Tim Stützle", pos="C", is_duplicate=False)
        self.assertEqual(val_s, 1.2)
        self.assertEqual(key_s, "tim stutzle")

        # Fallback wrapper
        self.assertEqual(get_player_booster(boosters, "Tim Stützle"), 1.2)

    def test_resolve_booster_namesake_ambiguous_warning(self):
        # Point 5: Bare name must NOT be used for duplicate, warning should be output
        boosters = {
            "sebastian aho": 2.0
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val, key = resolve_booster(boosters, "Sebastian Aho", pos="C", team="CAR", is_duplicate=True)
        self.assertEqual(val, 0.0)
        self.assertIsNone(key)
        self.assertIn("Ambiguous booster key", buf.getvalue())
        self.assertIn("Sebastian Aho", buf.getvalue())

    def test_load_boosters_file(self):
        content = (
            "# Header comment\n"
            "Tim Stützle = 1.2  # note after value\n"
            "Elias Pettersson #25 = 0.0  # number in player name\n"
            "Cale Makar (DEF) : +20%\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            b_map = load_boosters_file(temp_path)
            self.assertIn("tim stutzle", b_map)
            self.assertAlmostEqual(b_map["tim stutzle"], 1.2)
            self.assertIn("elias pettersson (D)", b_map)
            self.assertAlmostEqual(b_map["elias pettersson (D)"], 0.0)
            self.assertIn("cale makar (D)", b_map)
            self.assertAlmostEqual(b_map["cale makar (D)"], 0.20)
        finally:
            os.remove(temp_path)

    def test_boosters_file_io_roundtrip(self):
        # Point 19: Full cycle write and read of boosters.txt with comments, active, DTD, and depth players
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            temp_path = tf.name

        try:
            boosters = {
                "elias pettersson (D)": 0.0,
                "elias pettersson (C)": 1.4,
                "nathan mackinnon": 1.2,
                "sam malinski": 0.0
            }
            active = [
                {"name": "Nathan MacKinnon", "team": "COL", "pos": "C", "xRRS": 8.0, "booster": 1.2},
                {"name": "Elias Pettersson", "team": "VAN", "pos": "C", "xRRS": 6.5, "booster": 1.4},
                {"name": "Elias Pettersson", "team": "VAN", "pos": "D", "xRRS": 3.0, "booster": 0.0}
            ]
            dtd = [
                {"name": "Artturi Lehkonen", "team": "COL", "pos": "L", "inj_detail": "Upper Body"}
            ]
            depth = [
                {"name": "Sam Malinski", "team": "COL", "pos": "D", "avg_toi": 12.5}
            ]

            save_boosters_file(boosters, active, dtd_players=dtd, depth_players=depth, path=temp_path)

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("COL — Colorado Avalanche", content)
            self.assertIn("VAN — Vancouver Canucks", content)
            self.assertIn("Nathan MacKinnon", content)
            self.assertIn("Elias Pettersson (C)", content)
            self.assertIn("Elias Pettersson (D)", content)
            self.assertIn("# [DTD] Artturi Lehkonen", content)
            self.assertIn("# [DEPTH] Sam Malinski", content)
            self.assertIn("12:30", content)

            loaded = load_boosters_file(temp_path)
            self.assertEqual(loaded.get("nathan mackinnon"), 1.2)
            self.assertEqual(loaded.get("elias pettersson (C)"), 1.4)
            self.assertEqual(loaded.get("elias pettersson (D)"), 0.0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestGoaliesAndInjuries(unittest.TestCase):
    def test_team_slug_map_utah(self):
        self.assertEqual(TEAM_SLUG_MAP.get("utah-mammoth"), "UTA")
        self.assertEqual(TEAM_SLUG_MAP.get("utah-hockey-club"), "UTA")

    def test_compute_goalie_factor(self):
        self.assertEqual(compute_goalie_factor(0.0), 1.0)
        self.assertGreaterEqual(compute_goalie_factor(0.880), 1.0)
        self.assertLessEqual(compute_goalie_factor(0.930), 1.0)

    def test_match_injury_espn_mapping(self):
        # Point 7: Matches ESPN injury with team mapping
        injuries = {
            ("Jack Hughes", "NJ"): {"name": "Jack Hughes", "team": "NJ", "status": "Out", "detail": "Upper Body"},
            ("Victor Hedman", "TB"): {"name": "Victor Hedman", "team": "TB", "status": "Day-To-Day", "detail": "Lower Body"}
        }
        inj = match_injury(injuries, "Jack Hughes", "NJD", is_duplicate=False)
        self.assertIsNotNone(inj)
        self.assertEqual(inj["status"], "Out")

        inj2 = match_injury(injuries, "Victor Hedman", "TBL", is_duplicate=False)
        self.assertIsNotNone(inj2)
        self.assertEqual(inj2["status"], "Day-To-Day")

    def test_goalie_factor_in_prediction(self):
        pred = PerformancePredictor()
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.5, 0.5, 3.0, 18.0]]
        pred.multi_model = mock_model
        pred.trained = True
        res_weak_goalie = pred.predict(
            game_logs=[{"toi": "18:00", "goals": 1, "assists": 1, "shots": 3, "gameDate": "2026-01-01"}],
            is_home=True,
            opponent_abbrev="MTL",
            game_date="2026-01-10",
            standings={"MTL": {"defense_factor": 1.0}},
            position="F",
            goalie_factor=1.20
        )
        res_normal_goalie = pred.predict(
            game_logs=[{"toi": "18:00", "goals": 1, "assists": 1, "shots": 3, "gameDate": "2026-01-01"}],
            is_home=True,
            opponent_abbrev="MTL",
            game_date="2026-01-10",
            standings={"MTL": {"defense_factor": 1.0}},
            position="F",
            goalie_factor=1.00
        )
        self.assertGreater(res_weak_goalie["xG"], res_normal_goalie["xG"])
        self.assertGreater(res_weak_goalie["xA"], res_normal_goalie["xA"])
        self.assertEqual(res_weak_goalie["xSOG"], res_normal_goalie["xSOG"])

    def test_prediction_train_logs_only(self):
        # Point 17: Verify build_skater_projection calls real function and ignores logs >= game_date
        slate_date = "2026-03-10"
        logs = [
            {"gameDate": "2026-03-01", "goals": 0, "assists": 0, "shots": 1, "toi": "15:00"},
            {"gameDate": "2026-03-05", "goals": 0, "assists": 0, "shots": 1, "toi": "15:00"},
            {"gameDate": "2026-03-10", "goals": 50, "assists": 50, "shots": 50, "toi": "25:00"}  # Same day match: MUST BE EXCLUDED
        ]
        skater = {
            "id": 1,
            "name": "Test Skater",
            "pos": "F",
            "opp": "MTL",
            "is_home": True,
            "goalie_factor": 1.0,
            "xRRS": 1.5,
            "base_pts": 1.5
        }
        standings = {"MTL": {"defense_factor": 1.0}}
        proj_xrrs, _ = build_skater_projection(skater, logs, standings, game_date=slate_date, today_str=slate_date)
        # Excludes 50-goal match, staying well under 10.0 xRRS
        self.assertLess(proj_xrrs, 10.0)

    def test_xrrs_projection_peripheral_weights(self):
        # Point 12: xsog * 0.25 + xblk * 0.50
        # Compare projection with varying shots and blocks to isolate peripheral weights
        base_proj = calculate_xrrs_projection(xg=0.2, xa=0.3, xsog=1.0, xblk=1.0)
        sog_proj = calculate_xrrs_projection(xg=0.2, xa=0.3, xsog=3.0, xblk=1.0)
        blk_proj = calculate_xrrs_projection(xg=0.2, xa=0.3, xsog=1.0, xblk=2.0)

        # Delta for +2 SOG = 2 * 0.25 = 0.50
        self.assertAlmostEqual(sog_proj - base_proj, 0.50, places=2)
        # Delta for +1 BLK = 1 * 0.50 = 0.50
        self.assertAlmostEqual(blk_proj - base_proj, 0.50, places=2)


class TestLineupOptimizer(unittest.TestCase):
    def test_solve_optimal_lineup_dp(self):
        cands = []
        for i in range(15):
            cands.append({
                "id": i + 1,
                "name": f"Player {i+1}",
                "team": f"T{i%4}",
                "pos": "F" if i % 3 != 0 else "D",
                "xRRS": round(2.0 + i * 0.3, 2),
                "base_pts": round(2.0 + i * 0.3, 2),
                "booster": 0.2 if i % 2 == 0 else 0.0
            })

        lineup, score = solve_optimal_lineup(cands)
        self.assertEqual(len(lineup), 5)
        self.assertGreater(score, 0.0)

        for idx, item in enumerate(lineup):
            self.assertEqual(item["slot"], idx + 1)
            self.assertEqual(item["slot_boost"], [2.0, 1.8, 1.6, 1.4, 1.2][idx])

    def test_solve_optimal_lineup_greedy_vs_naive(self):
        # Point 19: High-booster player with lower xRRS beats naive top 5
        cands = [
            {"id": 1, "name": "P1", "team": "T1", "pos": "F", "xRRS": 5.0, "base_pts": 5.0, "booster": 0.0},
            {"id": 2, "name": "P2", "team": "T2", "pos": "F", "xRRS": 4.8, "base_pts": 4.8, "booster": 0.0},
            {"id": 3, "name": "P3", "team": "T3", "pos": "F", "xRRS": 4.6, "base_pts": 4.6, "booster": 0.0},
            {"id": 4, "name": "P4", "team": "T4", "pos": "F", "xRRS": 4.4, "base_pts": 4.4, "booster": 0.0},
            {"id": 5, "name": "P5", "team": "T5", "pos": "F", "xRRS": 4.2, "base_pts": 4.2, "booster": 0.0},
            {"id": 6, "name": "P6", "team": "T6", "pos": "F", "xRRS": 4.0, "base_pts": 4.0, "booster": 3.0}
        ]
        naive_lineup, naive_score = assign_slots(cands[:5])
        opt_lineup, opt_score = solve_optimal_lineup(cands)

        # Optimal score should strictly outperform naive top-5
        self.assertGreater(opt_score, naive_score)
        opt_ids = [p["id"] for p in opt_lineup]
        self.assertIn(6, opt_ids)
        self.assertNotIn(5, opt_ids)

    def test_solve_optimal_lineup_empty_or_small(self):
        empty_lineup, score = solve_optimal_lineup([])
        self.assertEqual(empty_lineup, [])
        self.assertEqual(score, 0.0)

        small = [{
            "id": 1,
            "name": "Single Player",
            "team": "COL",
            "pos": "F",
            "xRRS": 4.0,
            "base_pts": 4.0,
            "booster": 0.5
        }]
        small_lineup, small_score = solve_optimal_lineup(small)
        self.assertEqual(len(small_lineup), 1)
        self.assertEqual(small_lineup[0]["slot"], 1)

    def test_solve_stacked_lineup(self):
        cands = [
            {"id": 1, "name": "MacKinnon", "team": "COL", "pos": "C", "xRRS": 5.5, "base_pts": 5.5, "booster": 0.5, "goalie_factor": 1.1},
            {"id": 2, "name": "Rantanen", "team": "COL", "pos": "R", "xRRS": 5.0, "base_pts": 5.0, "booster": 0.4, "goalie_factor": 1.1},
            {"id": 3, "name": "Makar", "team": "COL", "pos": "D", "xRRS": 4.8, "base_pts": 4.8, "booster": 0.3, "goalie_factor": 1.1},
            {"id": 4, "name": "McDavid", "team": "EDM", "pos": "C", "xRRS": 5.8, "base_pts": 5.8, "booster": 0.2, "goalie_factor": 1.0},
            {"id": 5, "name": "Draisaitl", "team": "EDM", "pos": "C", "xRRS": 5.2, "base_pts": 5.2, "booster": 0.2, "goalie_factor": 1.0},
            {"id": 6, "name": "Bouchard", "team": "EDM", "pos": "D", "xRRS": 4.2, "base_pts": 4.2, "booster": 0.1, "goalie_factor": 1.0},
        ]
        lineup, base_score, syn_score = solve_stacked_lineup(cands)
        self.assertEqual(len(lineup), 5)
        stacks = [p.get("stack") for p in lineup if p.get("stack")]
        self.assertTrue(any("Pair" in s or "x" in s for s in stacks))

    def test_solve_stacked_lineup_synergy_and_defensemen(self):
        # Point 13: Team with 2 defensemen as top options pairs top D with first forward
        cands = [
            {"id": 1, "name": "Makar", "team": "COL", "pos": "D", "xRRS": 6.0, "base_pts": 6.0, "booster": 0.0, "goalie_factor": 1.15},
            {"id": 2, "name": "Toews", "team": "COL", "pos": "D", "xRRS": 5.5, "base_pts": 5.5, "booster": 0.0, "goalie_factor": 1.15},
            {"id": 3, "name": "MacKinnon", "team": "COL", "pos": "C", "xRRS": 5.0, "base_pts": 5.0, "booster": 0.0, "goalie_factor": 1.15},
            {"id": 4, "name": "Kucherov", "team": "TBL", "pos": "R", "xRRS": 5.2, "base_pts": 5.2, "booster": 0.0, "goalie_factor": 1.0},
            {"id": 5, "name": "Point", "team": "TBL", "pos": "C", "xRRS": 5.0, "base_pts": 5.0, "booster": 0.0, "goalie_factor": 1.0},
            {"id": 6, "name": "Hedman", "team": "TBL", "pos": "D", "xRRS": 4.5, "base_pts": 4.5, "booster": 0.0, "goalie_factor": 1.0}
        ]
        lineup, base_score, syn_score = solve_stacked_lineup(cands)
        col_players = [p for p in lineup if p["team"] == "COL"]
        # Must pair Makar (top D) with MacKinnon (first forward), not drop COL stack
        col_names = [p["name"] for p in col_players]
        self.assertIn("Makar", col_names)
        self.assertIn("MacKinnon", col_names)

    def test_assign_slots_labels(self):
        players = [
            {"id": 1, "name": "P1", "team": "CAR", "pos": "F", "xRRS": 4.0, "booster": 0.0},
            {"id": 2, "name": "P2", "team": "CAR", "pos": "F", "xRRS": 3.8, "booster": 0.0},
            {"id": 3, "name": "P3", "team": "CAR", "pos": "D", "xRRS": 3.5, "booster": 0.0},
            {"id": 4, "name": "P4", "team": "NYR", "pos": "F", "xRRS": 3.0, "booster": 0.0},
            {"id": 5, "name": "P5", "team": "NYR", "pos": "F", "xRRS": 2.5, "booster": 0.0},
        ]
        assigned, score = assign_slots(players)
        self.assertEqual(assigned[0]["stack"], "[CAR x3]")
        self.assertEqual(assigned[3]["stack"], "[NYR Pair]")

    def test_print_quant_banner_output(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_quant_banner(mode="Sync Roster & Boosters", target_date="2026-10-17")
        out = buf.getvalue()
        self.assertIn("RRS ENGINE v2.2", out)
        self.assertIn("Mode: Sync Roster & Boosters", out)
        self.assertIn("Date: 2026-10-17", out)
        self.assertIn("____", out)

    def test_rookie_without_train_logs_fallback(self):
        # Point 17: Skater with no prior games calls real build_skater_projection
        skater = {
            "id": 99999,
            "name": "Rookie Phenom",
            "team": "SJS",
            "pos": "C",
            "xRRS": 3.45,
            "base_pts": 3.45,
            "booster": 0.0,
            "goalie_factor": 1.0,
            "is_home": True,
            "opp": "ANA",
            "opp_goalie": ""
        }
        xrrs, base = build_skater_projection(
            skater, logs=[], standings={}, game_date="2099-01-01", today_str="2099-01-01"
        )
        self.assertEqual(xrrs, 3.45)
        self.assertEqual(base, 3.45)

        # Point 15: For a past slate, rookie fallback does not leak future data
        xrrs_past, base_past = build_skater_projection(
            skater, logs=[], standings={}, game_date="2020-01-01", today_str="2026-10-18"
        )
        self.assertEqual(xrrs_past, 0.0)
        self.assertEqual(base_past, 0.0)


class TestDatabaseAndHistory(unittest.TestCase):
    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.db = Database(self.temp_db_path)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_get_connection_context_manager(self):
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            row = cur.fetchone()
            self.assertEqual(row[0], 1)

    def test_delete_boosters(self):
        self.db.set_booster("player_one", 1.0)
        self.db.set_booster("player_two", 1.5)
        self.db.delete_boosters(["player_one"])
        boosters = self.db.get_all_boosters()
        self.assertNotIn("player_one", boosters)
        self.assertIn("player_two", boosters)

    def test_save_lineup_history_atomic_rewrite(self):
        lineup = [
            {"id": 1, "name": "P1", "team": "COL", "opp": "vsEDM", "slot": 1, "slot_boost": 2.0, "total_boost": 2.2, "xRRS": 4.0, "score": 8.8, "booster": 0.2},
            {"id": 2, "name": "P2", "team": "COL", "opp": "vsEDM", "slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "xRRS": 3.5, "score": 6.3, "booster": 0.0},
            {"id": 3, "name": "P3", "team": "EDM", "opp": "@COL", "slot": 3, "slot_boost": 1.6, "total_boost": 1.6, "xRRS": 3.2, "score": 5.12, "booster": 0.0},
            {"id": 4, "name": "P4", "team": "EDM", "opp": "@COL", "slot": 4, "slot_boost": 1.4, "total_boost": 1.4, "xRRS": 3.0, "score": 4.2, "booster": 0.0},
            {"id": 5, "name": "P5", "team": "EDM", "opp": "@COL", "slot": 5, "slot_boost": 1.2, "total_boost": 1.2, "xRRS": 2.8, "score": 3.36, "booster": 0.0},
        ]
        self.db.save_lineup_history("2026-10-15", "standard", lineup)
        slates = self.db.get_history_slates()
        self.assertEqual(len(slates), 1)
        self.assertEqual(slates[0]["game_date"], "2026-10-15")
        self.assertAlmostEqual(slates[0]["total_proj_rrs"], 27.78, places=1)

        # Overwrite with updated lineup
        lineup[0]["score"] = 9.0
        self.db.save_lineup_history("2026-10-15", "standard", lineup)
        slates = self.db.get_history_slates()
        self.assertEqual(len(slates), 1)

    def test_calculate_rrs_from_stats(self):
        score_def = calculate_rrs_from_stats(goals=0, assists=0, shots=2, blocks=1)
        self.assertEqual(score_def, 1.0)

        score_goal = calculate_rrs_from_stats(goals=1, assists=0, shots=3, blocks=0)
        self.assertAlmostEqual(score_goal, 3.25, places=1)

    def test_settle_past_lineups_dnp(self):
        lineup = [
            {"id": 101, "name": "Scratched Player", "team": "BOS", "opp": "vsTOR", "slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "xRRS": 3.0, "score": 6.0, "booster": 0.0}
        ]
        self.db.save_player(101, "Scratched Player", "BOS", "F")
        self.db.save_lineup_history("2026-10-10", "standard", lineup)

        client = MagicMock()
        client.get_player_current_game_logs.return_value = [
            {"gameDate": "2026-10-12", "goals": 1, "assists": 0, "shots": 2, "gameId": 9999}
        ]

        settle_past_lineups(client, self.db)
        details = self.db.get_history_slate_details("2026-10-10")
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["settled"], 1)
        self.assertEqual(details[0]["actual_rrs"], 0.0)
        self.assertEqual(details[0]["actual_toi"], "DNP")

    def test_settle_past_lineups_merged_db_logs(self):
        # Point 2: Settle past match present in db.get_game_logs even if /now returns empty
        lineup = [
            {"id": 202, "name": "Veteran Player", "team": "DET", "opp": "vsCHI", "slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "xRRS": 4.0, "score": 8.0, "booster": 0.0}
        ]
        self.db.save_player(202, "Veteran Player", "DET", "F")
        self.db.save_lineup_history("2026-03-01", "standard", lineup)
        self.db.save_game_logs(202, [
            {"gameDate": "2026-03-01", "goals": 1, "assists": 1, "shots": 4, "gameId": 5555}
        ])

        client = MagicMock()
        client.get_player_current_game_logs.return_value = []
        client.get_boxscore_skaters.return_value = {
            202: {"goals": 1, "assists": 1, "sog": 4, "blockedShots": 0, "toi": "18:30"}
        }

        settle_past_lineups(client, self.db)
        details = self.db.get_history_slate_details("2026-03-01")
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["settled"], 1)
        self.assertAlmostEqual(details[0]["actual_goals"], 1.0)
        self.assertAlmostEqual(details[0]["actual_assists"], 1.0)
        self.assertNotEqual(details[0]["actual_toi"], "DNP")

    def test_settle_past_lineups_empty_boxscore_skips(self):
        # Point 3: If game has gameId but boxscore is empty and match < 3 days old, skip record until next run
        yesterday_str = (datetime.strptime(nhl_today(), "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        lineup = [
            {"id": 303, "name": "Waiting Player", "team": "TOR", "opp": "vsMTL", "slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "xRRS": 3.0, "score": 6.0, "booster": 0.0}
        ]
        self.db.save_player(303, "Waiting Player", "TOR", "F")
        self.db.save_lineup_history(yesterday_str, "standard", lineup)

        client = MagicMock()
        client.get_player_current_game_logs.return_value = [
            {"gameDate": yesterday_str, "goals": 1, "assists": 0, "shots": 2, "gameId": 8888}
        ]
        client.get_boxscore_skaters.return_value = {}  # Network error / protocol unavailable

        settle_past_lineups(client, self.db)
        details = self.db.get_history_slate_details(yesterday_str)
        self.assertEqual(len(details), 1)
        # Record must remain unsettled
        self.assertEqual(details[0]["settled"], 0)

    def test_settle_past_lineups_skips_invalid_record(self):
        # Point 18: Record without player_id is skipped safely without exception
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO lineup_history (game_date, mode, slot, player_id, player_name, team, opponent, slot_boost, total_boost, proj_xrrs, proj_rrs, booster, created_at)
                VALUES ('2026-02-01', 'standard', 1, NULL, 'Corrupt Row', 'TOR', 'vsMTL', 2.0, 2.0, 3.0, 6.0, 0.0, 1000.0)
            """)
        client = MagicMock()
        settle_past_lineups(client, self.db)

    def test_slate_started_time_utc(self):
        # Point 17: Call real is_slate_started function
        now = datetime.now(timezone.utc)
        past_time = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        future_time = (now + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

        started_games = [{"startTimeUTC": past_time, "gameState": "FUT"}]
        upcoming_games = [{"startTimeUTC": future_time, "gameState": "FUT"}]

        today_str = nhl_today()
        self.assertTrue(is_slate_started(started_games, game_date=today_str, today_str=today_str))
        self.assertFalse(is_slate_started(upcoming_games, game_date=today_str, today_str=today_str))

    def test_injuries_composite_pk_and_atomic_replace(self):
        # Point 7: Composite primary key (player_name, team) and atomic replace
        injuries = [
            {"name": "Sebastian Aho", "team": "CAR", "status": "Day-To-Day", "detail": "Illness"},
            {"name": "Sebastian Aho", "team": "NYI", "status": "Out", "detail": "Upper Body"}
        ]
        self.db.save_injuries(injuries)
        cached = self.db.get_injuries()
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[("Sebastian Aho", "CAR")]["status"], "Day-To-Day")
        self.assertEqual(cached[("Sebastian Aho", "NYI")]["status"], "Out")

        # Atomic replace: CAR player recovers and is omitted
        updated_injuries = [
            {"name": "Sebastian Aho", "team": "NYI", "status": "Out", "detail": "Upper Body"},
            {"name": "Connor McDavid", "team": "EDM", "status": "Day-To-Day", "detail": "Lower Body"}
        ]
        self.db.save_injuries(updated_injuries)
        cached2 = self.db.get_injuries()
        self.assertEqual(len(cached2), 2)
        self.assertNotIn(("Sebastian Aho", "CAR"), cached2)
        self.assertIn(("Connor McDavid", "EDM"), cached2)

    def test_goalies_cache_by_game_date(self):
        # Point 11: Goalies cached by game_date
        goalies_day1 = {
            "EDM": {"goalie_name": "Skinner", "status": "Confirmed", "save_pct": 0.910, "gaa": 2.60}
        }
        self.db.save_starting_goalies(goalies_day1, game_date="2026-10-18")

        # Query same date
        cached_today = self.db.get_starting_goalies(game_date="2026-10-18")
        self.assertIn("EDM", cached_today)
        self.assertEqual(cached_today["EDM"]["goalie_name"], "Skinner")

        # Query different date
        cached_tomorrow = self.db.get_starting_goalies(game_date="2026-10-19")
        self.assertIsNone(cached_tomorrow)


class TestNHLClient(unittest.TestCase):
    def setUp(self):
        # Point 19: Temp database so tests do not touch production DB
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.db = Database(self.temp_db_path)
        self.client = NHLClient(db=self.db)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_get_boxscore_skaters(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "gameState": "FINAL",
            "playerByGameStats": {
                "homeTeam": {
                    "forwards": [
                        {"playerId": 8478402, "goals": 1, "assists": 2, "sog": 4, "blockedShots": 1, "toi": "21:30"}
                    ],
                    "defense": [
                        {"playerId": 8480069, "goals": 0, "assists": 1, "sog": 2, "blockedShots": 3, "toi": "24:15"}
                    ]
                },
                "awayTeam": {
                    "forwards": [
                        {"playerId": 8477934, "goals": 2, "assists": 0, "sog": 5, "blockedShots": 0, "toi": "19:40"}
                    ],
                    "defense": []
                }
            }
        }
        with patch.object(self.client.session, "get", return_value=mock_response):
            skaters = self.client.get_boxscore_skaters(2024020100)
            self.assertEqual(len(skaters), 3)
            self.assertEqual(skaters[8478402]["goals"], 1)
            self.assertEqual(skaters[8478402]["assists"], 2)
            self.assertEqual(skaters[8478402]["sog"], 4)
            self.assertEqual(skaters[8478402]["blockedShots"], 1)
            self.assertEqual(skaters[8480069]["blockedShots"], 3)

    def test_get_boxscore_skaters_off_status(self):
        # Point 19: Protocol accessible when gameState is 'OFF'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "gameState": "OFF",
            "playerByGameStats": {
                "homeTeam": {
                    "forwards": [
                        {"playerId": 8478402, "goals": 2, "assists": 1, "sog": 5, "blockedShots": 0, "toi": "20:00"}
                    ],
                    "defense": []
                },
                "awayTeam": {"forwards": [], "defense": []}
            }
        }
        with patch.object(self.client.session, "get", return_value=mock_response):
            skaters = self.client.get_boxscore_skaters(2024020102)
            self.assertEqual(len(skaters), 1)
            self.assertEqual(skaters[8478402]["goals"], 2)

    def test_get_boxscore_skaters_not_final(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"gameState": "LIVE"}
        with patch.object(self.client.session, "get", return_value=mock_response):
            skaters = self.client.get_boxscore_skaters(2024020101)
            self.assertEqual(skaters, {})

    def test_standings_zero_games_played(self):
        # Point 4: Team with 0 GP has defense_factor=1.0 and is excluded from league average GA
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "standings": [
                {"teamAbbrev": {"default": "COL"}, "gamesPlayed": 10, "goalAgainst": 30},
                {"teamAbbrev": {"default": "EDM"}, "gamesPlayed": 10, "goalAgainst": 30},
                {"teamAbbrev": {"default": "UTA"}, "gamesPlayed": 0, "goalAgainst": 0}
            ]
        }
        with patch.object(self.client.session, "get", return_value=mock_response):
            metrics = self.client.get_standings(force=True)
            self.assertIn("UTA", metrics)
            self.assertEqual(metrics["UTA"]["defense_factor"], 1.0)
            self.assertEqual(metrics["UTA"]["ga_per_game"], 3.0)
            # League avg GA should only use COL and EDM: (30+30)/(10+10) = 3.0
            self.assertAlmostEqual(metrics["COL"]["defense_factor"], 1.0)

    def test_preseason_and_stale_cache_refresh(self):
        # Point 10: gameTypeId == 1 filtered out, and stale cached logs trigger season refresh
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "gameTypeId": 1,
            "gameLog": [{"gameDate": "2026-09-25", "goals": 1}]
        }
        with patch.object(self.client.session, "get", return_value=mock_resp):
            logs = self.client.get_player_current_game_logs(8478402)
            self.assertEqual(logs, [])

    def test_get_all_club_stats_roster_filter_and_deduplication(self):
        def mock_club_stats(team):
            if team == "BUF":
                return [
                    {"playerId": 8478402, "firstName": {"default": "Jack"}, "lastName": {"default": "Eichel"}},
                    {"playerId": 8478403, "firstName": {"default": "Tage"}, "lastName": {"default": "Thompson"}}
                ]
            elif team == "VGK":
                return [
                    {"playerId": 8478402, "firstName": {"default": "Jack"}, "lastName": {"default": "Eichel"}},
                    {"playerId": 8478404, "firstName": {"default": "Mark"}, "lastName": {"default": "Stone"}}
                ]
            return []

        def mock_roster(team):
            if team == "BUF":
                return [{"id": 8478403}]
            elif team == "VGK":
                return [{"id": 8478402}, {"id": 8478404}]
            return []

        with patch.object(self.client, "get_club_stats", side_effect=mock_club_stats), \
             patch.object(self.client, "get_team_roster", side_effect=mock_roster):
            results = self.client.get_all_club_stats(["BUF", "VGK"])
            buf_pids = [p["playerId"] for p in results.get("BUF", [])]
            vgk_pids = [p["playerId"] for p in results.get("VGK", [])]

            self.assertNotIn(8478402, buf_pids)
            self.assertIn(8478402, vgk_pids)


class TestQAThirdReview(unittest.TestCase):
    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.db = Database(self.temp_db_path)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    # Point 1: Tuple typing import in db.py
    def test_point_1_tuple_typing_import(self):
        import db as db_module
        import typing
        self.assertIn("Tuple", db_module.__dict__)
        hints = typing.get_type_hints(db_module.Database.get_injuries)
        self.assertIn("return", hints)

    # Point 2: parse_booster_val rules (>5 divided by 100, x kept, comma to dot, .5)
    def test_point_2_parse_booster_val(self):
        # Number without x or % > 5.0 divided by 100 with warning
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val80 = parse_booster_val("80")
        self.assertAlmostEqual(val80, 0.8)
        self.assertIn("0.8", buf.getvalue())

        # Multiplier with x kept
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val8x = parse_booster_val("8x")
        self.assertAlmostEqual(val8x, 8.0)
        self.assertIn("large booster multiplier 8x", buf.getvalue())

        # Comma replaced with dot and .5 supported
        self.assertAlmostEqual(parse_booster_val("1,5"), 1.5)
        self.assertAlmostEqual(parse_booster_val("0,8"), 0.8)
        self.assertAlmostEqual(parse_booster_val(".5"), 0.5)

        # 10.5 gives 0.105 in warning
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val105 = parse_booster_val("10.5")
        self.assertAlmostEqual(val105, 0.105)
        self.assertIn("0.105", buf.getvalue())

    # Point 3: Duplicate name with one Out player preserves position tag in boosters file
    def test_point_3_duplicate_name_with_out_status(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            client = MagicMock()
            client.get_all_club_stats.return_value = {
                "CAR": [{"playerId": 101, "firstName": {"default": "Sebastian"}, "lastName": {"default": "Aho"}, "positionCode": "F", "gamesPlayed": 40, "avgTimeOnIcePerGame": 1200}],
                "NYI": [{"playerId": 102, "firstName": {"default": "Sebastian"}, "lastName": {"default": "Aho"}, "positionCode": "D", "gamesPlayed": 40, "avgTimeOnIcePerGame": 1200}]
            }
            team_matchups = {"CAR": ("NYI", True), "NYI": ("CAR", False)}
            goalies = {"CAR": {}, "NYI": {}}
            injuries = {("Sebastian Aho", "NYI"): {"name": "Sebastian Aho", "team": "NYI", "status": "Out"}}
            boosters = {}
            active, dtd, depth, excluded, _, _ = collect_slate_skaters(
                client, self.db, team_matchups, goalies, injuries, boosters
            )
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["name"], "Sebastian Aho")
            self.assertTrue(active[0]["is_dup"])

            save_boosters_file(boosters, active, path=path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Sebastian Aho (F)", content)

            # Test that resolve_booster finds it
            test_boosters = {"sebastian aho (F)": 1.5}
            val, key = resolve_booster(test_boosters, "Sebastian Aho", pos="F", team="CAR", is_duplicate=True)
            self.assertEqual(val, 1.5)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point 4: Bayesian shrinkage for small games played
    def test_point_4_bayesian_shrinkage(self):
        pred = PerformancePredictor()
        # 1 match rookie (1 goal, 3 shots, 0 assists, 15 min TOI)
        logs = [{"goals": 1, "assists": 0, "shots": 3, "toi": "15:00", "gameDate": "2026-01-01"}]
        res_1g = pred.predict(logs, is_home=True, opponent_abbrev="MTL", game_date="2026-01-05", standings={"MTL": {"defense_factor": 1.0}}, position="F")
        # Shrunk from superstar 3.92 down to ~2.20
        self.assertLess(res_1g["xRRS"], 3.0)
        self.assertGreater(res_1g["xRRS"], 1.5)

        # 0 games rookie
        res_0g = pred.predict([], is_home=True, opponent_abbrev="MTL", game_date="2026-01-05", standings={"MTL": {"defense_factor": 1.0}}, position="F")
        self.assertGreater(res_0g["xRRS"], 1.5)

    # Point 5: sync_boosters_to_db does not delete DB if boosters empty
    def test_point_5_sync_boosters_to_db_empty_guard(self):
        self.db.set_booster("connor mcdavid", 1.5)
        sync_boosters_to_db(self.db, {})
        self.assertEqual(self.db.get_all_boosters().get("connor mcdavid"), 1.5)

    # Point 6: Goalie cache with composite PRIMARY KEY (team, game_date)
    def test_point_6_goalies_composite_pk(self):
        self.db.save_starting_goalies({"EDM": {"goalie_name": "Stuart Skinner", "save_pct": 0.910, "gaa": 2.50}}, game_date="2026-10-18")
        self.db.save_starting_goalies({"EDM": {"goalie_name": "Calvin Pickard", "save_pct": 0.900, "gaa": 2.80}}, game_date="2026-10-19")
        g18 = self.db.get_starting_goalies(game_date="2026-10-18")
        g19 = self.db.get_starting_goalies(game_date="2026-10-19")
        self.assertIsNotNone(g18)
        self.assertIsNotNone(g19)
        self.assertEqual(g18["EDM"]["goalie_name"], "Stuart Skinner")
        self.assertEqual(g19["EDM"]["goalie_name"], "Calvin Pickard")

    # Point 7: Settle past lineups falls back to game log if match >= 3 days old and boxscore empty
    def test_point_7_settle_past_lineups_old_match_fallback(self):
        old_date = (datetime.strptime(nhl_today(), "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        lineup = [{"id": 505, "name": "Veteran Player", "team": "EDM", "opp": "vsCGY", "slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "xRRS": 3.0, "score": 6.0, "booster": 0.0}]
        self.db.save_player(505, "Veteran Player", "EDM", "F")
        self.db.save_lineup_history(old_date, "standard", lineup)
        client = MagicMock()
        client.get_player_current_game_logs.return_value = [
            {"gameDate": old_date, "goals": 2, "assists": 1, "shots": 4, "toi": "20:00", "gameId": 7777}
        ]
        client.get_boxscore_skaters.return_value = {}  # Empty boxscore
        settle_past_lineups(client, self.db)
        details = self.db.get_history_slate_details(old_date)
        self.assertEqual(details[0]["settled"], 1)
        self.assertEqual(details[0]["actual_goals"], 2.0)
        self.assertEqual(details[0]["actual_assists"], 1.0)

    # Point 8: get_player_game_logs loads missing seasons if latest log is before current season
    def test_point_8_get_player_game_logs_missing_seasons(self):
        self.db.save_game_logs(888, [{"gameId": 1, "gameDate": "2025-11-15", "goals": 1, "assists": 0, "shots": 2, "toi": "15:00", "seasonId": "20252026"}])
        client = NHLClient(db=self.db)
        fetched_seasons = []
        def mock_fetch(p_id, s):
            fetched_seasons.append(s)
            return [{"gameId": 2, "gameDate": "2026-01-15", "goals": 0, "assists": 1, "shots": 3, "toi": "16:00", "seasonId": s}]
        client._fetch_game_log = mock_fetch
        logs = client.get_player_game_logs(888)
        self.assertIn("20252026", fetched_seasons)
        self.assertGreater(len(logs), 1)

    # Point 9: normalize_booster_name respects specified position tag for Pettersson
    def test_point_9_normalize_booster_name_pettersson(self):
        self.assertEqual(normalize_booster_name("Elias Pettersson (D, VAN)"), "elias pettersson (D, VAN)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (C, VAN)"), "elias pettersson (C, VAN)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (D)"), "elias pettersson (D)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (F)"), "elias pettersson (F)")

    # Point 10: save_boosters_file formats boosters with :g
    def test_point_10_save_boosters_file_g_format(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            active = [{"id": 1, "name": "Connor McDavid", "team": "EDM", "pos": "C", "xRRS": 4.5}]
            boosters = {"connor mcdavid (C)": 0.25, "connor mcdavid": 0.25}
            save_boosters_file(boosters, active, path=path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("0.25", content)
            self.assertNotIn("0.2 ", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point 11: save_injuries INSERT OR REPLACE and clearing on empty ESPN response
    def test_point_11_save_injuries_replace_and_clear_on_empty(self):
        dupe_injuries = [
            {"name": "No Team Guy", "team": "", "status": "Out"},
            {"name": "No Team Guy", "team": "", "status": "Out"}
        ]
        self.db.save_injuries(dupe_injuries)
        self.assertIsNotNone(self.db.get_injuries())

        client = NHLClient(db=self.db)
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"injuries": []}
        client.session.get = MagicMock(return_value=mock_res)
        client.get_injuries(force=True)
        self.assertIsNone(self.db.get_injuries())

    # Point 12: Depth player TOI divmod formatting
    def test_point_12_depth_toi_divmod(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            depth = [{"name": "Depth Guy", "team": "ANA", "pos": "D", "avg_toi": 12.999}]
            save_boosters_file({}, [], depth_players=depth, path=path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("13:00", content)
            self.assertNotIn("12:60", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point 13: parse_iso_date_arg and parse_history_arg format normalization
    def test_point_13_date_normalization(self):
        self.assertEqual(parse_iso_date_arg("2026-10-1"), "2026-10-01")
        self.assertEqual(parse_history_arg("2026-10-1"), "2026-10-01")

    # Point 14: load_boosters_file hints on edited commented line
    def test_point_14_load_boosters_file_commented_hint(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# [DEPTH] Drew Helleson = 2.4 # ANA D 12:45\n")
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                boosters = load_boosters_file(path)
            self.assertEqual(boosters, {})
            self.assertIn("💡 Hint", buf.getvalue())
            self.assertIn("Remove '#' to activate", buf.getvalue())
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
