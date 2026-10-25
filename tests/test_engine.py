import io
import os
import sys
import shutil
import unittest
import tempfile
import argparse
import requests
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import time
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from db import Database, season_from_date, TeamLines
from nhl_api import (
    NHLClient,
    nhl_now,
    nhl_today,
    current_season_start_year,
    compute_goalie_factor,
    TEAM_SLUG_MAP,
    USEasternTz
)
from model import (
    PerformancePredictor,
    calculate_rrs_from_stats,
    calculate_xrrs_projection,
    compute_line_factor,
    calibrate_rrs_formula,
    evaluate_model
)
import optimizer
from optimizer import (
    compute_pair_correlation,
    build_lineup_correlation_matrix,
    simulate_lineup_distribution
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
    is_out_status,
    is_dtd_status,
    solve_optimal_lineup,
    solve_stacked_lineup,
    compute_pair_synergy,
    sync_game_day,
    show_history_report,
    assign_slots,
    settle_past_lineups,
    print_quant_banner,
    print_lineup_table,
    handle_record_app_scores,
    _warned_ambiguous_names,
    main as main_entry
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
        self.assertIn("RRS ENGINE v2.5", out)
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
    def test_injury_cache_survives_empty_upstream_response(self):
        dupe_injuries = [
            {"name": "No Team Guy", "team": "", "status": "Out"},
            {"name": "No Team Guy", "team": "", "status": "Out"}
        ]
        self.db.save_injuries(dupe_injuries)
        self.assertEqual(len(self.db.get_injuries()), 1)

        self.db.save_injuries([])
        self.assertEqual(len(self.db.get_injuries()), 1)

        client = NHLClient(db=self.db)
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"injuries": []}
        client.session.get = MagicMock(return_value=mock_res)
        client.get_injuries(force=True)
        self.assertEqual(len(self.db.get_injuries()), 1)

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


class TestTeamLinesAndDeployment(unittest.TestCase):
    def setUp(self):
        self.fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd)
        self.db = Database(db_path=self.db_path)
        self.client = NHLClient(db=self.db)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_db_save_and_get_team_lines(self):
        lines_data = {
            "connor mcdavid": {"ev_line": "L1", "pp_unit": "PP1", "pk_unit": None, "position": "C"},
            "evan bouchard": {"ev_line": "D1", "pp_unit": "PP1", "pk_unit": "PK2", "position": "RD"}
        }
        self.db.save_team_lines("EDM", lines_data)

        fetched = self.db.get_team_lines("EDM")
        self.assertIsNotNone(fetched)
        self.assertEqual(len(fetched), 2)
        self.assertEqual(fetched["connor mcdavid"]["ev_line"], "L1")
        self.assertEqual(fetched["connor mcdavid"]["pp_unit"], "PP1")
        self.assertEqual(fetched["evan bouchard"]["ev_line"], "D1")

        # Batch lookup
        batch = self.db.get_all_lines_for_teams(["EDM", "NYR"])
        self.assertIn("EDM", batch)
        self.assertEqual(batch["EDM"]["connor mcdavid"]["pp_unit"], "PP1")
        self.assertNotIn("NYR", batch)

        # Expired check
        expired = self.db.get_team_lines("EDM", max_age_hours=0.0)
        self.assertIsNone(expired)

    def test_nhl_client_get_team_lines_dfo_parsing(self):
        mock_html = """
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">
        {
            "props": {
                "pageProps": {
                    "combinations": {
                        "players": [
                            {"name": "Connor McDavid", "categoryIdentifier": "ev", "groupIdentifier": "f1", "positionIdentifier": "c"},
                            {"name": "Connor McDavid", "categoryIdentifier": "pp", "groupIdentifier": "pp1", "positionIdentifier": "c"},
                            {"name": "Evan Bouchard", "categoryIdentifier": "ev", "groupIdentifier": "d1", "positionIdentifier": "rd"},
                            {"name": "Evan Bouchard", "categoryIdentifier": "pp", "groupIdentifier": "pp1", "positionIdentifier": "rd"},
                            {"name": "Ryan Shea", "categoryIdentifier": "ev", "groupIdentifier": "d2", "positionIdentifier": "ld"}
                        ]
                    }
                }
            }
        }
        </script>
        </body></html>
        """
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.text = mock_html
        with patch.object(self.client.session, "get", return_value=mock_res):
            res = self.client.get_team_lines("EDM", force=True)
            self.assertIn("connor mcdavid", res)
            self.assertEqual(res["connor mcdavid"]["ev_line"], "L1")
            self.assertEqual(res["connor mcdavid"]["pp_unit"], "PP1")
            self.assertEqual(res["evan bouchard"]["ev_line"], "D1")
            self.assertEqual(res["evan bouchard"]["pp_unit"], "PP1")
            self.assertEqual(res["ryan shea"]["ev_line"], "D2")
            self.assertIsNone(res["ryan shea"]["pp_unit"])

    def test_compute_line_factor_multipliers(self):
        # PP1 + L1 forward: elevated goal, assist, and TOI
        g_mult, a_mult, toi_adj = compute_line_factor("L1", "PP1", "F")
        self.assertGreater(g_mult, 1.10)
        self.assertGreater(a_mult, 1.15)
        self.assertGreater(toi_adj, 1.5)

        # L4 forward without PP: reduced multipliers
        g_mult4, a_mult4, toi_adj4 = compute_line_factor("L4", None, "F")
        self.assertLess(g_mult4, 0.90)
        self.assertLess(a_mult4, 0.85)
        self.assertLess(toi_adj4, 0.0)

        # No line data: clean 1.0, 1.0, 0.0 neutral fallback
        g_none, a_none, toi_none = compute_line_factor(None, None, "F")
        self.assertEqual((g_none, a_none, toi_none), (1.0, 1.0, 0.0))

        # D1 defense with PP1: elevated assists and TOI
        g_d, a_d, toi_d = compute_line_factor("D1", "PP1", "D")
        self.assertGreater(a_d, 1.15)
        self.assertGreaterEqual(toi_d, 2.5)

    def test_predict_with_ev_line_and_pp_unit(self):
        predictor = PerformancePredictor()
        standings = {"CGY": {"defense_factor": 1.0}}

        pred_top = predictor.predict(
            game_logs=[],
            is_home=True,
            opponent_abbrev="CGY",
            game_date="2026-10-18",
            standings=standings,
            position="F",
            ev_line="L1",
            pp_unit="PP1"
        )

        pred_bot = predictor.predict(
            game_logs=[],
            is_home=True,
            opponent_abbrev="CGY",
            game_date="2026-10-18",
            standings=standings,
            position="F",
            ev_line="L4",
            pp_unit=None
        )

        self.assertGreater(pred_top["xG"], pred_bot["xG"])
        self.assertGreater(pred_top["xA"], pred_bot["xA"])
        self.assertGreater(pred_top["xRRS"], pred_bot["xRRS"])
        self.assertEqual(pred_top["ev_line"], "L1")
        self.assertEqual(pred_top["pp_unit"], "PP1")

    def test_solve_stacked_lineup_pp1_synergy_and_label(self):
        candidates = [
            {"id": 1, "name": "Connor McDavid", "team": "EDM", "pos": "F", "xRRS": 9.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 2, "name": "Leon Draisaitl", "team": "EDM", "pos": "F", "xRRS": 8.5, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 3, "name": "Nathan MacKinnon", "team": "COL", "pos": "F", "xRRS": 9.2, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 4, "name": "Cale Makar", "team": "COL", "pos": "D", "xRRS": 8.8, "booster": 0.0, "ev_line": "D1", "pp_unit": "PP1"},
            {"id": 5, "name": "Kirill Kaprizov", "team": "MIN", "pos": "F", "xRRS": 8.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 6, "name": "Sidney Crosby", "team": "PIT", "pos": "F", "xRRS": 7.5, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"}
        ]
        lineup, base_score, syn_score = solve_stacked_lineup(candidates)
        self.assertEqual(len(lineup), 5)
        self.assertGreater(syn_score, base_score)
        stacks = [p["stack"] for p in lineup if p.get("stack")]
        self.assertTrue(any("PP1" in s for s in stacks))

    def test_print_lineup_table_with_line_column(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "name": "Connor McDavid", "team": "EDM", "opp": "vsCGY", "opp_goalie": "D. Wolf", "xRRS": 9.5, "booster": 0.0, "score": 19.0, "line_tag": "L1/PP1", "stack": "[EDM PP1]"},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "name": "Leon Draisaitl", "team": "EDM", "opp": "vsCGY", "opp_goalie": "D. Wolf", "xRRS": 8.5, "booster": 0.0, "score": 15.3, "line_tag": "L1/PP1", "stack": "[EDM PP1]"},
            {"slot": 3, "slot_boost": 1.6, "total_boost": 1.6, "name": "Cale Makar", "team": "COL", "opp": "@PHI", "opp_goalie": "S. Ersson", "xRRS": 8.0, "booster": 0.0, "score": 12.8, "line_tag": "D1/PP1", "stack": ""},
            {"slot": 4, "slot_boost": 1.4, "total_boost": 1.4, "name": "Kirill Kaprizov", "team": "MIN", "opp": "vsCBJ", "opp_goalie": "E. Merzlikins", "xRRS": 7.5, "booster": 0.0, "score": 10.5, "line_tag": "L1/PP1", "stack": ""},
            {"slot": 5, "slot_boost": 1.2, "total_boost": 1.2, "name": "Sidney Crosby", "team": "PIT", "opp": "@UTA", "opp_goalie": "C. Ingram", "xRRS": 7.0, "booster": 0.0, "score": 8.4, "line_tag": "L1/PP1", "stack": ""}
        ]
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_lineup_table("2026-10-18", lineup, 66.0, title="Tournament Stack", synergy_score=72.5)
        out = buf.getvalue()
        self.assertIn("Line", out)
        self.assertIn("L1/PP1", out)
        self.assertIn("D1/PP1", out)
        self.assertIn("[EDM PP1]", out)


class TestQAFourthReview(unittest.TestCase):
    def setUp(self):
        self.fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd)
        self.db = Database(db_path=self.db_path)
        self.client = NHLClient(db=self.db)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    # Point A1: Namesake lines disambiguation in same team (Vancouver's Elias Pettersson C vs D)
    def test_a1_namesakes_lines_disambiguation(self):
        mock_html = """
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">
        {
            "props": {
                "pageProps": {
                    "combinations": {
                        "players": [
                            {"name": "Elias Pettersson", "positionIdentifier": "c", "categoryIdentifier": "ev", "groupIdentifier": "f1"},
                            {"name": "Elias Pettersson", "positionIdentifier": "c", "categoryIdentifier": "pp", "groupIdentifier": "pp1"},
                            {"name": "Elias Pettersson", "positionIdentifier": "ld", "categoryIdentifier": "ev", "groupIdentifier": "d3"}
                        ]
                    }
                }
            }
        }
        </script>
        </body></html>
        """
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.text = mock_html
        with patch.object(self.client.session, "get", return_value=mock_res):
            lines = self.client.get_team_lines("VAN", force=True)

        self.assertIn(("elias pettersson", "F"), lines)
        self.assertIn(("elias pettersson", "D"), lines)
        self.assertEqual(lines.get(("elias pettersson", "F"))["ev_line"], "L1")
        self.assertEqual(lines.get(("elias pettersson", "F"))["pp_unit"], "PP1")
        self.assertEqual(lines.get(("elias pettersson", "D"))["ev_line"], "D3")
        self.assertIsNone(lines.get(("elias pettersson", "D"))["pp_unit"])

        # Save to DB and retrieve
        self.db.save_team_lines("VAN", lines)
        db_lines = self.db.get_team_lines("VAN")
        self.assertEqual(db_lines.get(("elias pettersson", "F"))["ev_line"], "L1")
        self.assertEqual(db_lines.get(("elias pettersson", "D"))["ev_line"], "D3")

        # In collect_slate_skaters, forward gets L1/PP1, defenseman gets D3
        mock_skaters = [
            {"playerId": 101, "firstName": {"default": "Elias"}, "lastName": {"default": "Pettersson"}, "positionCode": "C", "gamesPlayed": 20, "goals": 10, "assists": 15, "shots": 50, "blockedShots": 10, "avgTimeOnIcePerGame": 19.5 * 60},
            {"playerId": 102, "firstName": {"default": "Elias"}, "lastName": {"default": "Pettersson"}, "positionCode": "D", "gamesPlayed": 20, "goals": 1, "assists": 2, "shots": 15, "blockedShots": 25, "avgTimeOnIcePerGame": 16.0 * 60}
        ]
        with patch.object(self.client, "get_all_club_stats", return_value={"VAN": mock_skaters}):
            active, _, _, _, _, _ = collect_slate_skaters(
                self.client, self.db,
                team_matchups={"VAN": ("CGY", True)},
                goalies={}, injuries={}, boosters={},
                team_lines={"VAN": lines},
                game_date="2026-10-18",
                today_str="2026-10-18"
            )
        f_skater = next(s for s in active if s["id"] == 101)
        d_skater = next(s for s in active if s["id"] == 102)
        self.assertEqual(f_skater["ev_line"], "L1")
        self.assertEqual(f_skater["pp_unit"], "PP1")
        self.assertEqual(d_skater["ev_line"], "D3")
        self.assertGreater(f_skater["xRRS"], d_skater["xRRS"])
        self.assertGreater(f_skater["xRRS"], 4.0)
        self.assertLess(d_skater["xRRS"], 2.0)

    # Point A2: No-PP penalty guard when team has no PP records
    def test_a2_no_pp_penalty_guard(self):
        # Without PP data for team (team_has_pp=False): no penalty
        g_m, a_m, toi_d = compute_line_factor("L1", None, "F", team_has_pp=False)
        self.assertAlmostEqual(g_m, 1.05, places=3)
        self.assertAlmostEqual(a_m, 1.05, places=3)
        self.assertAlmostEqual(toi_d, 1.0, places=3)

        # With PP data present for team (team_has_pp=True): penalty applied
        g_p, a_p, toi_p = compute_line_factor("L1", None, "F", team_has_pp=True)
        self.assertLess(g_p, 1.0)
        self.assertLess(a_p, 1.0)
        self.assertAlmostEqual(toi_p, 0.5, places=3)

        # PerformancePredictor passes team_has_pp correctly
        predictor = PerformancePredictor()
        standings = {"CGY": {"defense_factor": 1.0}}
        pred_guard = predictor.predict(
            game_logs=[], is_home=True, opponent_abbrev="CGY", game_date="2026-10-18",
            standings=standings, position="F", ev_line="L1", pp_unit=None, team_has_pp=False
        )
        pred_penalized = predictor.predict(
            game_logs=[], is_home=True, opponent_abbrev="CGY", game_date="2026-10-18",
            standings=standings, position="F", ev_line="L1", pp_unit=None, team_has_pp=True
        )
        self.assertGreater(pred_guard["xG"], pred_penalized["xG"])
        self.assertGreater(pred_guard["xA"], pred_penalized["xA"])

    # Point A3: boosters.txt protection on empty stats and partial team stats
    def test_a3_save_boosters_file_protection(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            initial_text = "# Existing boosters file\nConnor McDavid = 2.0\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(initial_text)

            # 1. Empty candidates does NOT overwrite
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                save_boosters_file({}, [], path=path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, initial_text)
            self.assertIn("boosters.txt was not modified", buf.getvalue())

            # 2. Partial team sync preserves missing teams
            existing_with_teams = (
                "# ==============================================================================\n"
                "# EDM — Edmonton Oilers\n"
                "# ==============================================================================\n"
                "Connor McDavid = 2.0\n\n"
                "# ==============================================================================\n"
                "# TOR — Toronto Maple Leafs\n"
                "# ==============================================================================\n"
                "Auston Matthews = 1.8\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(existing_with_teams)

            edm_only_candidates = [
                {"name": "Connor McDavid", "team": "EDM", "pos": "F", "xRRS": 9.5, "avg_toi": 21.5, "booster": 2.0}
            ]
            save_boosters_file({}, edm_only_candidates, path=path)
            with open(path, "r", encoding="utf-8") as f:
                new_content = f.read()
            self.assertIn("EDM — Edmonton Oilers", new_content)
            self.assertIn("TOR — Toronto Maple Leafs", new_content)
            self.assertIn("Auston Matthews = 1.8", new_content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point A4: Tournament stack upside >= standard DP optimal lineup base score
    def test_a4_stack_vs_standard_ev_guarantee(self):
        # Scenario 1: Players from 5 completely different teams (no stacks possible)
        candidates_diff_teams = [
            {"id": 1, "name": "Connor McDavid", "team": "EDM", "pos": "F", "xRRS": 9.5, "booster": 0.0},
            {"id": 2, "name": "Nathan MacKinnon", "team": "COL", "pos": "F", "xRRS": 9.2, "booster": 0.0},
            {"id": 3, "name": "Kirill Kaprizov", "team": "MIN", "pos": "F", "xRRS": 8.5, "booster": 0.0},
            {"id": 4, "name": "Nikita Kucherov", "team": "TBL", "pos": "F", "xRRS": 8.2, "booster": 0.0},
            {"id": 5, "name": "Auston Matthews", "team": "TOR", "pos": "F", "xRRS": 8.0, "booster": 0.0},
            {"id": 6, "name": "Sidney Crosby", "team": "PIT", "pos": "F", "xRRS": 7.5, "booster": 0.0},
        ]
        dp_lineup, dp_base = solve_optimal_lineup(candidates_diff_teams)
        st_lineup, st_base, st_upside = solve_stacked_lineup(candidates_diff_teams)
        self.assertGreaterEqual(st_upside, dp_base)

        # Scenario 2: Candidates with low-correlation teammates vs strong individuals
        candidates_mixed = [
            {"id": 1, "name": "Connor McDavid", "team": "EDM", "pos": "F", "xRRS": 9.5, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 2, "name": "Nathan MacKinnon", "team": "COL", "pos": "F", "xRRS": 9.2, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 3, "name": "Kirill Kaprizov", "team": "MIN", "pos": "F", "xRRS": 8.5, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 4, "name": "Nikita Kucherov", "team": "TBL", "pos": "F", "xRRS": 8.2, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 5, "name": "Auston Matthews", "team": "TOR", "pos": "F", "xRRS": 8.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 6, "name": "Corey Perry", "team": "EDM", "pos": "F", "xRRS": 2.0, "booster": 0.0, "ev_line": "L4", "pp_unit": None},
        ]
        dp_l2, dp_b2 = solve_optimal_lineup(candidates_mixed)
        st_l2, st_b2, st_u2 = solve_stacked_lineup(candidates_mixed)
        self.assertGreaterEqual(st_u2, dp_b2)

    # Point B5: save_team_lines purges stale records before insert
    def test_b5_stale_lines_cleanup(self):
        initial_lines = {
            "old player": {"ev_line": "L1", "pp_unit": "PP1", "position": "C"}
        }
        self.db.save_team_lines("EDM", initial_lines)
        self.assertIn("old player", self.db.get_team_lines("EDM"))

        new_lines = {
            "new player": {"ev_line": "L1", "pp_unit": "PP1", "position": "C"}
        }
        self.db.save_team_lines("EDM", new_lines)
        fetched = self.db.get_team_lines("EDM")
        self.assertNotIn("old player", fetched)
        self.assertIn("new player", fetched)

    # Point B6: Historical game_date suppresses line factors
    def test_b6_past_date_line_suppression(self):
        lines = TeamLines({
            ("connor mcdavid", "F"): {"ev_line": "L1", "pp_unit": "PP1", "position": "C"}
        })
        mock_skaters = [
            {"playerId": 97, "firstName": {"default": "Connor"}, "lastName": {"default": "McDavid"}, "positionCode": "C", "gamesPlayed": 10, "goals": 10, "assists": 15, "shots": 40, "blockedShots": 5, "avgTimeOnIcePerGame": 21.0 * 60}
        ]
        with patch.object(self.client, "get_all_club_stats", return_value={"EDM": mock_skaters}):
            active, _, _, _, _, _ = collect_slate_skaters(
                self.client, self.db,
                team_matchups={"EDM": ("CGY", True)},
                goalies={}, injuries={}, boosters={},
                team_lines={"EDM": lines},
                game_date="2026-10-01",
                today_str="2026-10-18"
            )
        skater = active[0]
        self.assertIsNone(skater["ev_line"])
        self.assertIsNone(skater["pp_unit"])
        self.assertEqual(skater["line_tag"], "--")

        # build_skater_projection also ignores line multipliers on past dates
        logs = [{"gameDate": "2026-09-25", "goals": 1, "assists": 1, "shots": 4, "blockedShots": 0, "toi": "20:00", "opponentCommonName": {"default": "CGY"}, "homeRoadFlag": "H"}]
        xrrs_past, _ = build_skater_projection(
            {"pos": "C", "opp": "vsCGY", "is_home": True, "ev_line": "L1", "pp_unit": "PP1"},
            logs=logs, standings={"CGY": {"defense_factor": 1.0}},
            game_date="2026-10-01", today_str="2026-10-18"
        )
        xrrs_noline, _ = build_skater_projection(
            {"pos": "C", "opp": "vsCGY", "is_home": True, "ev_line": None, "pp_unit": None},
            logs=logs, standings={"CGY": {"defense_factor": 1.0}},
            game_date="2026-10-01", today_str="2026-10-18"
        )
        self.assertAlmostEqual(xrrs_past, xrrs_noline, places=2)

    # Point B7: Lines reporting string and DFO parsing
    def test_b7_lines_reporting_and_validation(self):
        mock_schedule = [
            {"date": "2026-10-18", "games": [{"homeTeam": "EDM", "awayTeam": "TOR"}]}
        ]
        with patch.object(self.client, "get_upcoming_schedule", return_value=mock_schedule), \
             patch.object(self.client, "get_injuries", return_value={}), \
             patch.object(self.client, "get_starting_goalies", return_value={}), \
             patch.object(self.client, "prefetch_teams_lines", return_value={"EDM": TeamLines({("connor mcdavid", "F"): {"ev_line": "L1", "pp_unit": "PP1"}})}), \
             patch.object(self.client, "get_all_club_stats", return_value={"EDM": []}):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                sync_game_day(self.client, self.db, target_date="2026-10-18", force=True)
            out = buf.getvalue()
            self.assertIn("Lines: loaded 1 of 2 teams (missing: TOR)", out)

    # Point B8: Booster value parsing and comment preservation
    def test_b8_booster_val_and_comments_parsing(self):
        self.assertEqual(parse_booster_val("2.8 #40 note"), 2.8)
        self.assertEqual(parse_booster_val("1."), 1.0)
        self.assertEqual(parse_booster_val(".5"), 0.5)
        self.assertEqual(parse_booster_val("+15%"), 0.15)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            val_80 = parse_booster_val("80")
        self.assertEqual(val_80, 0.8)
        self.assertIn("читаю 80 как 80%", buf.getvalue())

        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("Matt Boldy = 2.8 #40 note\nJack Eichel #25 = 1.5 # note\nConnor McDavid = 2.0\n")
            boosters = load_boosters_file(path)
            self.assertEqual(boosters.get("matt boldy"), 2.8)
            self.assertEqual(boosters.get("jack eichel #25"), 1.5)
            self.assertEqual(boosters.get("connor mcdavid"), 2.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point B9: settle_past_lineups boxscore fallback uses real blocks (no 1.0/0.3 fallback)
    def test_b9_settle_past_lineups_fallback_blocks(self):
        old_date = (nhl_now() - timedelta(days=4)).strftime("%Y-%m-%d")
        lineup = [{
            "id": 999, "name": "Old Defenseman", "team": "EDM", "slot": 1, "slot_boost": 2.0, "booster": 0.0,
            "total_boost": 2.0, "xRRS": 7.0, "score": 14.0, "base_pts": 7.0, "ev_line": "D1", "pp_unit": "PP1"
        }]
        self.db.save_lineup_history("2026-10-18", "optimal", lineup)
        with self.db._get_connection() as conn:
            conn.execute("UPDATE lineup_history SET game_date = ? WHERE player_id = 999", (old_date,))

        # Game log returns blockedShots = 2.0
        mock_logs = [{
            "gameDate": old_date, "gameId": 2026020001, "goals": 0, "assists": 1, "shots": 2, "blockedShots": 2.0, "toi": "22:00"
        }]
        with patch.object(self.client, "get_player_current_game_logs", return_value=mock_logs), \
             patch.object(self.client, "get_boxscore_skaters", return_value={}):  # empty boxscore
            settle_past_lineups(self.client, self.db)

        history = self.db.get_history_slate_details(old_date)
        self.assertEqual(len(history), 1)
        row = history[0]
        self.assertEqual(row["settled"], 1)
        # Expected xRRS with blocks = 2.0: 0*3.0 + 1*2.0 + 2*0.25 + 2*0.50 = 3.5
        expected_xrrs = calculate_rrs_from_stats(0.0, 1.0, 2.0, blocks=2.0)
        self.assertAlmostEqual(row["actual_xrrs"], expected_xrrs, places=2)

    # Point B10: lineup_history stores ev_line and pp_unit, with schema migration
    def test_b10_lineup_history_tracks_lines_and_migration(self):
        lineup = [{
            "id": 100, "name": "Connor McDavid", "team": "EDM", "slot": 1, "slot_boost": 2.0, "booster": 0.0,
            "total_boost": 2.0, "xRRS": 9.5, "score": 19.0, "base_pts": 9.5, "ev_line": "L1", "pp_unit": "PP1"
        }]
        self.db.save_lineup_history("2026-10-18", "optimal", lineup)
        records = self.db.get_history_slate_details("2026-10-18")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ev_line"], "L1")
        self.assertEqual(records[0]["pp_unit"], "PP1")

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            show_history_report(self.db, target_date="2026-10-18")
        out = buf.getvalue()
        self.assertIn("Line", out)
        self.assertIn("L1/PP1", out)

    # Point C15.1: Predict does not apply line factor twice
    def test_c15_1_no_double_line_factor_in_predict(self):
        predictor = PerformancePredictor()
        standings = {"CGY": {"defense_factor": 1.0}}
        pred_base = predictor.predict(
            game_logs=[], is_home=True, opponent_abbrev="CGY", game_date="2026-10-18",
            standings=standings, position="F", ev_line=None, pp_unit=None
        )
        pred_line = predictor.predict(
            game_logs=[], is_home=True, opponent_abbrev="CGY", game_date="2026-10-18",
            standings=standings, position="F", ev_line="L1", pp_unit="PP1"
        )
        ratio = pred_line["xG"] / pred_base["xG"]
        # Expected ratio: 1.05 * 1.12 = 1.176
        self.assertAlmostEqual(ratio, 1.176, places=2)
        # Double factor would be 1.176^2 = 1.383, which fails this assertion:
        self.assertLess(ratio, 1.30)

    # Point C15.2: Collect slate skaters applies line factor (L1/PP1 > L4)
    def test_c15_2_collect_slate_skaters_applies_line_factor(self):
        mock_skaters = [
            {"playerId": 201, "firstName": {"default": "Top"}, "lastName": {"default": "Guy"}, "positionCode": "F", "gamesPlayed": 30, "goals": 15, "assists": 20, "shots": 75, "blockedShots": 10, "avgTimeOnIcePerGame": 18.0 * 60},
            {"playerId": 202, "firstName": {"default": "Bot"}, "lastName": {"default": "Guy"}, "positionCode": "F", "gamesPlayed": 30, "goals": 15, "assists": 20, "shots": 75, "blockedShots": 10, "avgTimeOnIcePerGame": 18.0 * 60}
        ]
        lines = TeamLines({
            ("top guy", "F"): {"ev_line": "L1", "pp_unit": "PP1"},
            ("bot guy", "F"): {"ev_line": "L4", "pp_unit": None}
        })
        with patch.object(self.client, "get_all_club_stats", return_value={"EDM": mock_skaters}):
            active, _, _, _, _, _ = collect_slate_skaters(
                self.client, self.db,
                team_matchups={"EDM": ("CGY", True)},
                goalies={}, injuries={}, boosters={},
                team_lines={"EDM": lines},
                game_date="2026-10-18",
                today_str="2026-10-18"
            )
        s_top = next(s for s in active if s["id"] == 201)
        s_bot = next(s for s in active if s["id"] == 202)
        self.assertGreater(s_top["xRRS"], s_bot["xRRS"])

    # Point C15.3: Goalie table migration migrates old PK (team) to (team, game_date)
    def test_c15_3_goalie_table_migration(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE goalies (
                    team TEXT PRIMARY KEY,
                    goalie_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    save_pct REAL NOT NULL,
                    gaa REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("INSERT INTO goalies (team, goalie_name, status, save_pct, gaa, updated_at) VALUES ('EDM', 'Stuart Skinner', 'Confirmed', 0.910, 2.50, 1000.0)")
            conn.commit()
            conn.close()

            # Initialize Database: runs _init_tables and _migrate_goalies_table
            test_db = Database(db_path=db_path)
            # Insert second goalie for same team on different date
            test_db.save_starting_goalies({"EDM": {"goalie_name": "Calvin Pickard", "status": "Projected", "save_pct": 0.900, "gaa": 2.80}}, game_date="2026-10-19")
            with test_db._get_connection() as c:
                cur = c.cursor()
                cur.execute("SELECT team, game_date, goalie_name FROM goalies ORDER BY game_date")
                rows = cur.fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["goalie_name"], "Stuart Skinner")
            self.assertEqual(rows[0]["game_date"], "")
            self.assertEqual(rows[1]["goalie_name"], "Calvin Pickard")
            self.assertEqual(rows[1]["game_date"], "2026-10-19")
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    # Point C15.4: Booster file writes line comments (# L1/PP1, # D1)
    def test_c15_4_save_boosters_file_writes_line_comments(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            candidates = [
                {"name": "Connor McDavid", "team": "EDM", "pos": "F", "xRRS": 9.5, "avg_toi": 21.0, "ev_line": "L1", "pp_unit": "PP1"},
                {"name": "Evan Bouchard", "team": "EDM", "pos": "D", "xRRS": 8.0, "avg_toi": 23.0, "ev_line": "D1", "pp_unit": None}
            ]
            save_boosters_file({}, candidates, path=path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Connor McDavid", content)
            self.assertIn("# L1/PP1", content)
            self.assertIn("Evan Bouchard", content)
            self.assertIn("# D1", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Point C15.5: Bayesian shrinkage for gp < 10
    def test_c15_5_bayesian_shrinkage_low_gp(self):
        mock_skaters = [
            {"playerId": 301, "firstName": {"default": "Hot"}, "lastName": {"default": "Rookie"}, "positionCode": "F", "gamesPlayed": 1, "goals": 3, "assists": 0, "shots": 3, "blockedShots": 0, "avgTimeOnIcePerGame": 16.0 * 60}
        ]
        with patch.object(self.client, "get_all_club_stats", return_value={"EDM": mock_skaters}):
            active, _, _, _, _, _ = collect_slate_skaters(
                self.client, self.db,
                team_matchups={"EDM": ("CGY", True)},
                goalies={}, injuries={}, boosters={},
                team_lines={},
                game_date="2026-10-18",
                today_str="2026-10-18"
            )
        rookie = active[0]
        # Without shrinkage: 3 G/GP * 3.0 pts/G = 9.0 alone => xRRS > 9.0
        # With Bayesian shrinkage: (3 * 1 + 0.25 * 10) / 11 = 0.50 G/GP => xRRS < 5.0
        self.assertLess(rookie["xRRS"], 5.0)

    # Point C15.6: Teammate synergy only given to L1 and L2
    def test_c15_6_synergy_bonus_only_l1_l2(self):
        p_l1_a = {"id": 1, "pos": "F", "team": "EDM", "ev_line": "L1", "pp_unit": None, "goalie_factor": 1.0}
        p_l1_b = {"id": 2, "pos": "F", "team": "EDM", "ev_line": "L1", "pp_unit": None, "goalie_factor": 1.0}
        rate1, label1 = compute_pair_synergy(p_l1_a, p_l1_b)
        self.assertEqual(rate1, 0.10)
        self.assertEqual(label1, "[EDM L1]")

        p_l2_a = {"id": 3, "pos": "F", "team": "EDM", "ev_line": "L2", "pp_unit": None, "goalie_factor": 1.0}
        p_l2_b = {"id": 4, "pos": "F", "team": "EDM", "ev_line": "L2", "pp_unit": None, "goalie_factor": 1.0}
        rate2, label2 = compute_pair_synergy(p_l2_a, p_l2_b)
        self.assertEqual(rate2, 0.08)
        self.assertEqual(label2, "[EDM L2]")

        p_l3_a = {"id": 5, "pos": "F", "team": "EDM", "ev_line": "L3", "pp_unit": None, "goalie_factor": 1.0}
        p_l3_b = {"id": 6, "pos": "F", "team": "EDM", "ev_line": "L3", "pp_unit": None, "goalie_factor": 1.0}
        rate3, label3 = compute_pair_synergy(p_l3_a, p_l3_b)
        self.assertEqual(rate3, 0.06)
        self.assertEqual(label3, "[EDM Pair]")
        self.assertNotIn("L3", label3)


class TestReviewFixes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_fixes.db")
        self.db = Database(self.db_path)
        self.client = NHLClient(self.db)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    # 1. US Eastern Timezone fallback with accurate DST (EDT vs EST)
    def test_us_eastern_tz_dst(self):
        tz = USEasternTz()
        # Summer (July): EDT, UTC-4
        summer_dt = datetime(2026, 7, 15, 12, 0, tzinfo=tz)
        self.assertEqual(summer_dt.utcoffset(), timedelta(hours=-4))
        self.assertEqual(summer_dt.tzname(), "EDT")

        # Winter (January): EST, UTC-5
        winter_dt = datetime(2026, 1, 15, 12, 0, tzinfo=tz)
        self.assertEqual(winter_dt.utcoffset(), timedelta(hours=-5))
        self.assertEqual(winter_dt.tzname(), "EST")

    # 2. NHLClient.session thread safety
    def test_nhl_client_session_thread_safety(self):
        import threading

        # Verify threading.local guarantees each concurrent thread its own Session instance
        barrier = threading.Barrier(4)
        sessions = []

        def worker():
            s = self.client.session
            sessions.append(s)
            barrier.wait(timeout=5)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # 4 concurrent threads must have 4 distinct session objects
        self.assertEqual(len(sessions), 4)
        self.assertEqual(len(set(sessions)), 4)

        # Main thread gets its own distinct session
        self.assertNotIn(self.client.session, sessions)

        # Custom session setter overrides thread-local for mockability in tests
        mock_custom = MagicMock()
        self.client.session = mock_custom
        self.assertIs(self.client.session, mock_custom)

    # 3. Cache TTL default 3.0 hours for schedule and injuries
    def test_cache_ttl_max_age(self):
        # Schedule with max_age_hours
        with patch.object(self.db, "get_schedule", return_value=[{"date": "2026-10-18"}]) as mock_sched:
            sched = self.client.get_upcoming_schedule()
            mock_sched.assert_called_once_with(max_age_hours=3.0)

        # Injuries with max_age_hours
        with patch.object(self.db, "get_injuries", return_value={}) as mock_inj:
            inj = self.client.get_injuries()
            mock_inj.assert_called_once_with(max_age_hours=3.0)

    # 4. Expanded ESPN injury statuses
    def test_espn_injury_statuses(self):
        self.assertTrue(is_out_status("Out"))
        self.assertTrue(is_out_status("Injured Reserve"))
        self.assertTrue(is_out_status("IR"))
        self.assertTrue(is_out_status("Suspension"))
        self.assertTrue(is_out_status("Suspended"))
        self.assertFalse(is_out_status("Day-To-Day"))
        self.assertFalse(is_out_status("Questionable"))

        self.assertTrue(is_dtd_status("Day-To-Day"))
        self.assertTrue(is_dtd_status("Questionable"))
        self.assertTrue(is_dtd_status("Doubtful"))
        self.assertTrue(is_dtd_status("GTD"))
        self.assertTrue(is_dtd_status("Game-Time Decision"))
        self.assertFalse(is_dtd_status("Out"))
        self.assertFalse(is_dtd_status("Active"))

        # In collect_slate_skaters: Questionable skater without booster is routed to dtd_skaters
        skaters = [{"playerId": 10, "firstName": {"default": "Alex"}, "lastName": {"default": "Ovechkin"}, "positionCode": "L", "gamesPlayed": 10, "avgTimeOnIcePerGame": 20.0 * 60}]
        injuries = {("Alex Ovechkin", "WSH"): {"name": "Alex Ovechkin", "team": "WSH", "status": "Questionable", "detail": "Lower Body"}}
        with patch.object(self.client, "get_all_club_stats", return_value={"WSH": skaters}):
            active, dtd, depth, excluded, skipped, _ = collect_slate_skaters(
                self.client, self.db, {"WSH": ("PIT", True)},
                goalies={}, injuries=injuries, boosters={},
                game_date="2026-10-18", today_str="2026-10-18"
            )
        self.assertEqual(len(active), 0)
        self.assertEqual(len(dtd), 1)
        self.assertEqual(skipped, 1)
        self.assertIn("Questionable", dtd[0]["inj_detail"])

    # 5. Namesake 3-word alias generation in get_team_lines
    def test_namesake_alias_three_words(self):
        raw_html = '''
        <script id="__NEXT_DATA__" type="application/json">
        {
            "props": {
                "pageProps": {
                    "combinations": {
                        "players": [
                            {"name": "Elias Nils Pettersson", "positionIdentifier": "D", "groupIdentifier": "d3", "categoryIdentifier": "ev"}
                        ]
                    }
                }
            }
        }
        </script>
        '''
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = raw_html
        with patch.object(self.client.session, "get", return_value=mock_resp):
            lines = self.client.get_team_lines("VAN", force=True)
            self.assertIn(("elias pettersson", "D"), lines)

    # 6. Public db.get_role_bias_summary() and show_history_report
    def test_db_get_role_bias_summary(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 1, "name": "McDavid", "team": "EDM", "pos": "F", "xRRS": 10.0, "score": 20.0, "pp_unit": "PP1"},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "id": 2, "name": "Hyman", "team": "EDM", "pos": "F", "xRRS": 8.0, "score": 14.4, "pp_unit": "PP2"}
        ]
        self.db.save_lineup_history("2026-10-15", "standard", lineup)
        unsettled = self.db.get_unsettled_history()
        for r in unsettled:
            act = 22.0 if r["player_id"] == 1 else 12.4
            self.db.settle_history_player(r["id"], goals=1.0, assists=1.0, shots=3.0, toi="20:00", actual_xrrs=act, actual_rrs=act)

        summary = self.db.get_role_bias_summary()
        self.assertEqual(len(summary["pp1_diffs"]), 1)
        self.assertAlmostEqual(summary["pp1_diffs"][0], 2.0)
        self.assertEqual(len(summary["other_diffs"]), 1)
        self.assertAlmostEqual(summary["other_diffs"][0], -2.0)

        # Ensure show_history_report runs without private _get_connection
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            show_history_report(self.db)
        out = buf.getvalue()
        self.assertIn("Role Bias Tracking", out)
        self.assertIn("PP1 Avg Bias: +2.00 RRS", out)

    # 7. Synergy calculation matches assigned slot scores
    def test_solve_stacked_lineup_synergy_matches_assigned_slot_scores(self):
        cands = [
            {"id": 1, "name": "McDavid", "team": "EDM", "pos": "F", "xRRS": 10.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 2, "name": "Draisaitl", "team": "EDM", "pos": "F", "xRRS": 9.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 3, "name": "MacKinnon", "team": "COL", "pos": "F", "xRRS": 8.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"id": 4, "name": "Makar", "team": "COL", "pos": "D", "xRRS": 7.0, "booster": 0.0, "ev_line": "D1", "pp_unit": "PP1"},
            {"id": 5, "name": "Kucherov", "team": "TBL", "pos": "F", "xRRS": 6.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
        ]
        lineup, base_score, syn_score = solve_stacked_lineup(cands)
        edm_players = [p for p in lineup if p["team"] == "EDM"]
        col_players = [p for p in lineup if p["team"] == "COL"]
        edm_assigned_score = sum(p["score"] for p in edm_players)
        col_assigned_score = sum(p["score"] for p in col_players)
        expected_syn = round(edm_assigned_score * 0.15 + col_assigned_score * 0.13, 2)
        syn_diff = syn_score - base_score
        self.assertAlmostEqual(syn_diff, expected_syn, places=2)

    # 8. Past date guard against all-zero projections
    def test_main_past_date_all_zero_projections_warning(self):
        cands = [
            {"id": 1, "name": "Player 1", "team": "EDM", "pos": "F", "xRRS": 0.0, "base_pts": 0.0, "booster": 0.0, "is_home": True, "opp": "CGY", "opp_goalie": "", "goalie_factor": 1.0},
            {"id": 2, "name": "Player 2", "team": "EDM", "pos": "F", "xRRS": 0.0, "base_pts": 0.0, "booster": 0.0, "is_home": True, "opp": "CGY", "opp_goalie": "", "goalie_factor": 1.0},
        ]
        test_args = ["main.py", "--date", "2024-01-01"]
        mock_sched = [{"date": "2024-01-01", "games": [{"id": 1, "homeTeam": "EDM", "awayTeam": "CGY", "startTimeUTC": "2024-01-01T19:00:00Z"}]}]

        buf = io.StringIO()
        with patch("sys.argv", test_args), \
             patch("sys.stdout", buf), \
             patch("main.NHLClient") as mock_client_cls, \
             patch("main.Database") as mock_db_cls, \
             patch("main.collect_slate_skaters", return_value=(cands, [], [], 0, 0, set())), \
             patch("main.build_skater_projection", return_value=(0.0, 0.0)):
            mock_client = mock_client_cls.return_value
            mock_client.get_upcoming_schedule.return_value = mock_sched
            mock_client.get_standings.return_value = {}
            mock_client.prefetch_teams_lines.return_value = {}
            mock_client.get_starting_goalies.return_value = {}
            mock_client.get_injuries.return_value = {}
            mock_client.prefetch_players_game_logs.return_value = {}
            main_entry()

        out = buf.getvalue()
        self.assertIn("Warning: No valid historical projections available", out)
        self.assertNotIn("Lineup #1: Standard Optimal", out)


    # 9. OLS empirical calibration with settled history
    def test_calibrate_rrs_formula_with_settled_history(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 1, "name": "P1", "team": "EDM", "pos": "F", "xRRS": 5.0, "score": 10.0},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "id": 2, "name": "P2", "team": "EDM", "pos": "F", "xRRS": 4.0, "score": 7.2},
            {"slot": 3, "slot_boost": 1.6, "total_boost": 1.6, "id": 3, "name": "P3", "team": "EDM", "pos": "F", "xRRS": 3.0, "score": 4.8},
            {"slot": 4, "slot_boost": 1.4, "total_boost": 1.4, "id": 4, "name": "P4", "team": "EDM", "pos": "F", "xRRS": 2.5, "score": 3.5},
            {"slot": 5, "slot_boost": 1.2, "total_boost": 1.2, "id": 5, "name": "P5", "team": "EDM", "pos": "F", "xRRS": 2.0, "score": 2.4},
        ]
        self.db.save_lineup_history("2026-10-10", "standard", lineup)
        lineup2 = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 6, "name": "P6", "team": "COL", "pos": "F", "xRRS": 5.0, "score": 10.0}
        ]
        self.db.save_lineup_history("2026-10-11", "standard", lineup2)

        data = [
            (1, 2.0, 1.0, 4.0),
            (2, 1.0, 2.0, 3.0),
            (3, 0.0, 3.0, 5.0),
            (4, 3.0, 0.0, 6.0),
            (5, 0.0, 0.0, 2.0),
            (6, 1.0, 1.0, 1.0)
        ]
        unsettled = self.db.get_unsettled_history()
        for idx, (p_id, g, a, s) in enumerate(data):
            r = next(item for item in unsettled if item["player_id"] == p_id)
            rrs_val = 3.0 * g + 1.2 * a + 0.25 * s
            self.db.settle_history_player(r["id"], goals=g, assists=a, shots=s, toi="18:00", actual_xrrs=rrs_val, actual_rrs=rrs_val)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            res = calibrate_rrs_formula(self.db)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["weights"]["goals"], 3.0, delta=0.01)
        self.assertAlmostEqual(res["weights"]["assists"], 1.2, delta=0.01)
        self.assertAlmostEqual(res["weights"]["shots"], 0.25, delta=0.01)
        self.assertGreater(res["r2"], 0.99)
        self.assertIn("REAL SPORTS RRS EMPIRICAL FORMULA CALIBRATION", buf.getvalue())

    # 10. OLS empirical calibration with insufficient records
    def test_calibrate_rrs_formula_insufficient_samples(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            res = calibrate_rrs_formula(self.db)
        self.assertIsNone(res)
        self.assertIn("No settled player records found", buf.getvalue())

    # 11. Point-in-time evaluate_model eliminates look-ahead bias
    def test_evaluate_model_point_in_time(self):
        self.db.save_player(99, "Star Skater", "EDM", "F")
        logs = []
        for i in range(35):
            logs.append({
                "gameId": 1000 + i,
                "gameDate": f"2025-10-{i+1:02d}" if i < 30 else f"2025-11-{i-29:02d}",
                "season": "20252026",
                "homeRoadFlag": "H" if i % 2 == 0 else "R",
                "opponent": "CGY",
                "opponentAbbrev": "CGY",
                "goals": 1 if i % 3 == 0 else 0,
                "assists": 1 if i % 2 == 0 else 0,
                "shots": 3,
                "toi": "18:00",
                "toi_minutes": 18.0
            })
        self.db.save_game_logs(99, logs)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            evaluate_model(self.db, standings={"CGY": {"defense_factor": 0.5}}, min_games=30, point_in_time=True)
        out = buf.getvalue()
        self.assertIn("Validation Results", out)
        self.assertIn("Strict point-in-time (no look-ahead leakage)", out)
        self.assertIn("Goals (MAE):", out)

    # 12. Lineup-level validation summary and compounded multiplier metrics
    def test_lineup_level_validation_summary(self):
        lineup1 = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.5, "id": 101, "name": "P1", "team": "EDM", "pos": "F", "xRRS": 5.0, "score": 12.5, "booster": 0.5},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "id": 102, "name": "P2", "team": "EDM", "pos": "F", "xRRS": 4.0, "score": 7.2, "booster": 0.0},
        ]
        self.db.save_lineup_history("2026-10-12", "standard", lineup1)
        unsettled = self.db.get_unsettled_history()
        for r in unsettled:
            act = 15.0 if r["player_id"] == 101 else 6.0
            self.db.settle_history_player(r["id"], goals=1, assists=1, shots=3, toi="19:00", actual_xrrs=act, actual_rrs=act)

        summary = self.db.get_lineup_level_validation_summary()
        self.assertEqual(summary["settled_count"], 1)
        self.assertEqual(summary["hit_rate"], 1.0)
        self.assertAlmostEqual(summary["lineup_mae"], 1.3, delta=0.01)
        self.assertAlmostEqual(summary["lineup_bias"], 1.3, delta=0.01)
        self.assertIn(1, summary["slot_avg_errors"])
        self.assertIsNotNone(summary["booster_avg_error"])
        self.assertIsNotNone(summary["no_booster_avg_error"])

    # 13. Optimizer module direct export and equivalence
    def test_optimizer_module_direct_imports(self):
        candidates = [
            {"name": "P1", "team": "EDM", "xRRS": 6.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "P2", "team": "EDM", "xRRS": 5.5, "booster": 0.5, "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "P3", "team": "TOR", "xRRS": 5.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "P4", "team": "TOR", "xRRS": 4.5, "booster": 0.2, "ev_line": "L2", "pp_unit": "PP2", "pos": "F"},
            {"name": "P5", "team": "BOS", "xRRS": 4.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "P6", "team": "COL", "xRRS": 3.0, "booster": 1.0, "ev_line": "L3", "pp_unit": "NA", "pos": "F"},
        ]
        opt_lineup, opt_score = optimizer.solve_optimal_lineup(candidates)
        main_lineup, main_score = solve_optimal_lineup(candidates)
        self.assertEqual(len(opt_lineup), 5)
        self.assertEqual(opt_score, main_score)
        self.assertEqual([p["name"] for p in opt_lineup], [p["name"] for p in main_lineup])

        opt_stk, opt_base, opt_syn = optimizer.solve_stacked_lineup(candidates)
        main_stk, main_base, main_syn = solve_stacked_lineup(candidates)
        self.assertEqual(opt_syn, main_syn)
        self.assertEqual(opt_base, main_base)
        self.assertEqual(len(opt_stk), 5)
        self.assertEqual(optimizer.SLOT_BOOSTS, [2.0, 1.8, 1.6, 1.4, 1.2])

    # 14. DB app_rrs and app_payout schema, recording, and retrieval
    def test_db_app_rrs_and_payout_schema_and_recording(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.5, "id": 501, "name": "Skater 1", "team": "EDM", "pos": "F", "xRRS": 4.0, "score": 10.0, "booster": 0.5},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "id": 502, "name": "Skater 2", "team": "TOR", "pos": "F", "xRRS": 3.5, "score": 6.3, "booster": 0.0},
        ]
        self.db.save_lineup_history("2026-10-15", "standard", lineup)
        records = self.db.get_lineup_for_date("2026-10-15", mode="standard")
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0]["app_rrs"])

        # Record real app results
        self.db.record_app_results([
            {"id": records[0]["id"], "app_rrs": 5.2, "app_payout": 13.0},
            {"id": records[1]["id"], "app_rrs": 3.8, "app_payout": 6.84}
        ])
        updated = self.db.get_lineup_for_date("2026-10-15", mode="standard")
        self.assertEqual(updated[0]["app_rrs"], 5.2)
        self.assertEqual(updated[0]["app_payout"], 13.0)
        self.assertEqual(updated[1]["app_rrs"], 3.8)

        # Settle record, verify app_rrs preserved via COALESCE
        self.db.settle_history_player(records[0]["id"], goals=1, assists=1, shots=3, toi="18:00", actual_xrrs=4.7, actual_rrs=11.75, blocks=1.0)
        settled = self.db.get_lineup_for_date("2026-10-15", mode="standard")
        self.assertEqual(settled[0]["app_rrs"], 5.2)
        self.assertEqual(settled[0]["actual_blocks"], 1.0)

    # 15. Ground-truth OLS calibration and Multiplier Stacking Test (Additive vs Multiplicative)
    def test_calibrate_rrs_formula_with_ground_truth_app_rrs_and_stacking_test(self):
        lineup = [
            {"slot": i + 1, "slot_boost": [2.0, 1.8, 1.6, 1.4, 1.2, 1.0][i], "total_boost": [2.5, 2.0, 1.8, 1.4, 1.5, 1.0][i], "id": 600 + i, "name": f"P{i}", "team": "EDM", "pos": "F", "xRRS": 3.0, "score": 6.0, "booster": 0.5 if i % 2 == 0 else 0.0}
            for i in range(6)
        ]
        self.db.save_lineup_history("2026-10-16", "standard", lineup[:5])
        self.db.save_lineup_history("2026-10-17", "standard", [lineup[5]])

        # Create known ground-truth app ratings: app_rrs = 1.0 (intercept) + 3.0*G + 1.2*A + 0.25*S + 0.5*BLK
        data = [
            (600, 2.0, 1.0, 4.0, 1.0),
            (601, 1.0, 2.0, 3.0, 0.0),
            (602, 0.0, 3.0, 5.0, 2.0),
            (603, 3.0, 0.0, 6.0, 1.0),
            (604, 0.0, 0.0, 2.0, 0.0),
            (605, 1.0, 1.0, 1.0, 3.0)
        ]
        unsettled = self.db.get_unsettled_history()
        for p_id, g, a, s, blk in data:
            r = next(item for item in unsettled if item["player_id"] == p_id)
            ground_rrs = round(1.0 + 3.0 * g + 1.2 * a + 0.25 * s + 0.5 * blk, 2)
            tot_b = r["total_boost"]
            payout = round(ground_rrs * tot_b, 2)  # Strictly Additive: rrs * (slot + booster)
            self.db.settle_history_player(r["id"], goals=g, assists=a, shots=s, toi="18:00", actual_xrrs=ground_rrs, actual_rrs=payout, blocks=blk, app_rrs=ground_rrs, app_payout=payout)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            res = calibrate_rrs_formula(self.db)
        out = buf.getvalue()
        self.assertIsNotNone(res)
        self.assertTrue(res["is_ground_truth"])
        self.assertAlmostEqual(res["weights"]["intercept"], 1.0, delta=0.05)
        self.assertAlmostEqual(res["weights"]["goals"], 3.0, delta=0.05)
        self.assertAlmostEqual(res["weights"]["assists"], 1.2, delta=0.05)
        self.assertAlmostEqual(res["weights"]["shots"], 0.25, delta=0.05)
        self.assertAlmostEqual(res["weights"]["blocks"], 0.5, delta=0.05)
        self.assertIsNotNone(res["stacking_test"])
        self.assertEqual(res["stacking_test"]["verified_structure"], "Additive [slot + booster]")
        self.assertIn("Multiplier Stacking Structure Verification", out)

    # 16. Configurable slot boosts in optimizer
    def test_configurable_slot_boosts(self):
        candidates = [
            {"name": "P1", "team": "EDM", "xRRS": 5.0, "booster": 0.0, "pos": "F"},
            {"name": "P2", "team": "EDM", "xRRS": 4.5, "booster": 0.0, "pos": "F"},
            {"name": "P3", "team": "TOR", "xRRS": 4.0, "booster": 0.0, "pos": "F"},
            {"name": "P4", "team": "TOR", "xRRS": 3.5, "booster": 0.0, "pos": "F"},
            {"name": "P5", "team": "BOS", "xRRS": 3.0, "booster": 0.0, "pos": "F"},
        ]
        custom_slots = [3.0, 2.5, 2.0, 1.5, 1.0]
        lineup, score = optimizer.solve_optimal_lineup(candidates, slot_boosts=custom_slots)
        self.assertEqual(len(lineup), 5)
        self.assertEqual([p["slot_boost"] for p in lineup], [3.0, 2.5, 2.0, 1.5, 1.0])

        stk_lineup, base_sc, syn_sc = optimizer.solve_stacked_lineup(candidates, slot_boosts=custom_slots)
        self.assertEqual(len(stk_lineup), 5)
        self.assertEqual([p["slot_boost"] for p in stk_lineup], [3.0, 2.5, 2.0, 1.5, 1.0])

    # 17. Batch recording Real App ratings via handle_record_app_scores (pure ground truth)
    def test_handle_record_app_scores_batch(self):
        lineup = [
            {"slot": i + 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 700 + i, "name": f"Player {i+1}", "team": "EDM", "pos": "F", "xRRS": 3.0, "score": 6.0, "booster": 0.0}
            for i in range(5)
        ]
        self.db.save_lineup_history("2026-10-20", "standard", lineup)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            handle_record_app_scores(self.db, "2026-10-20:5.2,3.8,1.4,2.0,4.5")
        out = buf.getvalue()
        self.assertIn("Successfully recorded Real App scores for 5 players", out)

        recorded = self.db.get_lineup_for_date("2026-10-20", mode="standard")
        self.assertEqual([p["app_rrs"] for p in recorded], [5.2, 3.8, 1.4, 2.0, 4.5])
        # Payouts must be None when only ratings are provided to prevent circular stacking verification
        self.assertEqual([p["app_payout"] for p in recorded], [None, None, None, None, None])

        # Test explicit rrs=payout syntax
        buf2 = io.StringIO()
        with patch("sys.stdout", buf2):
            handle_record_app_scores(self.db, "2026-10-20:5.2=10.4,3.8=7.6,1.4,2.0,4.5")
        recorded2 = self.db.get_lineup_for_date("2026-10-20", mode="standard")
        self.assertEqual(recorded2[0]["app_payout"], 10.4)
        self.assertEqual(recorded2[1]["app_payout"], 7.6)
        self.assertIsNone(recorded2[2]["app_payout"])

    # 18. Graceful handling of invalid date in handle_record_app_scores
    def test_handle_record_app_scores_invalid_date(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            handle_record_app_scores(self.db, "invalid-date:5.2,3.8")
        out = buf.getvalue()
        self.assertIn("⚠️ Error:", out)

    # 19. Mutually exclusive group includes --record
    def test_record_mutually_exclusive_with_sync_and_history(self):
        test_args = ["main.py", "--record", "today", "--sync"]
        with patch("sys.argv", test_args), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                main_entry()

        test_args2 = ["main.py", "--record", "today", "--history"]
        with patch("sys.argv", test_args2), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                main_entry()

    # 20. show_history_report displays ground truth app_rrs, app_payout, and bias
    def test_show_history_report_with_app_ground_truth(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 801, "name": "McDavid", "team": "EDM", "pos": "F", "xRRS": 10.0, "score": 20.0, "pp_unit": "PP1"},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "id": 802, "name": "Draisaitl", "team": "EDM", "pos": "F", "xRRS": 8.0, "score": 14.4, "pp_unit": "PP1"}
        ]
        self.db.save_lineup_history("2026-10-21", "standard", lineup)
        unsettled = self.db.get_unsettled_history()
        for r in unsettled:
            self.db.settle_history_player(
                r["id"], goals=1, assists=1, shots=3, toi="20:00",
                actual_xrrs=10.0, actual_rrs=20.0, blocks=1.0,
                app_rrs=9.5, app_payout=19.0
            )

        # 1. Detail report
        buf_detail = io.StringIO()
        with patch("sys.stdout", buf_detail):
            show_history_report(self.db, target_date="2026-10-21")
        out_detail = buf_detail.getvalue()
        self.assertIn("App RRS", out_detail)
        self.assertIn("App Payout", out_detail)
        self.assertIn("9.5", out_detail)
        self.assertIn("19.0 RRS", out_detail)
        self.assertIn("Ground Truth Tracking", out_detail)
        self.assertIn("Model vs Real App Bias", out_detail)

        # 2. Summary report
        buf_summary = io.StringIO()
        with patch("sys.stdout", buf_summary):
            show_history_report(self.db)
        out_summary = buf_summary.getvalue()
        self.assertIn("App Payout", out_summary)
        self.assertIn("App Bias", out_summary)
        self.assertIn("App Ground Truth: 38.0 RRS", out_summary)

    # 21. Calibration sample size diagnostic warning when N < 15
    def test_calibrate_rrs_formula_sample_size_warning(self):
        lineup = [
            {"slot": i + 1, "slot_boost": 2.0, "total_boost": 2.0, "id": 900 + i, "name": f"P{i}", "team": "EDM", "pos": "F", "xRRS": 3.0, "score": 6.0, "booster": 0.0}
            for i in range(6)
        ]
        self.db.save_lineup_history("2026-10-22", "standard", lineup)
        unsettled = self.db.get_unsettled_history()
        for idx, r in enumerate(unsettled):
            self.db.settle_history_player(
                r["id"], goals=1, assists=1, shots=2, toi="18:00",
                actual_xrrs=4.0, actual_rrs=8.0, blocks=0.0,
                app_rrs=4.0, app_payout=8.0
            )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            res = calibrate_rrs_formula(self.db)
        out = buf.getvalue()
        self.assertIsNotNone(res)
        self.assertIsNotNone(res["sample_warning"])
        self.assertIn("Preliminary sample size (n=6 < 15)", out)


class TestPhase1FeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_phase1.db")
        self.db = Database(self.db_path)
        self.client = NHLClient(self.db)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_schedule_context_b2b_and_rest_days(self):
        schedule_days = [
            {
                "date": "2026-10-20",
                "games": [{"gameId": 1, "homeTeam": "CGY", "awayTeam": "EDM"}]
            },
            {
                "date": "2026-10-18",
                "games": [{"gameId": 2, "homeTeam": "TOR", "awayTeam": "MTL"}]
            },
            {
                "date": "2026-10-21",
                "games": [{"gameId": 3, "homeTeam": "EDM", "awayTeam": "TOR"}]
            }
        ]
        self.db.save_schedule(schedule_days)

        ctx = self.db.get_schedule_context("EDM", "TOR", "2026-10-21")
        self.assertTrue(ctx["is_home"])
        self.assertEqual(ctx["is_b2b"], 1)
        self.assertEqual(ctx["rest_days"], 1.0)
        self.assertEqual(ctx["opp_is_b2b"], 0)
        self.assertEqual(ctx["opp_rest_days"], 3.0)

    def test_schedule_context_defaults_on_missing_data(self):
        ctx = self.db.get_schedule_context("EDM", "TOR", "2026-10-21")
        self.assertTrue(ctx["is_home"])
        self.assertEqual(ctx["is_b2b"], 0)
        self.assertEqual(ctx["rest_days"], 2.0)
        self.assertEqual(ctx["opp_is_b2b"], 0)
        self.assertEqual(ctx["opp_rest_days"], 2.0)

    def test_standings_pk_and_shorthanded_persistence(self):
        standings_data = {
            "EDM": {
                "games_played": 10,
                "goals_against_avg": 2.8,
                "defense_factor": 0.95,
                "pk_pct": 0.852,
                "times_shorthanded_per_game": 3.4
            }
        }
        self.db.save_standings(standings_data)
        retrieved = self.db.get_standings()
        self.assertIn("EDM", retrieved)
        self.assertAlmostEqual(retrieved["EDM"]["pk_pct"], 0.852, places=3)
        self.assertAlmostEqual(retrieved["EDM"]["times_shorthanded_per_game"], 3.4, places=2)

    def test_game_logs_b2b_calculation_and_persistence(self):
        logs = [
            {
                "gameId": 101,
                "gameDate": "2026-10-15",
                "season": "20262027",
                "homeRoadFlag": "H",
                "opponent": "CGY",
                "opponentAbbrev": "CGY",
                "goals": 1,
                "assists": 0,
                "shots": 4,
                "toi": "19:00",
                "toi_minutes": 19.0
            },
            {
                "gameId": 102,
                "gameDate": "2026-10-16",
                "season": "20262027",
                "homeRoadFlag": "R",
                "opponent": "VAN",
                "opponentAbbrev": "VAN",
                "goals": 0,
                "assists": 2,
                "shots": 2,
                "toi": "18:30",
                "toi_minutes": 18.5
            }
        ]
        self.db.save_game_logs(5001, logs)
        saved = self.db.get_game_logs(5001)
        self.assertEqual(len(saved), 2)
        self.assertFalse(saved[0]["is_b2b"])
        self.assertTrue(saved[1]["is_b2b"])
        self.assertEqual(saved[1]["rest_days"], 1.0)

    def test_compute_line_factor_special_teams_modulation(self):
        g_base, a_base, toi_base = compute_line_factor("L1", "PP1", position="F", opp_pk_pct=0.80, opp_times_shorthanded=3.0)
        g_vuln, a_vuln, toi_vuln = compute_line_factor("L1", "PP1", position="F", opp_pk_pct=0.68, opp_times_shorthanded=4.5)
        g_elite, a_elite, toi_elite = compute_line_factor("L1", "PP1", position="F", opp_pk_pct=0.92, opp_times_shorthanded=1.8)

        self.assertGreater(g_vuln, g_base)
        self.assertGreater(a_vuln, a_base)
        self.assertGreater(toi_vuln, toi_base)
        self.assertLess(g_elite, g_base)
        self.assertLess(a_elite, a_base)
        self.assertLess(toi_elite, toi_base)

    def test_performance_predictor_poisson_glm_and_fatigue(self):
        predictor = PerformancePredictor(model_type="poisson")
        logs = []
        for i in range(15):
            logs.append({
                "gameId": 200 + i,
                "gameDate": f"2026-10-{i+1:02d}",
                "season": "20262027",
                "homeRoadFlag": "H" if i % 2 == 0 else "R",
                "opponent": "CGY",
                "opponentAbbrev": "CGY",
                "goals": 1 if i % 2 == 0 else 0,
                "assists": 1 if i % 3 == 0 else 0,
                "shots": 3,
                "toi": "20:00",
                "toi_minutes": 20.0
            })
        standings = {"CGY": {"defense_factor": 1.0, "pk_pct": 0.80, "times_shorthanded_per_game": 3.0}}
        predictor.train(logs, standings)

        pred_normal = predictor.predict(logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20", is_b2b=False, opp_is_b2b=False)
        pred_b2b = predictor.predict(logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20", is_b2b=True, opp_is_b2b=False)
        pred_opp_b2b = predictor.predict(logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20", is_b2b=False, opp_is_b2b=True)

        self.assertGreater(pred_normal["xRRS"], pred_b2b["xRRS"])
        self.assertGreater(pred_opp_b2b["xRRS"], pred_normal["xRRS"])
        self.assertGreaterEqual(pred_b2b["xG"], 0.0)
        self.assertGreaterEqual(pred_b2b["xA"], 0.0)
        self.assertGreaterEqual(pred_b2b["xSOG"], 0.0)

    def test_evaluate_model_comparative_benchmark_output(self):
        self.db.save_player(88, "Patrick Kane", "DET", "F")
        logs = []
        for i in range(32):
            logs.append({
                "gameId": 3000 + i,
                "gameDate": f"2025-10-{i+1:02d}" if i < 30 else f"2025-11-{i-29:02d}",
                "season": "20252026",
                "homeRoadFlag": "H" if i % 2 == 0 else "R",
                "opponent": "CHI",
                "opponentAbbrev": "CHI",
                "goals": 1 if i % 4 == 0 else 0,
                "assists": 1 if i % 2 == 0 else 0,
                "shots": 3,
                "toi": "18:00",
                "toi_minutes": 18.0
            })
        self.db.save_game_logs(88, logs)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            evaluate_model(self.db, standings={"CHI": {"defense_factor": 1.0}}, min_games=30, point_in_time=True)
        out = buf.getvalue()
        self.assertIn("MODEL VALIDATION & BENCHMARK COMPARISON", out)
        self.assertIn("20-Game Baseline", out)
        self.assertIn("Classic (Ridge)", out)
        self.assertIn("New (Poisson GLM)", out)
        self.assertIn("Spearman (r_s)", out)


class TestPhase2MonteCarloOptimizer(unittest.TestCase):
    def test_compute_pair_correlation_line_and_positions(self):
        # Different teams -> 0.0
        p_edm = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"}
        p_col = {"team": "COL", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm, p_col), 0.0)

        # Same team, L1 + PP1
        p_edm_f1 = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"}
        p_edm_f2 = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_f1, p_edm_f2), 0.40)

        # Same team, PP1 QB (D) + PP1 Forward (F)
        p_edm_d = {"team": "EDM", "ev_line": "D1", "pp_unit": "PP1", "pos": "D"}
        p_edm_f3 = {"team": "EDM", "ev_line": "L2", "pp_unit": "PP1", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_d, p_edm_f3), 0.35)

        # Same team, both PP1 forwards on different EV lines
        p_edm_f4 = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"}
        p_edm_f5 = {"team": "EDM", "ev_line": "L2", "pp_unit": "PP1", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_f4, p_edm_f5), 0.30)

        # Same team, L1 forwards (no PP)
        p_edm_f6 = {"team": "EDM", "ev_line": "L1", "pp_unit": "NA", "pos": "F"}
        p_edm_f7 = {"team": "EDM", "ev_line": "L1", "pp_unit": "NA", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_f6, p_edm_f7), 0.24)

        # Same team, L2 forwards
        p_edm_f8 = {"team": "EDM", "ev_line": "L2", "pp_unit": "NA", "pos": "F"}
        p_edm_f9 = {"team": "EDM", "ev_line": "L2", "pp_unit": "NA", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_f8, p_edm_f9), 0.20)

        # Same team, PP2 forwards
        p_edm_f10 = {"team": "EDM", "ev_line": "L3", "pp_unit": "PP2", "pos": "F"}
        p_edm_f11 = {"team": "EDM", "ev_line": "L4", "pp_unit": "PP2", "pos": "F"}
        self.assertEqual(compute_pair_correlation(p_edm_f10, p_edm_f11), 0.18)

        # Same team generic players
        p_edm_gen1 = {"team": "EDM", "ev_line": "L3", "pp_unit": "NA", "pos": "F"}
        p_edm_gen2 = {"team": "EDM", "ev_line": "D3", "pp_unit": "NA", "pos": "D"}
        self.assertEqual(compute_pair_correlation(p_edm_gen1, p_edm_gen2), 0.08)

        # Goalie factor modulation
        p_g1 = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F", "goalie_factor": 1.25}
        p_g2 = {"team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F", "goalie_factor": 1.25}
        self.assertGreater(compute_pair_correlation(p_g1, p_g2), 0.40)

    def test_build_lineup_correlation_matrix_properties(self):
        lineup = [
            {"name": "McDavid", "team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "Draisaitl", "team": "EDM", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "MacKinnon", "team": "COL", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
            {"name": "Makar", "team": "COL", "ev_line": "D1", "pp_unit": "PP1", "pos": "D"},
            {"name": "Kucherov", "team": "TBL", "ev_line": "L1", "pp_unit": "PP1", "pos": "F"},
        ]
        R = build_lineup_correlation_matrix(lineup)
        self.assertEqual(R.shape, (5, 5))
        self.assertTrue(np.allclose(R, R.T))
        self.assertTrue(np.allclose(np.diag(R), np.ones(5)))
        self.assertEqual(R[0, 2], 0.0)
        self.assertEqual(R[1, 4], 0.0)
        self.assertEqual(R[0, 1], 0.40)
        self.assertEqual(R[2, 3], 0.35)

    def test_simulate_lineup_distribution_speed_and_quantiles(self):
        lineup = [
            {"name": "P1", "team": "EDM", "pos": "F", "xRRS": 8.0, "slot_boost": 2.0, "total_boost": 2.0, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P2", "team": "EDM", "pos": "F", "xRRS": 7.0, "slot_boost": 1.8, "total_boost": 1.8, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P3", "team": "COL", "pos": "F", "xRRS": 6.5, "slot_boost": 1.6, "total_boost": 1.6, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P4", "team": "COL", "pos": "D", "xRRS": 6.0, "slot_boost": 1.4, "total_boost": 1.4, "booster": 0.0, "ev_line": "D1", "pp_unit": "PP1"},
            {"name": "P5", "team": "TBL", "pos": "F", "xRRS": 5.5, "slot_boost": 1.2, "total_boost": 1.2, "booster": 0.0, "ev_line": "L1", "pp_unit": "PP1"},
        ]
        t0 = time.perf_counter()
        sim = simulate_lineup_distribution(lineup, n_sims=10000, seed=42)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.50)
        self.assertIn("mean", sim)
        self.assertIn("std", sim)
        self.assertIn("floor_25", sim)
        self.assertIn("median_50", sim)
        self.assertIn("ceiling_90", sim)
        self.assertIn("ceiling_95", sim)
        self.assertLess(sim["floor_25"], sim["median_50"])
        self.assertLess(sim["median_50"], sim["ceiling_90"])
        self.assertLess(sim["ceiling_90"], sim["ceiling_95"])
        self.assertGreater(sim["std"], 0.0)

    def test_simulate_lineup_distribution_stacking_covariance_effect(self):
        lineup_uncorr = [
            {"name": "P1", "team": "EDM", "pos": "F", "xRRS": 8.0, "total_boost": 2.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P2", "team": "TOR", "pos": "F", "xRRS": 7.0, "total_boost": 1.8, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P3", "team": "COL", "pos": "F", "xRRS": 6.5, "total_boost": 1.6, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P4", "team": "BOS", "pos": "F", "xRRS": 6.0, "total_boost": 1.4, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P5", "team": "TBL", "pos": "F", "xRRS": 5.5, "total_boost": 1.2, "ev_line": "L1", "pp_unit": "PP1"},
        ]

        lineup_stacked = [
            {"name": "P1", "team": "EDM", "pos": "F", "xRRS": 8.0, "total_boost": 2.0, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P2", "team": "EDM", "pos": "F", "xRRS": 7.0, "total_boost": 1.8, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P3", "team": "COL", "pos": "F", "xRRS": 6.5, "total_boost": 1.6, "ev_line": "L1", "pp_unit": "PP1"},
            {"name": "P4", "team": "COL", "pos": "D", "xRRS": 6.0, "total_boost": 1.4, "ev_line": "D1", "pp_unit": "PP1"},
            {"name": "P5", "team": "TBL", "pos": "F", "xRRS": 5.5, "total_boost": 1.2, "ev_line": "L1", "pp_unit": "PP1"},
        ]

        res_uncorr = simulate_lineup_distribution(lineup_uncorr, n_sims=10000, seed=123)
        res_stacked = simulate_lineup_distribution(lineup_stacked, n_sims=10000, seed=123)

        self.assertAlmostEqual(res_uncorr["mean"], res_stacked["mean"], delta=1.5)
        self.assertGreater(res_stacked["std"], res_uncorr["std"])
        self.assertGreater(res_stacked["ceiling_90"], res_uncorr["ceiling_90"])
        self.assertGreater(res_stacked["ceiling_95"], res_uncorr["ceiling_95"])
        self.assertLess(res_stacked["floor_25"], res_uncorr["floor_25"])

    def test_print_lineup_table_includes_monte_carlo_quantiles(self):
        lineup = [
            {"slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "name": "Connor McDavid", "team": "EDM", "opp": "vsCGY", "opp_goalie": "D. Wolf", "xRRS": 9.5, "booster": 0.0, "score": 19.0, "line_tag": "L1/PP1", "stack": "[EDM PP1]"},
            {"slot": 2, "slot_boost": 1.8, "total_boost": 1.8, "name": "Leon Draisaitl", "team": "EDM", "opp": "vsCGY", "opp_goalie": "D. Wolf", "xRRS": 8.5, "booster": 0.0, "score": 15.3, "line_tag": "L1/PP1", "stack": "[EDM PP1]"},
            {"slot": 3, "slot_boost": 1.6, "total_boost": 1.6, "name": "Cale Makar", "team": "COL", "opp": "@PHI", "opp_goalie": "S. Ersson", "xRRS": 8.0, "booster": 0.0, "score": 12.8, "line_tag": "D1/PP1", "stack": ""},
            {"slot": 4, "slot_boost": 1.4, "total_boost": 1.4, "name": "Kirill Kaprizov", "team": "MIN", "opp": "vsCBJ", "opp_goalie": "E. Merzlikins", "xRRS": 7.5, "booster": 0.0, "score": 10.5, "line_tag": "L1/PP1", "stack": ""},
            {"slot": 5, "slot_boost": 1.2, "total_boost": 1.2, "name": "Sidney Crosby", "team": "PIT", "opp": "@UTA", "opp_goalie": "C. Ingram", "xRRS": 7.0, "booster": 0.0, "score": 8.4, "line_tag": "L1/PP1", "stack": ""}
        ]
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_lineup_table("2026-10-20", lineup, 66.0, title="Tournament Stack", synergy_score=72.5)
        out = buf.getvalue()

        self.assertIn("Monte-Carlo Simulation", out)
        self.assertIn("Expected Score (Mean):", out)
        self.assertIn("Floor (25th percentile):", out)
        self.assertIn("Ceiling (90th percentile):", out)
        self.assertIn("Ceiling (95th percentile):", out)

    def test_solve_optimal_lineup_runtime_complexity(self):
        large_cands = [
            {"id": i, "name": f"Skater_{i}", "team": f"T{i % 10}", "pos": "F", "xRRS": 2.0 + (i % 8) * 0.8, "booster": 0.1 * (i % 5)}
            for i in range(100)
        ]
        t0 = time.perf_counter()
        lineup, score = optimizer.solve_optimal_lineup(large_cands)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(lineup), 5)
        self.assertGreater(score, 0.0)
        self.assertLess(elapsed, 0.05)


class TestPhase3ScrapingAndFallback(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_phase3.db")
        self.db = Database(self.db_path)
        self.client = NHLClient(self.db)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_request_with_retry_exponential_backoff_on_transient_error(self):
        resp_500_1 = MagicMock(status_code=500)
        resp_500_2 = MagicMock(status_code=502)
        resp_200 = MagicMock(status_code=200, text="OK")

        mock_req = MagicMock(side_effect=[resp_500_1, resp_500_2, resp_200])
        with patch.object(self.client.session, "get", mock_req), patch("time.sleep") as mock_sleep:
            res = self.client.request_with_retry("https://api-test.com/endpoint", max_retries=3, backoff_base=0.1)
            self.assertEqual(res.status_code, 200)
            self.assertEqual(mock_req.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_any_call(0.1)
            mock_sleep.assert_any_call(0.2)

    def test_request_with_retry_raises_after_max_retries(self):
        mock_req = MagicMock(side_effect=requests.exceptions.ConnectionError("Connection dropped"))
        with patch.object(self.client.session, "get", mock_req), patch("time.sleep"):
            with self.assertRaises(requests.exceptions.ConnectionError):
                self.client.request_with_retry("https://api-test.com/fail", max_retries=3, backoff_base=0.01)

    def test_team_lines_dfo_failure_triggers_nhl_api_toi_fallback(self):
        dfo_resp = MagicMock(status_code=500)

        sample_skaters = [
            {"playerId": 101, "firstName": {"default": "Connor"}, "lastName": {"default": "McDavid"}, "positionCode": "C"},
            {"playerId": 102, "firstName": {"default": "Leon"}, "lastName": {"default": "Draisaitl"}, "positionCode": "C"},
            {"playerId": 103, "firstName": {"default": "Vasily"}, "lastName": {"default": "Podkolzin"}, "positionCode": "R"},
            {"playerId": 104, "firstName": {"default": "Zach"}, "lastName": {"default": "Hyman"}, "positionCode": "L"},
            {"playerId": 105, "firstName": {"default": "Kasperi"}, "lastName": {"default": "Kapanen"}, "positionCode": "R"},
            {"playerId": 106, "firstName": {"default": "Mathieu"}, "lastName": {"default": "Joseph"}, "positionCode": "R"},
            {"playerId": 201, "firstName": {"default": "Evan"}, "lastName": {"default": "Bouchard"}, "positionCode": "D"},
            {"playerId": 202, "firstName": {"default": "Ryan"}, "lastName": {"default": "Shea"}, "positionCode": "D"},
            {"playerId": 203, "firstName": {"default": "Connor"}, "lastName": {"default": "Murphy"}, "positionCode": "D"},
            {"playerId": 204, "firstName": {"default": "Mattias"}, "lastName": {"default": "Ekholm"}, "positionCode": "D"},
        ]

        toi_rows = [
            {"playerId": 101, "evTimeOnIcePerGame": 1100.0, "ppTimeOnIcePerGame": 210.0, "timeOnIcePerGame": 1310.0},
            {"playerId": 102, "evTimeOnIcePerGame": 1060.0, "ppTimeOnIcePerGame": 200.0, "timeOnIcePerGame": 1260.0},
            {"playerId": 103, "evTimeOnIcePerGame": 980.0, "ppTimeOnIcePerGame": 190.0, "timeOnIcePerGame": 1170.0},
            {"playerId": 104, "evTimeOnIcePerGame": 920.0, "ppTimeOnIcePerGame": 150.0, "timeOnIcePerGame": 1070.0},
            {"playerId": 105, "evTimeOnIcePerGame": 800.0, "ppTimeOnIcePerGame": 120.0, "timeOnIcePerGame": 920.0},
            {"playerId": 106, "evTimeOnIcePerGame": 780.0, "ppTimeOnIcePerGame": 30.0, "timeOnIcePerGame": 810.0},
            {"playerId": 201, "evTimeOnIcePerGame": 1250.0, "ppTimeOnIcePerGame": 205.0, "timeOnIcePerGame": 1455.0},
            {"playerId": 202, "evTimeOnIcePerGame": 1180.0, "ppTimeOnIcePerGame": 80.0, "timeOnIcePerGame": 1260.0},
            {"playerId": 203, "evTimeOnIcePerGame": 1150.0, "ppTimeOnIcePerGame": 10.0, "timeOnIcePerGame": 1160.0},
            {"playerId": 204, "evTimeOnIcePerGame": 1120.0, "ppTimeOnIcePerGame": 20.0, "timeOnIcePerGame": 1140.0},
        ]
        toi_resp = MagicMock(status_code=200)
        toi_resp.json.return_value = {"data": toi_rows}

        def fake_get(url, *args, **kwargs):
            if "dailyfaceoff.com" in url:
                return dfo_resp
            elif "timeonice" in url:
                return toi_resp
            return MagicMock(status_code=404)

        with patch.object(self.client, "request_with_retry", side_effect=fake_get), \
             patch.object(self.client, "_fetch_team_skaters", return_value=sample_skaters):
            lines = self.client.get_team_lines("EDM", force=True)

        self.assertIsNotNone(lines)
        self.assertEqual(lines._meta.get("source"), "nhl_api_fallback")
        self.assertTrue(lines._meta.get("has_pp"))

        mcdavid = lines.get(("connor mcdavid", "F"))
        self.assertIsNotNone(mcdavid)
        self.assertEqual(mcdavid["ev_line"], "L1")
        self.assertEqual(mcdavid["pp_unit"], "PP1")

        draisaitl = lines.get(("leon draisaitl", "F"))
        self.assertEqual(draisaitl["ev_line"], "L1")
        self.assertEqual(draisaitl["pp_unit"], "PP1")

        hyman = lines.get(("zach hyman", "F"))
        self.assertEqual(hyman["ev_line"], "L2")

        bouchard = lines.get(("evan bouchard", "D"))
        self.assertEqual(bouchard["ev_line"], "D1")
        self.assertEqual(bouchard["pp_unit"], "PP1")

        murphy = lines.get(("connor murphy", "D"))
        self.assertEqual(murphy["ev_line"], "D2")

        saved = self.db.get_team_lines("EDM")
        self.assertEqual(saved.get(("connor mcdavid", "F"))["ev_line"], "L1")


class TestPhase4ImageParserAndOCR(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_phase4.db")
        self.boosters_path = os.path.join(self.tmp_dir, "test_boosters.txt")
        self.db = Database(self.db_path)
        self.client = NHLClient(self.db)

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_parse_booster_multiplier_tokens(self):
        from image_parser import parse_booster_multiplier_token
        self.assertEqual(parse_booster_multiplier_token("+0.4"), 0.4)
        self.assertEqual(parse_booster_multiplier_token("1.2x"), 1.2)
        self.assertEqual(parse_booster_multiplier_token("+20%"), 0.2)
        self.assertEqual(parse_booster_multiplier_token("0.8"), 0.8)
        self.assertEqual(parse_booster_multiplier_token("3.0"), 3.0)
        self.assertEqual(parse_booster_multiplier_token("+0"), 0.0)
        self.assertEqual(parse_booster_multiplier_token("15%"), 0.15)
        self.assertIsNone(parse_booster_multiplier_token("invalid_token"))

    def test_extract_booster_candidates_single_line(self):
        from image_parser import extract_booster_candidates_from_text
        text = """
        Leon Draisaitl +0.4
        Matt Boldy = 0.8
        Evan Bouchard (D) +1.1
        Alexis Lafreniere: +20%
        """
        candidates = extract_booster_candidates_from_text(text)
        self.assertEqual(len(candidates), 4)

        names = {c["raw_name"]: c["booster_val"] for c in candidates}
        self.assertEqual(names.get("Leon Draisaitl"), 0.4)
        self.assertEqual(names.get("Matt Boldy"), 0.8)
        self.assertEqual(names.get("Evan Bouchard"), 1.1)
        self.assertEqual(names.get("Alexis Lafreniere"), 0.2)

        # Check Evan Bouchard has pos D
        bouchard = next(c for c in candidates if c["raw_name"] == "Evan Bouchard")
        self.assertEqual(bouchard.get("pos"), "D")

    def test_extract_booster_candidates_multiline_card(self):
        from image_parser import extract_booster_candidates_from_text
        text = """
        REAL SPORTS FANTASY DRAFT
        Connor McDavid
        EDM • C
        +0.4
        
        Cale Makar
        COL - D
        +1.4
        
        Nathan MacKinnon
        COL - C
        +0.0
        """
        candidates = extract_booster_candidates_from_text(text)
        self.assertGreaterEqual(len(candidates), 3)

        cand_map = {c["raw_name"]: c for c in candidates}
        self.assertIn("Connor McDavid", cand_map)
        self.assertEqual(cand_map["Connor McDavid"]["booster_val"], 0.4)
        self.assertEqual(cand_map["Connor McDavid"]["team"], "EDM")
        self.assertEqual(cand_map["Connor McDavid"]["pos"], "C")

        self.assertIn("Cale Makar", cand_map)
        self.assertEqual(cand_map["Cale Makar"]["booster_val"], 1.4)
        self.assertEqual(cand_map["Cale Makar"]["team"], "COL")
        self.assertEqual(cand_map["Cale Makar"]["pos"], "D")

    def test_match_player_candidate_fuzzy_accuracy(self):
        from image_parser import match_player_candidate
        slate = [
            {"id": 101, "name": "Leon Draisaitl", "team": "EDM", "pos": "C", "is_duplicate": False},
            {"id": 102, "name": "Connor McDavid", "team": "EDM", "pos": "C", "is_duplicate": False},
            {"id": 103, "name": "Alexis Lafrenière", "team": "NYR", "pos": "L", "is_duplicate": False},
        ]
        # Slight OCR typo: 'Leon Draisaiti'
        cand_typo = {"raw_name": "Leon Draisaiti", "team": "EDM", "pos": "C", "booster_val": 0.4}
        res = match_player_candidate(cand_typo, slate)
        self.assertIsNotNone(res)
        matched_p, canon_key, conf = res
        self.assertEqual(matched_p["name"], "Leon Draisaitl")
        self.assertEqual(canon_key, "leon draisaitl")
        self.assertGreater(conf, 85.0)

        # Accented name matching
        cand_acc = {"raw_name": "Alexis Lafreniere", "team": "NYR", "pos": "F", "booster_val": 0.2}
        res_acc = match_player_candidate(cand_acc, slate)
        self.assertIsNotNone(res_acc)
        self.assertEqual(res_acc[0]["name"], "Alexis Lafrenière")

    def test_match_player_candidate_namesake_disambiguation(self):
        from image_parser import match_player_candidate
        from main import normalize_booster_name
        slate = [
            {"id": 201, "name": "Elias Pettersson", "team": "VAN", "pos": "C", "is_duplicate": True},
            {"id": 202, "name": "Elias Pettersson", "team": "VAN", "pos": "D", "is_duplicate": True},
        ]
        # Case 1: Defenseman with explicit position
        cand_d = {"raw_name": "Elias Pettersson", "team": "VAN", "pos": "D", "booster_val": 0.0}
        res_d = match_player_candidate(cand_d, slate)
        self.assertIsNotNone(res_d)
        _, canon_key_d, _ = res_d
        self.assertEqual(canon_key_d, normalize_booster_name("Elias Pettersson (D)"))

        # Case 2: Center with explicit position
        cand_c = {"raw_name": "Elias Pettersson", "team": "VAN", "pos": "C", "booster_val": 1.4}
        res_c = match_player_candidate(cand_c, slate)
        self.assertIsNotNone(res_c)
        _, canon_key_c, _ = res_c
        self.assertEqual(canon_key_c, normalize_booster_name("Elias Pettersson (C)"))

        # Case 3: Missing position tag on duplicate player -> skipped to prevent misattribution
        cand_ambig = {"raw_name": "Elias Pettersson", "team": "VAN", "pos": None, "booster_val": 1.4}
        res_ambig = match_player_candidate(cand_ambig, slate)
        self.assertIsNone(res_ambig)

    def test_parse_images_directory_with_mock_ocr(self):
        from image_parser import MockOCR, parse_images_directory
        # Create dummy image files
        img1 = os.path.join(self.tmp_dir, "draft_page1.png")
        img2 = os.path.join(self.tmp_dir, "draft_page2.png")
        with open(img1, "wb") as f:
            f.write(b"PNG_FAKE_BYTES_1")
        with open(img2, "wb") as f:
            f.write(b"PNG_FAKE_BYTES_2")

        ocr_map = {
            "draft_page1.png": "Connor McDavid\nEDM - C\n+0.4\nLeon Draisaitl\nEDM - C\n+0.6\n",
            "draft_page2.png": "Evan Bouchard\nEDM - D\n+1.1\nMatt Boldy = 0.8\n"
        }
        mock_backend = MockOCR(text_mapping=ocr_map)

        slate = [
            {"id": 1, "name": "Connor McDavid", "team": "EDM", "pos": "C", "is_duplicate": False},
            {"id": 2, "name": "Leon Draisaitl", "team": "EDM", "pos": "C", "is_duplicate": False},
            {"id": 3, "name": "Evan Bouchard", "team": "EDM", "pos": "D", "is_duplicate": False},
            {"id": 4, "name": "Matt Boldy", "team": "MIN", "pos": "L", "is_duplicate": False},
        ]

        boosters, detailed = parse_images_directory(self.tmp_dir, slate, ocr_backend=mock_backend)
        self.assertEqual(len(boosters), 4)
        self.assertEqual(boosters.get("connor mcdavid"), 0.4)
        self.assertEqual(boosters.get("leon draisaitl"), 0.6)
        self.assertEqual(boosters.get("evan bouchard"), 1.1)
        self.assertEqual(boosters.get("matt boldy"), 0.8)
        self.assertEqual(len(detailed), 4)

    def test_handle_parse_images_orchestrator(self):
        from image_parser import MockOCR, handle_parse_images
        img = os.path.join(self.tmp_dir, "screen.png")
        with open(img, "wb") as f:
            f.write(b"DUMMY_IMAGE")

        mock_ocr = MockOCR(default_text="Nathan MacKinnon\nCOL - C\n+0.4\nCale Makar\nCOL - D\n+1.2\n")

        # Mock schedule with games
        fake_schedule = [
            {
                "date": "2026-10-21",
                "games": [{"homeTeam": "COL", "awayTeam": "EDM"}]
            }
        ]

        with patch.object(self.client, "get_upcoming_schedule", return_value=fake_schedule), \
             patch.object(self.client, "get_injuries", return_value=[]), \
             patch.object(self.client, "get_starting_goalies", return_value=[]), \
             patch.object(self.client, "prefetch_teams_lines", return_value={}), \
             patch("main.collect_slate_skaters", return_value=(
                 [
                     {"name": "Nathan MacKinnon", "team": "COL", "pos": "C", "id": 11, "is_duplicate": False, "xRRS": 12.0, "booster": 0.0},
                     {"name": "Cale Makar", "team": "COL", "pos": "D", "id": 12, "is_duplicate": False, "xRRS": 11.0, "booster": 0.0}
                 ],
                 [], [], 0, 0, set()
             )):
            res = handle_parse_images(
                self.tmp_dir,
                self.client,
                self.db,
                target_date="2026-10-21",
                boosters_path=self.boosters_path,
                ocr_backend=mock_ocr
            )

        self.assertEqual(len(res), 2)
        self.assertEqual(res.get("nathan mackinnon"), 0.4)
        self.assertEqual(res.get("cale makar"), 1.2)

        # Check file was written
        self.assertTrue(os.path.exists(self.boosters_path))
        with open(self.boosters_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Nathan MacKinnon", content)
        self.assertIn("Cale Makar", content)

        # Check DB was synchronized
        db_boosters = self.db.get_all_boosters()
        self.assertEqual(db_boosters.get("nathan mackinnon"), 0.4)
        self.assertEqual(db_boosters.get("cale makar"), 1.2)

    def test_cli_parse_images_mutually_exclusive_with_sync(self):
        from main import main as main_entry
        test_args = ["main.py", "--parse-images", "some_folder", "--sync"]
        with patch("sys.argv", test_args), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                main_entry()

    def test_cli_parse_images_mutually_exclusive_with_record(self):
        from main import main as main_entry
        test_args = ["main.py", "--parse-images", "some_folder", "--record", "today"]
        with patch("sys.argv", test_args), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                main_entry()


class TestPhase5ComprehensiveSuite(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_phase5.db")
        self.db = Database(self.db_path)
        self.client = NHLClient(self.db)

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_b2b_fatigue_schedule_edge_cases(self):
        # 1. Season opener edge case: no past games in schedule
        ctx_opener = self.db.get_schedule_context("EDM", "CGY", "2026-10-01")
        self.assertFalse(ctx_opener["is_b2b"])
        self.assertEqual(ctx_opener["rest_days"], 2.0)
        self.assertFalse(ctx_opener["opp_is_b2b"])
        self.assertEqual(ctx_opener["opp_rest_days"], 2.0)

        # 2. Double B2B: both teams played on 2026-10-19, playing against each other on 2026-10-20
        schedule_days = [
            {
                "date": "2026-10-19",
                "games": [
                    {"gameId": 10, "homeTeam": "EDM", "awayTeam": "VAN"},
                    {"gameId": 11, "homeTeam": "CGY", "awayTeam": "SEA"}
                ]
            },
            {
                "date": "2026-10-20",
                "games": [
                    {"gameId": 12, "homeTeam": "CGY", "awayTeam": "EDM"}
                ]
            }
        ]
        self.db.save_schedule(schedule_days)

        ctx_edm = self.db.get_schedule_context("EDM", "CGY", "2026-10-20")
        self.assertTrue(ctx_edm["is_b2b"])
        self.assertEqual(ctx_edm["rest_days"], 1.0)
        self.assertTrue(ctx_edm["opp_is_b2b"])
        self.assertEqual(ctx_edm["opp_rest_days"], 1.0)
        self.assertFalse(ctx_edm["is_home"])

        ctx_cgy = self.db.get_schedule_context("CGY", "EDM", "2026-10-20")
        self.assertTrue(ctx_cgy["is_b2b"])
        self.assertEqual(ctx_cgy["rest_days"], 1.0)
        self.assertTrue(ctx_cgy["opp_is_b2b"])
        self.assertEqual(ctx_cgy["opp_rest_days"], 1.0)
        self.assertTrue(ctx_cgy["is_home"])

        # 3. Fatigue modulation in PerformancePredictor
        predictor = PerformancePredictor(model_type="poisson")
        sample_logs = [
            {"gameDate": "2026-10-10", "goals": 1, "assists": 1, "shots": 4, "toi": "21:30"},
            {"gameDate": "2026-10-12", "goals": 0, "assists": 2, "shots": 3, "toi": "20:45"},
            {"gameDate": "2026-10-15", "goals": 1, "assists": 0, "shots": 4, "toi": "22:15"},
        ]
        # Predict with B2B fatigue vs well-rested
        pred_rested = predictor.predict(
            sample_logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20",
            is_b2b=False, opp_is_b2b=False
        )
        pred_fatigued = predictor.predict(
            sample_logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20",
            is_b2b=True, opp_is_b2b=False
        )
        pred_opp_b2b = predictor.predict(
            sample_logs, is_home=True, opponent_abbrev="CGY", game_date="2026-10-20",
            is_b2b=False, opp_is_b2b=True
        )

        self.assertLess(pred_fatigued["xG"], pred_rested["xG"])
        self.assertLess(pred_fatigued["xA"], pred_rested["xA"])
        self.assertGreater(pred_opp_b2b["xG"], pred_rested["xG"])
        self.assertGreater(pred_opp_b2b["xA"], pred_rested["xA"])

    def test_toi_fallback_incomplete_data_and_pp_distribution(self):
        # Test fallback when skater data has partial stats and missing TOI
        skaters = [
            {"playerId": 301, "firstName": {"default": "Forward"}, "lastName": {"default": "One"}, "positionCode": "C"},
            {"playerId": 302, "firstName": {"default": "Forward"}, "lastName": {"default": "Two"}, "positionCode": "L"},
            {"playerId": 303, "firstName": {"default": "Forward"}, "lastName": {"default": "Three"}, "positionCode": "R"},
            {"playerId": 304, "firstName": {"default": "Forward"}, "lastName": {"default": "Four"}, "positionCode": "C"},
            {"playerId": 305, "firstName": {"default": "Forward"}, "lastName": {"default": "Five"}, "positionCode": "L"},
            {"playerId": 306, "firstName": {"default": "Forward"}, "lastName": {"default": "Six"}, "positionCode": "R"},
            {"playerId": 401, "firstName": {"default": "Defense"}, "lastName": {"default": "One"}, "positionCode": "D"},
            {"playerId": 402, "firstName": {"default": "Defense"}, "lastName": {"default": "Two"}, "positionCode": "D"},
            {"playerId": 403, "firstName": {"default": "Defense"}, "lastName": {"default": "Three"}, "positionCode": "D"},
        ]

        # Only provide TOI for some skaters, others missing/zero
        toi_rows = [
            {"playerId": 301, "evTimeOnIcePerGame": 1200.0, "ppTimeOnIcePerGame": 240.0, "timeOnIcePerGame": 1440.0},
            {"playerId": 302, "evTimeOnIcePerGame": 1150.0, "ppTimeOnIcePerGame": 220.0, "timeOnIcePerGame": 1370.0},
            {"playerId": 303, "evTimeOnIcePerGame": 1100.0, "ppTimeOnIcePerGame": 210.0, "timeOnIcePerGame": 1310.0},
            {"playerId": 304, "evTimeOnIcePerGame": 1000.0, "ppTimeOnIcePerGame": 190.0, "timeOnIcePerGame": 1190.0},
            {"playerId": 305, "evTimeOnIcePerGame": 900.0, "ppTimeOnIcePerGame": 100.0, "timeOnIcePerGame": 1000.0},
            # 306 has 0 TOI / missing
            {"playerId": 401, "evTimeOnIcePerGame": 1300.0, "ppTimeOnIcePerGame": 250.0, "timeOnIcePerGame": 1550.0},
            {"playerId": 402, "evTimeOnIcePerGame": 1200.0, "ppTimeOnIcePerGame": 50.0, "timeOnIcePerGame": 1250.0},
        ]
        toi_resp = MagicMock(status_code=200)
        toi_resp.json.return_value = {"data": toi_rows}

        def fake_retry(url, *args, **kwargs):
            if "timeonice" in url:
                return toi_resp
            return MagicMock(status_code=500)

        with patch.object(self.client, "request_with_retry", side_effect=fake_retry), \
             patch.object(self.client, "_fetch_team_skaters", return_value=skaters):
            lines = self.client.get_team_lines("BOS", force=True)

        self.assertIsNotNone(lines)
        # Check top PP unit has exactly 4 forwards and 1 defenseman
        f_pp1 = [(k, v) for k, v in lines.items() if isinstance(k, tuple) and k[1] == "F" and v.get("pp_unit") == "PP1"]
        d_pp1 = [(k, v) for k, v in lines.items() if isinstance(k, tuple) and k[1] == "D" and v.get("pp_unit") == "PP1"]
        self.assertEqual(len(f_pp1), 4)
        self.assertEqual(len(d_pp1), 1)
        self.assertEqual(d_pp1[0][0][0], "defense one")

        # Check rookie / 0 TOI skater did not crash and was placed on lower line
        f_six = lines.get(("forward six", "F"))
        self.assertIsNotNone(f_six)
        self.assertEqual(f_six["ev_line"], "L2")

    def test_monte_carlo_distribution_mathematical_invariants(self):
        lineup = [
            {"name": "McDavid", "team": "EDM", "pos": "F", "ev_line": "L1", "pp_unit": "PP1", "xRRS": 12.0, "score": 24.0, "slot_boost": 2.0},
            {"name": "Draisaitl", "team": "EDM", "pos": "F", "ev_line": "L1", "pp_unit": "PP1", "xRRS": 10.0, "score": 18.0, "slot_boost": 1.8},
            {"name": "Bouchard", "team": "EDM", "pos": "D", "ev_line": "D1", "pp_unit": "PP1", "xRRS": 8.0, "score": 12.8, "slot_boost": 1.6},
            {"name": "MacKinnon", "team": "COL", "pos": "F", "ev_line": "L1", "pp_unit": "PP1", "xRRS": 11.0, "score": 15.4, "slot_boost": 1.4},
            {"name": "Makar", "team": "COL", "pos": "D", "ev_line": "D1", "pp_unit": "PP1", "xRRS": 9.0, "score": 10.8, "slot_boost": 1.2},
        ]
        # 1. Verify correlation matrix properties
        R = build_lineup_correlation_matrix(lineup)
        # Symmetry
        self.assertTrue(np.allclose(R, R.T))
        # Diagonal is identically 1.0
        self.assertTrue(np.allclose(np.diag(R), np.ones(5)))
        # Strictly positive eigenvalues
        eigvals = np.linalg.eigvalsh(R)
        self.assertTrue(all(ev > 0.0 for ev in eigvals))
        # Off-diagonal elements bounded in [0, 1)
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertGreaterEqual(R[i, j], 0.0)
                    self.assertLess(R[i, j], 1.0)

        # 2. Verify Monte Carlo monotonicity invariant: Floor_25 <= Mean <= Ceil_90 <= Ceil_95
        dist = simulate_lineup_distribution(lineup, n_sims=5000, seed=42)
        mean_score = dist["mean"]
        floor_25 = dist["floor_25"]
        ceil_90 = dist["ceiling_90"]
        ceil_95 = dist["ceiling_95"]

        self.assertLessEqual(floor_25, mean_score)
        self.assertLessEqual(mean_score, ceil_90)
        self.assertLessEqual(ceil_90, ceil_95)

        # Variance of stacked EDM trio should be non-zero
        self.assertGreater(dist["std"], 0.0)

    def test_no_look_ahead_bias_stress_test(self):
        # Populate historical database with past logs for a player
        p_id = 999
        self.db.save_player(p_id, "Test Skater", "TOR", "C")

        past_logs = [
            {"gameId": 101, "gameDate": "2026-10-10", "goals": 1, "assists": 1, "shots": 4, "toi": "20:00"},
            {"gameId": 102, "gameDate": "2026-10-12", "goals": 0, "assists": 2, "shots": 3, "toi": "19:30"},
            {"gameId": 103, "gameDate": "2026-10-14", "goals": 2, "assists": 0, "shots": 5, "toi": "21:00"},
        ]
        self.db.save_game_logs(p_id, past_logs)

        # Target split date: 2026-10-15
        predictor = PerformancePredictor(model_type="poisson")
        pred_before = predictor.predict(
            past_logs, is_home=True, opponent_abbrev="MTL", game_date="2026-10-15",
            is_b2b=False, opp_is_b2b=False
        )

        # Now inject FUTURE game logs (e.g. 2026-10-25) into the database
        future_logs = [
            {"gameId": 201, "gameDate": "2026-10-25", "goals": 5, "assists": 5, "shots": 10, "toi": "25:00"},
            {"gameId": 202, "gameDate": "2026-10-28", "goals": 4, "assists": 4, "shots": 12, "toi": "26:00"},
        ]
        self.db.save_game_logs(p_id, future_logs)

        # Query player logs strictly before 2026-10-15
        all_logs = self.db.get_game_logs(p_id)
        point_in_time_logs = [
            g for g in all_logs
            if (g.get("gameDate") or g.get("game_date") or "") < "2026-10-15"
        ]
        self.assertEqual(len(point_in_time_logs), 3)

        # Predictions using point-in-time filtered logs remain 100% invariant
        pred_after = predictor.predict(
            point_in_time_logs, is_home=True, opponent_abbrev="MTL", game_date="2026-10-15",
            is_b2b=False, opp_is_b2b=False
        )
        self.assertEqual(pred_before["xG"], pred_after["xG"])
        self.assertEqual(pred_before["xA"], pred_after["xA"])
        self.assertEqual(pred_before["xSOG"], pred_after["xSOG"])
        self.assertEqual(pred_before["xRRS"], pred_after["xRRS"])

    def test_monte_carlo_mean_preservation_and_empirical_cv(self):
        from optimizer import get_player_empirical_cv, simulate_lineup_distribution

        # Role-based empirical CV differences
        cv_star = get_player_empirical_cv({"pos": "F", "ev_line": "L1", "pp_unit": "PP1"})
        cv_depth = get_player_empirical_cv({"pos": "F", "ev_line": "L4", "pp_unit": None})
        self.assertLess(cv_star, cv_depth)
        self.assertEqual(cv_star, 0.48)
        self.assertEqual(cv_depth, 0.95)

        cv_d1 = get_player_empirical_cv({"pos": "D", "ev_line": "D1", "pp_unit": "PP1"})
        cv_d3 = get_player_empirical_cv({"pos": "D", "ev_line": "D3", "pp_unit": None})
        self.assertLess(cv_d1, cv_d3)
        self.assertEqual(cv_d1, 0.54)
        self.assertEqual(cv_d3, 0.90)

        # Mean preservation of log-normal copula simulation
        lineup = [
            {"name": "Star 1", "pos": "F", "ev_line": "L1", "pp_unit": "PP1", "xRRS": 7.5, "slot_boost": 2.0, "booster": 0.5},
            {"name": "Star 2", "pos": "F", "ev_line": "L1", "pp_unit": "PP1", "xRRS": 6.8, "slot_boost": 1.8, "booster": 0.4},
            {"name": "Mid 1", "pos": "F", "ev_line": "L2", "pp_unit": "PP2", "xRRS": 4.5, "slot_boost": 1.6, "booster": 0.2},
            {"name": "Def 1", "pos": "D", "ev_line": "D1", "pp_unit": "PP1", "xRRS": 4.2, "slot_boost": 1.4, "booster": 0.0},
            {"name": "Def 2", "pos": "D", "ev_line": "D2", "pp_unit": None, "xRRS": 2.8, "slot_boost": 1.2, "booster": 0.0},
        ]
        expected_payout = sum(p["xRRS"] * (p["slot_boost"] + p["booster"]) for p in lineup)

        sim = simulate_lineup_distribution(lineup, n_sims=10000, seed=42)
        # Expected simulated mean matches analytical expected payout within Monte-Carlo tolerance (< 0.2 RRS)
        self.assertAlmostEqual(sim["mean"], expected_payout, delta=0.20)
        self.assertGreater(sim["ceiling_90"], sim["mean"])
        self.assertLess(sim["floor_25"], sim["mean"])

    def test_glm_shrinkage_and_physiological_caps(self):
        predictor = PerformancePredictor(model_type="poisson")
        # Extreme player stats to test physiological capping
        extreme_logs = [
            {"gameId": 300 + i, "gameDate": f"2026-10-{i+1:02d}", "goals": 10, "assists": 10, "shots": 20, "toi": "35:00"}
            for i in range(10)
        ]
        standings = {"MTL": {"defense_factor": 1.5, "pk_pct": 0.60, "times_shorthanded_per_game": 5.0}}
        predictor.train(extreme_logs, standings)

        pred = predictor.predict(
            extreme_logs, is_home=True, opponent_abbrev="MTL", game_date="2026-10-15",
            position="F", ev_line="L1", pp_unit="PP1"
        )

        # Enforced physiological bounds
        self.assertLessEqual(pred["xG"], 1.10)
        self.assertLessEqual(pred["xA"], 1.70)
        self.assertLessEqual(pred["xSOG"], 6.50)
        toi_mins = float(pred["xTOI"].split(":")[0]) + float(pred["xTOI"].split(":")[1]) / 60.0
        self.assertLessEqual(toi_mins, 28.0)
        self.assertLessEqual(pred["xRRS"], 16.0)

    def test_numpy_spearmanr_accuracy_and_ties(self):
        from model import _numpy_spearmanr, _rankdata_numpy
        a = [1.0, 2.0, 2.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]

        ranks = _rankdata_numpy(a)
        self.assertEqual(list(ranks), [1.0, 2.5, 2.5, 4.0, 5.0])

        res = _numpy_spearmanr(a, b)
        self.assertGreater(res.statistic, 0.95)
        self.assertAlmostEqual(res.statistic, res.correlation, places=6)

    def test_schedule_context_fallback_to_game_logs(self):
        # Empty schedule table, verify fallback to game_logs
        p_id = 9901
        self.db.save_player(p_id, "Connor McDavid", team="EDM", position="C")
        self.db.save_game_logs(p_id, [
            {"gameId": 501, "gameDate": "2026-10-14", "goals": 1, "assists": 2, "shots": 4, "toi": "22:00", "is_home": 1, "opponentAbbrev": "VAN"}
        ])

        ctx = self.db.get_schedule_context("EDM", "CGY", "2026-10-15")
        self.assertEqual(ctx["rest_days"], 1.0)
        self.assertEqual(ctx["is_b2b"], 1)

    def test_save_game_logs_prior_batch_rest_calculation(self):
        p_id = 9902
        self.db.save_player(p_id, "Leon Draisaitl", team="EDM", position="C")

        # First batch ends on 2026-10-10
        batch1 = [
            {"gameId": 601, "gameDate": "2026-10-10", "goals": 0, "assists": 1, "shots": 3, "toi": "20:00"}
        ]
        self.db.save_game_logs(p_id, batch1)

        # Second batch starts on 2026-10-11 (B2B across batch boundary)
        batch2 = [
            {"gameId": 602, "gameDate": "2026-10-11", "goals": 1, "assists": 1, "shots": 4, "toi": "21:00"}
        ]
        self.db.save_game_logs(p_id, batch2)

        logs = self.db.get_game_logs(p_id)
        g2 = [g for g in logs if (g.get("gameId") or g.get("game_id")) == 602][0]
        self.assertEqual(g2["rest_days"], 1.0)
        self.assertEqual(g2["is_b2b"], 1)

    def test_nhl_api_http_adapter_retry_config(self):
        session = self.client._create_session()
        adapter = session.adapters.get("https://")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.max_retries.total, 0)
        self.assertFalse(adapter.max_retries.raise_on_status)


class TestLineupHistoryPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp_dir, "history.db"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _lineup(names):
        return [
            {
                "slot": index + 1,
                "slot_boost": 2.0 - 0.2 * index,
                "name": name,
                "team": "COL",
                "opp": "vs BOS",
                "ev_line": "L1",
                "pp_unit": "PP1",
                "booster": 0.1,
                "total_boost": 2.1 - 0.2 * index,
                "xRRS": 5.0,
                "score": 10.0,
            }
            for index, name in enumerate(names)
        ]

    def test_resaving_lineup_preserves_settled_results(self):
        self.db.save_lineup_history("2026-10-20", "standard", self._lineup(["A", "B", "C", "D", "E"]))
        entry = self.db.get_lineup_for_date("2026-10-20", "standard")[0]
        self.db.settle_history_player(entry["id"], 1.0, 0.0, 3.0, "18:00", 3.0, 6.0)
        self.db.record_app_results([{"id": entry["id"], "app_rrs": 5.5, "app_payout": 11.0}])

        self.db.save_lineup_history("2026-10-20", "standard", self._lineup(["A", "B", "C", "D"]))

        rows = self.db.get_lineup_for_date("2026-10-20", "standard")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["settled"], 1)
        self.assertEqual(rows[0]["actual_goals"], 1.0)
        self.assertEqual(rows[0]["app_rrs"], 5.5)
        self.assertEqual(rows[0]["app_payout"], 11.0)


if __name__ == "__main__":
    unittest.main()


