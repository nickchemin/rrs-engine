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
    name_key,
    parse_booster_val,
    normalize_booster_name,
    resolve_booster,
    get_player_booster,
    load_boosters_file,
    match_injury,
    solve_optimal_lineup,
    solve_stacked_lineup,
    assign_slots,
    print_lineup_table,
    settle_past_lineups
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


class TestBoosters(unittest.TestCase):
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

    def test_normalize_booster_name(self):
        self.assertEqual(normalize_booster_name("Elias Pettersson (D)"), "elias pettersson (D)")
        self.assertEqual(normalize_booster_name("Elias Pettersson #25"), "elias pettersson (D)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (C)"), "elias pettersson (C)")
        self.assertEqual(normalize_booster_name("Elias Pettersson"), "elias pettersson (C)")
        self.assertEqual(normalize_booster_name("Elias Pettersson (F)"), "elias pettersson (C)")
        self.assertEqual(normalize_booster_name("[DTD] Cale Makar (DEF)"), "cale makar (D)")
        self.assertEqual(normalize_booster_name("Tim Stützle (C)"), "tim stutzle (C)")
        self.assertEqual(normalize_booster_name("Leo Carlsson"), "leo carlsson")

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


class TestGoaliesAndInjuries(unittest.TestCase):
    def test_team_slug_map_utah(self):
        self.assertEqual(TEAM_SLUG_MAP.get("utah-mammoth"), "UTA")
        self.assertEqual(TEAM_SLUG_MAP.get("utah-hockey-club"), "UTA")

    def test_compute_goalie_factor(self):
        self.assertEqual(compute_goalie_factor(0.0), 1.0)
        self.assertGreaterEqual(compute_goalie_factor(0.880), 1.0)
        self.assertLessEqual(compute_goalie_factor(0.930), 1.0)

    def test_match_injury_espn_mapping(self):
        injuries = {
            "Jack Hughes": {"team": "NJ", "status": "Out", "detail": "Upper Body"},
            "Victor Hedman": {"team": "TB", "status": "Day-To-Day", "detail": "Lower Body"}
        }
        inj = match_injury(injuries, "Jack Hughes", "NJD", is_duplicate=False)
        self.assertIsNotNone(inj)
        self.assertEqual(inj["status"], "Out")

        inj2 = match_injury(injuries, "Victor Hedman", "TBL", is_duplicate=False)
        self.assertIsNotNone(inj2)
        self.assertEqual(inj2["status"], "Day-To-Day")

    def test_goalie_factor_in_prediction(self):
        pred = PerformancePredictor()
        # Mock ML models as trained
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0]
        pred.models = {
            "goals": mock_model,
            "assists": mock_model,
            "shots": mock_model,
            "toi": mock_model
        }
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
        # Shots should NOT be multiplied by goalie_factor
        self.assertEqual(res_weak_goalie["xSOG"], res_normal_goalie["xSOG"])


class TestOptimizer(unittest.TestCase):
    def test_solve_optimal_lineup_dp(self):
        # Generate 15 candidates
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

        # Slots must be in order 1 to 5 with correct descending multipliers
        for idx, item in enumerate(lineup):
            self.assertEqual(item["slot"], idx + 1)
            self.assertEqual(item["slot_boost"], [2.0, 1.8, 1.6, 1.4, 1.2][idx])

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
        # Should detect stacks
        stacks = [p.get("stack") for p in lineup if p.get("stack")]
        self.assertTrue(any("Pair" in s or "x" in s for s in stacks))

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
        # 0 points, shots and blocks
        # raw = 2 * 0.25 + 1 * 0.50 = 1.0
        score_def = calculate_rrs_from_stats(goals=0, assists=0, shots=2, blocks=1)
        self.assertEqual(score_def, 1.0)

        # 1 Goal: 2.5 + shots*0.25 + blocks*0.50
        score_goal = calculate_rrs_from_stats(goals=1, assists=0, shots=3, blocks=0)
        self.assertAlmostEqual(score_goal, 3.25, places=1)

    def test_settle_past_lineups_dnp(self):
        lineup = [
            {"id": 101, "name": "Scratched Player", "team": "BOS", "opp": "vsTOR", "slot": 1, "slot_boost": 2.0, "total_boost": 2.0, "xRRS": 3.0, "score": 6.0, "booster": 0.0}
        ]
        self.db.save_player(101, "Scratched Player", "BOS", "F")
        self.db.save_lineup_history("2026-10-10", "standard", lineup)

        client = MagicMock()
        # Returns current logs with games that are newer than 2026-10-10, meaning player was DNP on 10-10
        client.get_player_current_game_logs.return_value = [
            {"gameDate": "2026-10-12", "goals": 1, "assists": 0, "shots": 2, "gameId": 9999}
        ]

        settle_past_lineups(client, self.db)
        details = self.db.get_history_slate_details("2026-10-10")
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["settled"], 1)
        self.assertEqual(details[0]["actual_rrs"], 0.0)
        self.assertEqual(details[0]["actual_toi"], "DNP")



class TestNHLClientFeatures(unittest.TestCase):
    def test_get_boxscore_skaters(self):
        client = NHLClient()
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
        with patch.object(client.session, "get", return_value=mock_response):
            skaters = client.get_boxscore_skaters(2024020100)
            self.assertEqual(len(skaters), 3)
            self.assertEqual(skaters[8478402]["goals"], 1)
            self.assertEqual(skaters[8478402]["assists"], 2)
            self.assertEqual(skaters[8478402]["sog"], 4)
            self.assertEqual(skaters[8478402]["blockedShots"], 1)
            self.assertEqual(skaters[8480069]["blockedShots"], 3)

    def test_get_boxscore_skaters_not_final(self):
        client = NHLClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"gameState": "LIVE"}
        with patch.object(client.session, "get", return_value=mock_response):
            skaters = client.get_boxscore_skaters(2024020101)
            self.assertEqual(skaters, {})

    def test_get_all_club_stats_roster_filter_and_deduplication(self):
        client = NHLClient()
        # Traded player (playerId 8478402) is present in BUF club stats and VGK club stats
        # but BUF roster has player 8478402, VGK roster does not
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
                return [{"id": 8478403}]  # Jack Eichel left BUF
            elif team == "VGK":
                return [{"id": 8478402}, {"id": 8478404}]  # Jack Eichel is on VGK roster
            return []

        with patch.object(client, "get_club_stats", side_effect=mock_club_stats), \
             patch.object(client, "get_team_roster", side_effect=mock_roster):
            results = client.get_all_club_stats(["BUF", "VGK"])
            buf_pids = [p["playerId"] for p in results.get("BUF", [])]
            vgk_pids = [p["playerId"] for p in results.get("VGK", [])]

            # Player 8478402 filtered out of BUF because not on BUF roster
            self.assertNotIn(8478402, buf_pids)
            # Player 8478402 is present in VGK
            self.assertIn(8478402, vgk_pids)


if __name__ == "__main__":
    unittest.main()
