import argparse
from typing import Optional

from cli_utils import parse_iso_date_arg
from db import Database
from nhl_api import nhl_today


def show_history_report(db: Database, target_date: Optional[str] = None):
    if target_date:
        details = db.get_history_slate_details(target_date)
        if not details:
            print(f"No history records found for date: {target_date}")
            return

        table_width = 126
        print("\n" + "=" * table_width)
        print(f"RRS Engine - Lineup Performance Detail ({target_date})")
        print("=" * table_width)
        print(f"{'Mode':<10} {'Slot':<5} {'Player':<20} {'Team':<5} {'Line':<7} {'Boost':<6} {'xRRS':<6} {'Act xR':<8} {'App RRS':<8} {'Proj RRS':<9} {'Act RRS':<10} {'App Payout':<11} {'Stats'}")
        print("-" * table_width)

        for d in details:
            mode_str = d["mode"].capitalize()[:10]
            slot_str = f"#{d['slot']}"
            p_name = d["player_name"][:19]
            team_str = d["team"][:4]
            ev_line = d.get("ev_line")
            pp_unit = d.get("pp_unit")
            if ev_line and pp_unit:
                line_str = f"{ev_line}/{pp_unit}"
            elif ev_line:
                line_str = ev_line
            elif pp_unit:
                line_str = pp_unit
            else:
                line_str = "--"
            tot_b = f"{d['total_boost']:.1f}x"
            p_base = d.get('proj_xrrs', d.get('proj_base_pts', 0.0))
            proj_base = f"{p_base:.1f}"
            app_rrs_str = f"{d['app_rrs']:.1f}" if d.get('app_rrs') is not None else "--"
            app_payout_str = f"{d['app_payout']:.1f} RRS" if d.get('app_payout') is not None else "--"

            if d["settled"]:
                a_base = d.get('actual_xrrs', d.get('actual_base_pts', 0.0))
                act_base = f"{a_base:.1f}"
                proj_rrs = f"{d.get('proj_rrs', d.get('proj_score', 0.0)):.1f} RRS"
                act_rrs = f"{d.get('actual_rrs', d.get('actual_score', 0.0)):.1f} RRS"
                stats_str = f"{int(d['actual_goals'])}G, {int(d['actual_assists'])}A, {int(d['actual_shots'])}S ({d.get('actual_toi', '')})"
            else:
                act_base = "Pending"
                proj_rrs = f"{d.get('proj_rrs', d.get('proj_score', 0.0)):.1f} RRS"
                act_rrs = "Pending"
                stats_str = "--"

            print(f"{mode_str:<10} {slot_str:<5} {p_name:<20} {team_str:<5} {line_str:<7} {tot_b:<6} {proj_base:<6} {act_base:<8} {app_rrs_str:<8} {proj_rrs:<9} {act_rrs:<10} {app_payout_str:<11} {stats_str}")

        print("-" * table_width)
        has_app_data = any(d.get("app_rrs") is not None for d in details)
        if has_app_data:
            app_payouts = [d["app_payout"] for d in details if d.get("app_payout") is not None]
            app_rrs_list = [d["app_rrs"] for d in details if d.get("app_rrs") is not None]
            tot_app_payout = sum(app_payouts) if app_payouts else None
            payout_info = f" | Total App Payout: {tot_app_payout:.1f} RRS" if tot_app_payout is not None else ""
            print(f"Ground Truth Tracking: Recorded App RRS for {len(app_rrs_list)}/{len(details)} players{payout_info}")
            if tot_app_payout is not None:
                tot_proj = sum(d.get("proj_rrs", 0.0) for d in details if d.get("app_payout") is not None)
                bias = tot_app_payout - tot_proj
                print(f"Model vs Real App Bias: {'+' if bias >= 0 else ''}{bias:.2f} RRS")
        print("=" * table_width + "\n")
        return

    slates = db.get_history_slates()
    if not slates:
        print("No history records recorded yet. Run without arguments to generate and track today's lineups.")
        return

    table_width = 100
    print("\n" + "=" * table_width)
    print("RRS Engine - Performance & Rating Tracking Summary")
    print("=" * table_width)
    print(f"{'Date':<12} {'Mode':<10} {'Status':<9} {'Proj RRS':<11} {'Act RRS':<11} {'App Payout':<12} {'Act Diff':<11} {'App Bias'}")
    print("-" * table_width)

    total_actual_rrs = 0.0
    settled_slates = 0
    total_app_payout_sum = 0.0
    app_payout_slates = 0

    for s in slates:
        date_str = s["game_date"]
        mode_str = s["mode"].capitalize()
        status_str = "Settled" if s["all_settled"] else "Pending"
        p_rrs = s.get('total_proj_rrs', s.get('total_proj_score', 0.0))
        proj_rrs = f"{p_rrs:.1f} RRS"

        if s["all_settled"]:
            a_rrs = s.get('total_actual_rrs', s.get('total_actual_score', 0.0))
            act_rrs = f"{a_rrs:.1f} RRS"
            diff_rrs = a_rrs - p_rrs
            diff_str = f"{'+' if diff_rrs >= 0 else ''}{diff_rrs:.1f} RRS"
            total_actual_rrs += a_rrs
            settled_slates += 1
        else:
            act_rrs = "Pending"
            diff_str = "--"

        if s.get("app_payout_count", 0) > 0:
            app_p = s.get("total_app_payout", 0.0)
            app_payout_str = f"{app_p:.1f} RRS"
            total_app_payout_sum += app_p
            app_payout_slates += 1
        else:
            app_payout_str = "--"

        if s.get("avg_app_bias") is not None:
            b_val = s["avg_app_bias"]
            app_bias_str = f"{'+' if b_val >= 0 else ''}{b_val:.2f} RRS"
        else:
            app_bias_str = "--"

        print(f"{date_str:<12} {mode_str:<10} {status_str:<9} {proj_rrs:<11} {act_rrs:<11} {app_payout_str:<12} {diff_str:<11} {app_bias_str}")

    print("-" * table_width)
    if settled_slates > 0 or app_payout_slates > 0:
        summary_line = f"Total Settled: {settled_slates} slates | Cumulative Payout (Heuristic): {total_actual_rrs:.1f} RRS"
        if total_app_payout_sum > 0:
            summary_line += f" | App Ground Truth: {total_app_payout_sum:.1f} RRS"
        print(summary_line)
        bias_summary = db.get_role_bias_summary()
        pp1_diffs = bias_summary.get("pp1_diffs", [])
        other_diffs = bias_summary.get("other_diffs", [])
        if pp1_diffs or other_diffs:
            pp1_str = f"{sum(pp1_diffs)/len(pp1_diffs):+.2f} RRS (n={len(pp1_diffs)})" if pp1_diffs else "N/A"
            other_str = f"{sum(other_diffs)/len(other_diffs):+.2f} RRS (n={len(other_diffs)})" if other_diffs else "N/A"
            print(f"Role Bias Tracking: PP1 Avg Bias: {pp1_str} | Non-PP1 Avg Bias: {other_str}")
    else:
        print("Games are currently in progress or upcoming. Check back after matches finish for settled RRS payout.")
    print("=" * table_width + "\n")


def handle_record_app_scores(db: Database, record_arg: str):
    """
    Interactively or in batch records actual Real Sports mobile app ratings (app_rrs)
    and payouts (app_payout) into lineup_history for ground-truth model calibration.
    """
    try:
        if ":" in record_arg:
            date_part, scores_part = record_arg.split(":", 1)
            target_date = parse_iso_date_arg(date_part.strip())
            score_tokens = [s.strip() for s in scores_part.split(",") if s.strip()]
        else:
            target_date = parse_iso_date_arg(record_arg.strip()) if record_arg != "today" else nhl_today()
            score_tokens = []
    except (argparse.ArgumentTypeError, ValueError) as e:
        print(f"Error: {e}")
        return

    lineup = db.get_lineup_for_date(target_date, mode="standard")
    if not lineup:
        all_lineups = db.get_lineup_for_date(target_date)
        if all_lineups:
            lineup = [p for p in all_lineups if p.get("mode") == all_lineups[0].get("mode")]

    if not lineup:
        print(f"\nNo lineup history found for date: {target_date}. Generate lineups first before recording results.\n")
        return

    updates = []
    print(f"\nRecording Real Sports app ratings for {target_date}:")
    print("-" * 75)
    for idx, p in enumerate(lineup):
        p_name = p.get("player_name", "Unknown")
        slot = p.get("slot", idx + 1)
        proj_r = p.get("proj_rrs", 0.0)
        tot_b = p.get("total_boost", p.get("slot_boost", 1.0) + p.get("booster", 0.0))

        app_rrs_val = None
        app_payout_val = None

        if idx < len(score_tokens):
            token = score_tokens[idx]
            if "=" in token:
                parts = token.split("=")
                try:
                    app_rrs_val = float(parts[0].strip())
                    app_payout_val = float(parts[1].strip())
                except ValueError:
                    pass
            else:
                try:
                    app_rrs_val = float(token.strip())
                    # Do not auto-calculate payout with additive assumption to prevent circular stacking verification
                    app_payout_val = None
                except ValueError:
                    pass
        else:
            try:
                prompt_str = f"Slot #{slot} | {p_name:<20} (Proj: {proj_r:>4.1f} RRS, Boost: {tot_b:.1f}x) -> App RRS (or rrs=payout): "
                val_in = input(prompt_str).strip()
                if val_in:
                    if "=" in val_in:
                        parts = val_in.split("=")
                        app_rrs_val = float(parts[0].strip())
                        app_payout_val = float(parts[1].strip())
                    else:
                        app_rrs_val = float(val_in)
                        app_payout_val = None
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled.")
                break
            except ValueError:
                print(f"Skipping invalid entry for {p_name}.")
                continue

        if app_rrs_val is not None:
            updates.append({
                "id": p["id"],
                "app_rrs": app_rrs_val,
                "app_payout": app_payout_val
            })
            payout_str = f" (Payout = {app_payout_val:.1f} RRS)" if app_payout_val is not None else ""
            print(f"  Slot #{slot} {p_name:<20}: App RRS = {app_rrs_val:.1f}{payout_str}")

    if updates:
        db.record_app_results(updates)
        print("-" * 75)
        print(f"Successfully recorded Real App scores for {len(updates)} players on {target_date}.\n")
