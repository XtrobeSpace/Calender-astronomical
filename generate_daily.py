#!/usr/bin/env python3
"""
XTROBE Calendar Engine — Fast Daily Dynamic Sync v5.5

Runs daily on GitHub Actions (takes <10 seconds for cached queries).
Queries open APIs for 4 distinct dynamic event categories:
  1. Near-Earth Asteroid Flybys (NASA JPL CAD API)
  2. Active Bright Comets (COBS Planner API & MPC Catalog)
  3. Space Weather Alerts & Aurora Warnings (NOAA SWPC Open API)
  4. Atmospheric Fireball Detections (NASA JPL CNEOS Fireball API)

Usage:
  python generate_daily.py              # Default: 60-day window, dist_max=20.0 LD
  python generate_daily.py --days 365   # Full-year dynamic sync (~300+ events)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

# ── Shared Library ────────────────────────────────────────────
from shared import (
    OUTPUT_DIR, fetch_asteroids_jpl_cad, fetch_cobs_comets,
    fetch_mpc_comets, fetch_nasa_fireballs, fetch_noaa_space_weather,
    init_skyfield, load_resolved_cache, log, resolve_asteroid_position,
    save_resolved_cache
)

def main() -> None:
    p = argparse.ArgumentParser(description="XTROBE Daily Dynamic Event Sync v5.5")
    p.add_argument("--days", type=int, default=60, help="Number of days to query for asteroid flybys (default: 60)")
    p.add_argument("--dist-max", type=float, default=20.0, help="Maximum distance in Lunar Distances (default: 20.0 LD)")
    args = p.parse_args()

    log.info(f"=== STARTING FULL DYNAMIC SYNC (Window: {args.days} Days, Max Distance: {args.dist_max} LD) ===")
    now_utc = datetime.now(timezone.utc)
    ts, eph = init_skyfield()
    cache = load_resolved_cache()

    # 1. Fetch Dynamic Near-Earth Asteroid Flybys (NASA JPL CAD)
    date_min = now_utc.strftime("%Y-%m-%d")
    date_max = (now_utc + timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    asteroids = fetch_asteroids_jpl_cad(date_min, date_max, dist_max_ld=args.dist_max)
    resolved_asteroids = []
    for ast in asteroids:
        try:
            resolved_asteroids.append(resolve_asteroid_position(ast, ts, eph, cache))
        except Exception as exc:
            log.warning(f"Asteroid cascade failed for {ast.get('des','?')}: {exc}")

    # 2. Fetch Dynamic Bright Comets (COBS + MPC Fallback)
    cobs_comets = fetch_cobs_comets(now_utc, cache, mag_limit=12.0)
    cobs_ids = {ev["id"] for ev in cobs_comets}
    mpc_comets = fetch_mpc_comets(cobs_ids, ts, eph, now_utc, cache, mag_limit=12.0)
    active_comets = cobs_comets + mpc_comets

    # 3. Fetch Live Space Weather & Aurora Alerts (NOAA SWPC)
    space_weather_events = fetch_noaa_space_weather()

    # 4. Fetch Recent Atmospheric Fireballs / Bolides (NASA JPL CNEOS)
    fireball_events = fetch_nasa_fireballs(limit=10)

    save_resolved_cache(cache)
    dynamic_events = resolved_asteroids + active_comets + space_weather_events + fireball_events
    dynamic_events.sort(key=lambda e: e.get("utc", ""))

    log.info(
        f"Dynamic Sync Summary: {len(resolved_asteroids)} Asteroids, "
        f"{len(active_comets)} Comets, {len(space_weather_events)} Space Weather Alerts, "
        f"{len(fireball_events)} Fireball Detections."
    )

    daily_doc = {
        "schema":             "xtrobe-daily-5.5",
        "synced_at":          now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year":               now_utc.year,
        "window_days":        args.days,
        "dist_max_ld":        args.dist_max,
        "count":              len(dynamic_events),
        "categories":         {
            "asteroid":       len(resolved_asteroids),
            "comet":          len(active_comets),
            "space_weather":  len(space_weather_events),
            "fireball":       len(fireball_events)
        },
        "events":             dynamic_events,
    }

    daily_path = os.path.join(OUTPUT_DIR, "daily_events.json")

    with open(daily_path, "w", encoding="utf-8") as f:
        json.dump(daily_doc, f, separators=(",", ":"))

    log.info(f"Saved: {daily_path} ({os.path.getsize(daily_path)/1024:.1f} KB)")
    log.info("=== FULL DYNAMIC SYNC COMPLETE ===")

if __name__ == "__main__":
    main()
