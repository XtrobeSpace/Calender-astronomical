#!/usr/bin/env python3
"""
XTROBE Calendar Engine — Shared Utilities & Cascades v5.4

Common mathematical helpers, Skyfield initialization, caching, 
robust API schema extractors, retry HTTP sessions, catalog data,
embedded IMO ZHR forecasts, and multi-tier fallback cascades for:
  - Asteroid Telemetry (NASA JPL CAD)
  - Comet Telemetry (COBS + MPC)
  - Live Space Weather Alerts (NOAA SWPC)
  - Atmospheric Fireballs (NASA JPL CNEOS)
"""

import io
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Skyfield ──────────────────────────────────────────────────
from skyfield.api import Loader, Star, wgs84
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.data import mpc as sk_mpc
from skyfield.data.spice import inertial_frames
from skyfield.keplerlib import _KeplerOrbit

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xtrobe_shared")

# ── Base Paths ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
SKYFIELD_CACHE = os.path.join(BASE_DIR, "skyfield_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SKYFIELD_CACHE, exist_ok=True)

# ── Constants & Catalogs ──────────────────────────────────────
AU_KM = 1.495978707e8
LD_KM = 384_400.0

# API Endpoints (All 100% Free, Keyless, Open Endpoints)
CAD_API       = "https://ssd-api.jpl.nasa.gov/cad.api"
SBDB_API      = "https://ssd-api.jpl.nasa.gov/sbdb.api"
HORIZONS      = "https://ssd.jpl.nasa.gov/api/horizons.api"
FIREBALL_API  = "https://ssd-api.jpl.nasa.gov/fireball.api"
COBS_API      = "https://cobs.si/api/planner.api"
MPC_URL       = "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt"
NOAA_ALERTS   = "https://services.swpc.noaa.gov/products/alerts.json"

RESOLVED_CACHE_FILE = os.path.join(SKYFIELD_CACHE, "resolved_objects_cache.json")
MPC_CACHE_FILE      = os.path.join(SKYFIELD_CACHE, "CometEls_cache.txt")
MPC_CACHE_META      = os.path.join(SKYFIELD_CACHE, "CometEls_cache_meta.json")

# Catalogs
ALGOL = {
    "ra_h": 3.136,
    "dec_d": 40.95,
    "period_days": 2.867315,
    "t0_jd": 2447041.852,
    "mag_min": 3.4,
    "mag_max": 2.1,
}

IMO_ZHR_FORECASTS = {
  "QUA": {"zhr": 120, "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Full Moon at peak — severe interference expected"},
  "LYR": {"zhr": 18,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Favorable conditions expected"},
  "ETA": {"zhr": 50,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Southern Hemisphere observers favored"},
  "SDA": {"zhr": 25,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Best viewed from southern latitudes"},
  "PER": {"zhr": 100, "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "New Moon — IDEAL dark sky conditions"},
  "ORI": {"zhr": 25,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Parent body 1P/Halley debris stream"},
  "TAU": {"zhr": 10,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Long-duration shower, slow bright fireballs"},
  "LEO": {"zhr": 20,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Non-storm year; 55P/Tempel-Tuttle not at perihelion"},
  "GEM": {"zhr": 150, "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Favorable conditions — Gemini well-placed all night"},
  "URS": {"zhr": 10,  "zhr_confidence": "imo_confirmed_2026", "moon_note_2026": "Circumpolar for northern observers"}
}

METEOR_SHOWERS_V3 = [
    ("QUA","Quadrantids",     15.3, 49.5, 283.16, 0.5, 120, 2, 41, "2003 EH1"),
    ("LYR","Lyrids",          18.1, 33.5, 32.32,  1.0,  18, 4, 49, "C/1861 G1 Thatcher"),
    ("ETA","Eta Aquariids",   22.5, -1.0, 45.5,   3.0,  50, 4, 66, "1P/Halley"),
    ("SDA","S. Delta Aquariids",22.7,-16.0,125.0,  2.5,  25, 5, 41, "96P/Machholz"),
    ("PER","Perseids",         3.1, 58.0, 140.0,  1.5, 100, 3, 59, "109P/Swift-Tuttle"),
    ("ORI","Orionids",         6.3, 15.5, 208.0,  2.0,  25, 4, 66, "1P/Halley"),
    ("TAU","Taurids",          3.7, 14.0, 220.0,  5.0,  10,20, 27, "2P/Encke"),
    ("LEO","Leonids",         10.1, 21.5, 235.27, 0.5,  20, 2, 71, "55P/Tempel-Tuttle"),
    ("GEM","Geminids",         7.5, 32.5, 262.0,  1.0, 150, 3, 35, "3200 Phaethon"),
    ("URS","Ursids",          14.5, 75.5, 270.7,  1.0,  10, 2, 33, "8P/Tuttle"),
]

MESSIER_CATALOG = [
    ("M1","Crab Nebula","SNR",5.575,22.01,8.4,7.0),
    ("M2","","GC",21.558,-0.82,6.5,16.0),
    ("M3","","GC",13.703,28.38,6.2,18.0),
    ("M4","","GC",16.393,-26.53,5.6,36.0),
    ("M5","","GC",15.310,2.08,5.6,23.0),
    ("M6","Butterfly Cluster","OC",17.667,-32.22,4.2,25.0),
    ("M7","Ptolemy Cluster","OC",17.897,-34.82,3.3,80.0),
    ("M8","Lagoon Nebula","EN",18.063,-24.38,5.8,90.0),
    ("M13","Hercules Cluster","GC",16.695,36.46,5.8,20.0),
    ("M27","Dumbbell Nebula","PN",19.993,22.72,7.4,8.0),
    ("M31","Andromeda Galaxy","Gx",0.712,41.27,3.4,178.0),
    ("M33","Triangulum Galaxy","Gx",1.564,30.66,5.7,73.0),
    ("M42","Orion Nebula","EN",5.588,-5.39,4.0,85.0),
    ("M45","Pleiades","OC",3.790,24.12,1.6,110.0),
    ("M51","Whirlpool Galaxy","Gx",13.498,47.20,8.4,11.0),
    ("M57","Ring Nebula","PN",18.893,33.03,8.8,1.5),
    ("M81","Bode's Galaxy","Gx",9.927,69.07,6.9,21.0),
    ("M82","Cigar Galaxy","Gx",9.928,69.68,8.4,11.0),
    ("M104","Sombrero Galaxy","Gx",12.666,-11.62,8.0,9.0),
]

CALDWELL_SOUTH = [
    ("C77","Centaurus A","Gx",13.426,-43.02,6.8,21.0),
    ("C80","Omega Centauri","GC",13.447,-47.48,3.9,36.0),
    ("C92","Eta Carinae Nebula","EN",10.740,-59.87,1.0,120.0),
    ("C94","Jewel Box","OC",12.897,-60.37,4.2,10.0),
    ("C99","Coalsack Nebula","DN",12.550,-63.00,None,420.0),
    ("LMC","Large Magellanic Cloud","Irr",5.383,-69.76,0.9,650.0),
    ("SMC","Small Magellanic Cloud","Irr",0.875,-72.83,2.7,320.0),
]

# ─────────────────────────────────────────────────────────────
# 1. SKYFIELD INITIALIZATION
# ─────────────────────────────────────────────────────────────

def init_skyfield():
    """Initialize Skyfield timescale and DE440s ephemeris from local cache."""
    loader = Loader(SKYFIELD_CACHE)
    ts  = loader.timescale()
    eph = loader("de440s.bsp")
    log.info(f"Skyfield DE440s ephemeris initialized from: {SKYFIELD_CACHE}")
    return ts, eph

# ─────────────────────────────────────────────────────────────
# 2. HTTP RETRY SESSION & API ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

def create_robust_session(retries: int = 5, backoff_factor: float = 2.0) -> requests.Session:
    """
    Creates an HTTP Session with automatic retry and backoff.
    Handles Rate Limits (429), Server Errors (500, 502, 503, 504), and Forbidden (403).
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 (XTROBE Astronomy Calendar Bot)"
        )
    })
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

http = create_robust_session()

def safe_json_extract(data: dict, candidate_keys: list[str], default=None):
    """
    Robust JSON Schema Normalizer / Catcher.
    Tries multiple candidate dot-separated key paths in response JSON dictionaries
    to guard against upstream third-party schema changes.
    """
    if not isinstance(data, dict):
        return default
    for path in candidate_keys:
        curr = data
        found = True
        for part in path.split("."):
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            elif isinstance(curr, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(curr):
                    curr = curr[idx]
                else:
                    found = False; break
            else:
                found = False; break
        if found and curr is not None:
            return curr
    return default

# ─────────────────────────────────────────────────────────────
# 3. ASTRONOMICAL MATHEMATICAL HELPERS
# ─────────────────────────────────────────────────────────────

def _safe_round(x, n=2):
    try:
        if x is None or math.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), n)

def moon_illum_at(ts, eph, utc_dt: datetime) -> float:
    """Compute Moon illumination fraction (0.0 to 1.0) at specific UTC datetime."""
    t  = ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
    e  = eph["earth"]
    sp = e.at(t).observe(eph["sun"]).apparent()
    mp = e.at(t).observe(eph["moon"]).apparent()
    sep = sp.separation_from(mp).degrees
    return round((1.0 - math.cos(math.radians(sep))) / 2.0, 3)

def barycentric_correction(ts, eph, utc_dt: datetime, ra_hours: float, dec_degrees: float) -> float:
    """
    Rømer Light-Travel Time Delay Correction (UTC -> BJD_TDB).
    Computes delta_days to convert geocentric arrival times to Solar System Barycenter.
    """
    C_AU_PER_DAY = 173.1446327
    t = ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
    r_earth = eph["earth"].at(t).position.au

    ra_rad  = math.radians(ra_hours * 15.0)
    dec_rad = math.radians(dec_degrees)
    n_hat = np.array([
        math.cos(dec_rad) * math.cos(ra_rad),
        math.cos(dec_rad) * math.sin(ra_rad),
        math.sin(dec_rad),
    ])
    delta_days = float(np.dot(r_earth, n_hat)) / C_AU_PER_DAY
    return delta_days

# ─────────────────────────────────────────────────────────────
# 4. RESOLVED OBJECTS LOCAL DISK CACHE
# ─────────────────────────────────────────────────────────────

def load_resolved_cache() -> dict:
    if os.path.exists(RESOLVED_CACHE_FILE):
        try:
            with open(RESOLVED_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not read resolved objects cache: {e}")
    return {}

def save_resolved_cache(cache: dict) -> None:
    try:
        with open(RESOLVED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, separators=(",", ":"))
    except Exception as e:
        log.warning(f"Could not save resolved objects cache: {e}")

# ─────────────────────────────────────────────────────────────
# 5. ASTEROID TELEMETRY & 3-TIER FALLBACK CASCADE
# ─────────────────────────────────────────────────────────────

def _parse_cad_datetime(cd_str: str) -> tuple[datetime | None, int | None]:
    time_unc_min = None
    cd_clean = cd_str.strip()
    if "±" in cd_clean or "+-" in cd_clean:
        parts = re.split(r'[±]|\+\-', cd_clean, 1)
        cd_clean = parts[0].strip()
        try:
            unc_match = re.search(r'\d+', parts[-1])
            if unc_match:
                time_unc_min = int(unc_match.group())
        except (ValueError, IndexError):
            pass
    for fmt in ["%Y-%b-%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%b-%d", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(cd_clean, fmt).replace(tzinfo=timezone.utc)
            return dt, time_unc_min
        except ValueError:
            continue
    return None, time_unc_min

def parse_horizons_csv_dist(text: str) -> tuple[float, float, float | None] | None:
    start = text.find("$$SOE")
    end   = text.find("$$EOE")
    if start == -1 or end == -1:
        return None
    data_lines = text[start + 5:end].strip().splitlines()
    if not data_lines:
        return None
    fields = [f.strip() for f in data_lines[0].split(",")]
    if len(fields) < 5:
        return None
    try:
        ra_deg  = float(fields[3])
        dec_deg = float(fields[4])
        delta   = float(fields[5]) if len(fields) > 5 else None
        return ra_deg, dec_deg, delta
    except (ValueError, IndexError):
        return None

def _horizons_radec(command: str, utc_dt: datetime) -> tuple[float, float, float | None] | None:
    t0 = utc_dt.strftime("%Y-%m-%d %H:%M")
    t1 = (utc_dt + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
    params = {
        "format":     "json",
        "COMMAND":    f"'{command}'",
        "OBJ_DATA":   "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "CENTER":     "'500@399'",
        "START_TIME": f"'{t0}'",
        "STOP_TIME":  f"'{t1}'",
        "STEP_SIZE":  "'1m'",
        "QUANTITIES": "'1,20'",
        "ANG_FORMAT": "'DEG'",
        "CSV_FORMAT": "'YES'",
    }
    try:
        resp = http.get(HORIZONS, params=params, timeout=12)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if payload.get("error"):
            return None
        result = payload.get("result", "")
        return parse_horizons_csv_dist(result)
    except Exception as exc:
        log.debug(f"Horizons query failed [{command}]: {exc}")
        return None

def fetch_asteroids_jpl_cad(date_min: str, date_max: str, dist_max_ld: float = 15.0) -> list[dict]:
    """Fetch Near-Earth Asteroid close approaches from NASA JPL CAD Open API."""
    params = {
        "date-min": date_min,
        "date-max": date_max,
        "dist-max": f"{dist_max_ld}LD",
        "sort":     "date",
        "fullname": "true",
    }
    log.info(f"Querying JPL CAD API (range: {date_min} -> {date_max}, max: {dist_max_ld} LD)...")
    try:
        resp = http.get(CAD_API, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error(f"JPL CAD API request failed: {exc}")
        return []

    fields = safe_json_extract(data, ["fields"], [])
    rows   = safe_json_extract(data, ["data"], [])
    if not fields or not rows:
        log.warning("JPL CAD returned empty dataset.")
        return []

    idx = {f: i for i, f in enumerate(fields)}
    needed = {"des", "cd", "dist", "v_rel", "h"}
    if needed - set(idx.keys()):
        log.warning(f"JPL CAD missing expected fields: {needed - set(idx.keys())}")
        return []

    events = []
    seen = set()

    for row in rows:
        try:
            des     = str(row[idx["des"]]).strip()
            cd_str  = str(row[idx["cd"]]).strip()
            dist_au = float(str(row[idx["dist"]]).strip())
            v_rel   = float(str(row[idx["v_rel"]]).strip())
            h_mag   = float(str(row[idx["h"]]).strip())

            utc_dt, time_unc_min = _parse_cad_datetime(cd_str)
            if utc_dt is None:
                continue

            dist_km = dist_au * AU_KM
            dist_ld = dist_km / LD_KM
            fullname = str(row[idx["fullname"]]).strip() if "fullname" in idx else des

            ev_id = f"AST_{des.replace(' ','_').replace('/','_')[:20]}_{utc_dt.strftime('%Y%m%d')}"
            if ev_id in seen:
                continue
            seen.add(ev_id)

            precision = "exact" if (time_unc_min is None or time_unc_min <= 60) else "approximate"
            unc_str  = f" ±{time_unc_min} min" if time_unc_min is not None else ""

            if dist_ld < 5.0:
                parallax_note = (
                    "⚠️ CLOSE APPROACH (<5 LD): Geocentric coordinates from Horizons may "
                    "differ from your surface location by several degrees due to parallax. "
                    "Use a topocentric Horizons query for critical observations."
                )
            else:
                parallax_note = (
                    "Geocentric coordinates (500@399): parallax error is ≤arcminutes "
                    "at this distance (≥5 LD) and does not significantly affect visibility."
                )

            events.append({
                "id":                   ev_id,
                "category":             "asteroid",
                "type":                 "Asteroid Close Approach",
                "subtype":              "near_earth_object",
                "name":                 fullname,
                "des":                  des,
                "utc":                  utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":                 None,
                "dec_d":                None,
                "distance_ld":          round(dist_ld, 3),
                "distance_km":          round(dist_km, 0),
                "v_rel_kms":            round(v_rel, 2),
                "h_mag":                round(h_mag, 1),
                "time_uncertainty_min": time_unc_min,
                "global_event":         True,
                "visibility_available": False,
                "visibility_note":      "Coordinates pending cascade resolution.",
                "parallax_caveat":      parallax_note,
                "precision":            precision,
                "method":               "jpl_cad_api",
                "description":          (
                    f"{fullname} passes Earth at {round(dist_ld,2)} "
                    f"lunar distances ({round(dist_km/1e6,3):.3f} million km), "
                    f"nominal time{unc_str} UTC, "
                    f"relative velocity {round(v_rel,1)} km/s. H={round(h_mag,1)}."
                ),
            })
        except Exception as row_exc:
            log.debug(f"Skipping CAD row: {row_exc}")
            continue

    log.info(f"JPL CAD: Retrieved {len(events)} asteroid approaches.")
    return events

def _sbdb_propagate(des: str, utc_dt: datetime, ts, eph) -> tuple[float, float, float] | None:
    try:
        resp = http.get(SBDB_API, params={"sstr": des}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    orbit_data = safe_json_extract(data, ["orbit"], {})
    elements   = safe_json_extract(orbit_data, ["elements"], [])
    epoch_jd   = safe_json_extract(orbit_data, ["epoch"], None)
    if not elements or not epoch_jd:
        return None

    el = {e["name"]: float(e["value"]) for e in elements if "name" in e and "value" in e}
    a, e, i_deg, om, w, ma = el.get("a"), el.get("e"), el.get("i"), el.get("om"), el.get("w"), el.get("ma")
    if any(v is None for v in [a, e, i_deg, om, w, ma]):
        return None

    try:
        p = a * (1.0 - e * e)
        t_epoch = ts.tt_jd(float(epoch_jd))
        orbit = _KeplerOrbit._from_mean_anomaly(
            p, e, i_deg, om, w, ma, t_epoch, GM_SUN, 10, des
        )
        orbit._rotation = inertial_frames["ECLIPJ2000"].T
        body = eph["sun"] + orbit
        t_now = ts.from_datetime(utc_dt)
        astrometric = eph["earth"].at(t_now).observe(body)
        ra, dec, dist = astrometric.radec()
        return round(float(ra.hours), 3), round(float(dec.degrees), 2), round(float(dist.au), 4)
    except Exception:
        return None

def resolve_asteroid_position(ast: dict, ts, eph, cache: dict) -> dict:
    """3-Tier Cascade: Tier 1 (JPL Horizons) -> Tier 2 (JPL SBDB) -> Tier 3 (CAD Only)"""
    des = ast.get("des", ast.get("name", ""))
    utc_dt_str = ast["utc"]
    utc_dt = datetime.strptime(utc_dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    cache_key = f"{des}_{utc_dt_str}"
    if cache_key in cache:
        cached = cache[cache_key]
        ast.update(cached)
        return ast

    # Tier 1: JPL Horizons
    result = _horizons_radec(f"DES={des};", utc_dt)
    if result is None:
        result = _horizons_radec(f"{des};", utc_dt)

    if result is not None:
        ra_deg, dec_deg, delta = result
        ra_h, dec_d = round(ra_deg / 15.0, 3), round(dec_deg, 2)
        delta_au = round(delta, 4) if delta else round(ast.get("distance_ld", 0) * LD_KM / AU_KM, 4)
        
        resolved_fields = {
            "ra_h":                 ra_h,
            "dec_d":                dec_d,
            "distance_au":          delta_au,
            "visibility_available": True,
            "global_event":         False,
            "precision":            "horizons_perturbed",
            "method":               "jpl_horizons",
            "visibility_note":      "ONE geocentric RA/Dec from JPL Horizons N-body perturbed ephemeris.",
            "accuracy_note":        "Time+distance from JPL CAD. Coordinates from JPL Horizons (500@399). " + ast.get("parallax_caveat", "")
        }
        ast.update(resolved_fields)
        cache[cache_key] = resolved_fields
        return ast

    # Tier 2: SBDB Propagation
    sbdb_res = _sbdb_propagate(des, utc_dt, ts, eph)
    if sbdb_res is not None:
        ra_h, dec_d, delta_au = sbdb_res
        resolved_fields = {
            "ra_h":                 ra_h,
            "dec_d":                dec_d,
            "distance_au":          delta_au,
            "visibility_available": True,
            "global_event":         False,
            "precision":            "approximate",
            "method":               "sbdb_two_body",
            "visibility_note":      "ONE geocentric RA/Dec from SBDB two-body propagation (approximate).",
            "accuracy_note":        "Coordinates from SBDB Keplerian elements via Skyfield two-body propagation. " + ast.get("parallax_caveat", "")
        }
        ast.update(resolved_fields)
        cache[cache_key] = resolved_fields
        return ast

    # Tier 3: CAD Only
    resolved_fields = {
        "ra_h":                 None,
        "dec_d":                None,
        "distance_au":          None,
        "visibility_available": False,
        "global_event":         True,
        "precision":            "global_location_dependent",
        "method":               "jpl_cad_only",
        "visibility_note":      "Neither JPL Horizons nor SBDB could resolve RA/Dec.",
        "accuracy_note":        "Time and distance from JPL CAD. Orbit position unconstrained."
    }
    ast.update(resolved_fields)
    cache[cache_key] = resolved_fields
    return ast

# ─────────────────────────────────────────────────────────────
# 6. COMET TELEMETRY & FALLBACK CASCADE
# ─────────────────────────────────────────────────────────────

def _parse_hms(ra_str: str) -> float | None:
    try:
        parts = re.split(r'[:\s]+', ra_str.strip())
        return float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
    except Exception:
        return None

def _parse_dms(dec_str: str) -> float | None:
    try:
        sign = -1 if dec_str.strip().startswith('-') else 1
        parts = re.split(r'[:\s]+', dec_str.strip().lstrip('+-'))
        return sign * (float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0)
    except Exception:
        return None

def fetch_cobs_comets(now_utc: datetime, cache: dict, mag_limit: float = 12.0) -> list[dict]:
    """Fetch active comets from COBS Planner Open API with Horizons upgrade."""
    date_str = now_utc.strftime("%Y-%m-%d")
    log.info(f"Querying COBS Comet Database for {date_str} (limit_mag={mag_limit})...")
    try:
        resp = http.get(COBS_API, params={"date": date_str, "limit_mag": mag_limit}, timeout=15)
        resp.raise_for_status()
        comet_list = safe_json_extract(resp.json(), ["comet_list"], [])
    except Exception as exc:
        log.warning(f"COBS Planner request failed: {exc}")
        return []

    events = []
    for c in comet_list:
        try:
            fullname = safe_json_extract(c, ["comet_fullname", "comet_name"], "Unknown Comet")
            des      = safe_json_extract(c, ["comet_name"], fullname)
            mag      = safe_json_extract(c, ["magnitude"], None)
            best_ra  = safe_json_extract(c, ["best_ra"], "")
            best_dec = safe_json_extract(c, ["best_dec"], "")
            best_time = safe_json_extract(c, ["best_time"], now_utc.strftime("%Y-%m-%d 00:00"))

            try:
                obs_dt = datetime.strptime(best_time, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except Exception:
                obs_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

            ra_h_cobs  = _parse_hms(best_ra)
            dec_d_cobs = _parse_dms(best_dec)

            if mag is not None and float(mag) > mag_limit:
                continue

            ev_id = f"CMT_{des.replace(' ','_').replace('/','_')[:20]}"
            ev = {
                "id":                 ev_id,
                "category":           "comet",
                "type":               "Bright Comet Visible",
                "subtype":            "comet",
                "name":               fullname,
                "des":                des,
                "utc":                obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":               ra_h_cobs,
                "dec_d":              dec_d_cobs,
                "magnitude_observed": float(mag) if mag is not None else None,
                "magnitude_display":  f"~{mag}" if mag is not None else "check reports",
                "magnitude_source":   "cobs_observed",
                "sun_elongation_deg": safe_json_extract(c, ["sun_elongation"], None),
                "moon_elongation_deg": safe_json_extract(c, ["moon_elongation"], None),
                "constellation":      safe_json_extract(c, ["constelation"], None),
                "trend":              safe_json_extract(c, ["trend"], "unknown"),
                "global_event":       False,
                "visibility_available": ra_h_cobs is not None and dec_d_cobs is not None,
                "precision":          "approximate",
                "method":             "cobs_planner",
                "accuracy_note":      "Magnitude from COBS Planner (real observer reports). Position upgraded if Horizons available.",
                "description":        f"Comet {fullname} — observed magnitude ~{mag}. Source: COBS Planner.",
            }

            # Attempt Horizons Upgrade
            cache_comet_key = f"COMET_{des}_{obs_dt.strftime('%Y%m%d')}"
            pos = cache.get(cache_comet_key)
            if not pos:
                m_periodic = re.match(r'^(\d+)P', des.strip())
                cmd = str(90000000 + int(m_periodic.group(1))) if m_periodic else des
                res = _horizons_radec(cmd, obs_dt)
                if res:
                    pos = (round(res[0] / 15.0, 3), round(res[1], 2))
                    cache[cache_comet_key] = pos

            if pos:
                ev["ra_h"] = pos[0]
                ev["dec_d"] = pos[1]
                ev["precision"] = "horizons_perturbed"
                ev["method"] = "cobs+horizons"
                ev["accuracy_note"] = "Magnitude from COBS Planner. Coordinates from JPL Horizons N-body ephemeris."

            events.append(ev)
        except Exception as c_exc:
            log.debug(f"Skipping COBS comet row: {c_exc}")
            continue

    log.info(f"COBS Comets: Retrieved {len(events)} visible comets.")
    return events

def mpc_cache_valid(max_age_hours: float = 24.0) -> bool:
    if not os.path.exists(MPC_CACHE_META) or not os.path.exists(MPC_CACHE_FILE):
        return False
    try:
        with open(MPC_CACHE_META) as f:
            meta = json.load(f)
        cached_time = datetime.fromisoformat(meta["fetched_utc"])
        return (datetime.now(timezone.utc) - cached_time).total_seconds() < max_age_hours * 3600
    except Exception:
        return False

def load_mpc_cached() -> io.BytesIO | None:
    if os.path.exists(MPC_CACHE_FILE):
        try:
            with open(MPC_CACHE_FILE, "rb") as f:
                return io.BytesIO(f.read())
        except Exception:
            pass
    return None

def save_mpc_cache(content: bytes) -> None:
    try:
        with open(MPC_CACHE_FILE, "wb") as f:
            f.write(content)
        with open(MPC_CACHE_META, "w") as f:
            json.dump({"fetched_utc": datetime.now(timezone.utc).isoformat(), "size_bytes": len(content)}, f)
    except Exception as exc:
        log.warning(f"Could not save MPC cache: {exc}")

def fetch_mpc_comets(known_ids: set[str], ts, eph, now_utc: datetime, cache: dict, mag_limit: float = 12.0) -> list[dict]:
    """Fallback Comet Fetcher from Minor Planet Center Catalog."""
    comets_data = None
    if mpc_cache_valid(24.0):
        log.info("Using cached MPC Comet catalog (<24h old)...")
        comets_data = load_mpc_cached()
    else:
        try:
            log.info("Fetching Comet catalog from Minor Planet Center...")
            resp = http.get(MPC_URL, timeout=30)
            resp.raise_for_status()
            save_mpc_cache(resp.content)
            comets_data = io.BytesIO(resp.content)
        except Exception as exc:
            log.warning(f"MPC live fetch failed ({exc}) — attempting disk cache fallback...")
            comets_data = load_mpc_cached()

    if not comets_data:
        log.warning("No MPC comet data available.")
        return []

    events = []
    try:
        comets_df = sk_mpc.load_comets_dataframe(comets_data)
        comet_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        t_now = ts.from_datetime(comet_utc)
        earth, sun = eph["earth"], eph["sun"]

        for _, row in comets_df.iterrows():
            designation = str(row.get("designation", "Unknown"))
            clean_id = designation.replace(" ", "_").replace("/", "_")[:20]
            ev_id = f"CMT_{clean_id}"
            if ev_id in known_ids:
                continue

            comet_body = sun + sk_mpc.comet_orbit(row, ts, GM_SUN)
            astrometric = earth.at(t_now).observe(comet_body)
            ra, dec, dist = astrometric.radec()
            dist_au = float(dist.au)
            if dist_au > 15.0:
                continue

            events.append({
                "id":                 ev_id,
                "category":           "comet",
                "type":               "Bright Comet Visible",
                "subtype":            "comet",
                "name":               designation,
                "des":                designation,
                "utc":                comet_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":               round(float(ra.hours), 3),
                "dec_d":              round(float(dec.degrees), 2),
                "distance_au":        round(dist_au, 3),
                "magnitude_display":  "check reports",
                "magnitude_source":   "mpc_catalog",
                "global_event":       False,
                "visibility_available": True,
                "precision":          "approximate",
                "method":             "mpc_two_body",
                "accuracy_note":      "Position from MPC two-body Keplerian orbit propagation.",
                "description":        f"Comet {designation} at RA {float(ra.hours):.2f}h, Dec {float(dec.degrees):.1f}°.",
            })
    except Exception as exc:
        log.error(f"Failed to parse MPC dataset: {exc}")

    log.info(f"MPC Comets: Fallback provided {len(events)} additional comets.")
    return events

# ─────────────────────────────────────────────────────────────
# 7. SPACE WEATHER TELEMETRY (NOAA SWPC)
# ─────────────────────────────────────────────────────────────

def fetch_noaa_space_weather() -> list[dict]:
    """Fetch live Space Weather Alerts (Geomagnetic Storms, Auroras, Solar Flares) from NOAA SWPC Open API."""
    log.info("Querying NOAA SWPC Open API for Space Weather Alerts & Aurora Warnings...")
    events = []
    try:
        resp = http.get(NOAA_ALERTS, timeout=10)
        resp.raise_for_status()
        alerts = resp.json()
        seen = set()
        for alert in alerts[:20]:
            msg = alert.get("message", "")
            issue_dt_str = alert.get("issue_datetime", "")
            if not msg or not issue_dt_str:
                continue

            is_aurora = "Geomagnetic K-index" in msg or "Aurora" in msg or "WARK" in alert.get("product_id", "")
            is_flare  = "Solar Flare" in msg or "X-ray" in msg or "FLA" in alert.get("product_id", "")

            if not (is_aurora or is_flare):
                continue

            try:
                issue_dt = datetime.strptime(issue_dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                continue

            ev_id = f"SW_{alert.get('product_id','ALT')}_{issue_dt.strftime('%Y%m%d_%H%M')}"
            if ev_id in seen:
                continue
            seen.add(ev_id)

            summary_line = msg.splitlines()[0] if msg.splitlines() else "Space Weather Alert"
            events.append({
                "id":           ev_id,
                "category":     "space_weather",
                "type":         "Geomagnetic Storm / Aurora Alert" if is_aurora else "Solar Flare Alert",
                "subtype":      "space_weather",
                "utc":          issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "product_id":   alert.get("product_id"),
                "global_event": True,
                "precision":    "live_alert",
                "method":       "noaa_swpc_api",
                "description":  summary_line,
                "full_message": msg.strip()[:300]
            })
    except Exception as exc:
        log.warning(f"NOAA Space Weather API fetch failed: {exc}")

    log.info(f"NOAA SWPC: Retrieved {len(events)} active space weather alerts.")
    return events

# ─────────────────────────────────────────────────────────────
# 8. ATMOSPHERIC FIREBALL / BOLIDE TELEMETRY (NASA CNEOS)
# ─────────────────────────────────────────────────────────────

def fetch_nasa_fireballs(limit: int = 10) -> list[dict]:
    """Fetch recent Atmospheric Fireball / Bolide impact reports from NASA JPL CNEOS Open API."""
    log.info("Querying NASA JPL CNEOS Open API for Atmospheric Fireballs / Bolides...")
    events = []
    try:
        resp = http.get(f"{FIREBALL_API}?limit={limit}", timeout=12)
        resp.raise_for_status()
        data = resp.json()
        fields = safe_json_extract(data, ["fields"], [])
        rows   = safe_json_extract(data, ["data"], [])
        if not fields or not rows:
            return []

        idx = {f: i for i, f in enumerate(fields)}
        seen = set()
        for row in rows:
            dt_str = str(row[idx["date"]]).strip()
            try:
                utc_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                continue

            energy = row[idx["energy"]] if "energy" in idx else None
            lat = str(row[idx["lat"]]) + str(row[idx["lat-dir"]]) if ("lat" in idx and row[idx["lat"]]) else "Unknown"
            lon = str(row[idx["lon"]]) + str(row[idx["lon-dir"]]) if ("lon" in idx and row[idx["lon"]]) else "Unknown"

            ev_id = f"FIR_{utc_dt.strftime('%Y%m%d_%H%M')}"
            if ev_id in seen:
                continue
            seen.add(ev_id)

            events.append({
                "id":                     ev_id,
                "category":               "fireball",
                "type":                   "Bright Fireball / Bolide Impact",
                "subtype":                "fireball",
                "utc":                    utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_impact_energy_kt": float(energy) if energy else None,
                "latitude":               lat,
                "longitude":              lon,
                "global_event":           False,
                "precision":              "observed",
                "method":                 "nasa_cneos_fireball_api",
                "description":            f"Atmospheric meteor fireball detected at {utc_dt.strftime('%Y-%m-%d %H:%M')} UTC (Location: {lat}, {lon}, Energy: {energy} kt).",
            })
    except Exception as exc:
        log.warning(f"NASA CNEOS Fireball API fetch failed: {exc}")

    log.info(f"NASA CNEOS: Retrieved {len(events)} recent fireball reports.")
    return events
