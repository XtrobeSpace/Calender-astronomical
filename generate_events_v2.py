#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   XTROBE CALENDAR — FULLY DYNAMIC EVENT GENERATOR v2.1      ║
║                                                              ║
║  ZERO hardcoded event dates. Everything is computed.        ║
║                                                              ║
║  Install:                                                    ║
║    pip install skyfield requests pytz numpy python-dotenv   ║
║                                                              ║
║  .env file (put in project root, never commit):             ║
║    NASA_API_KEY_1=your_first_key                            ║
║    NASA_API_KEY_2=your_fallback_key                         ║
║                                                              ║
║  Run once per year (static events + visibility JSONs):      ║
║    python generate_events_v2.py --type static --year 2026   ║
║                                                              ║
║  Run nightly (asteroids + comets update):                   ║
║    python generate_events_v2.py --type daily                ║
║                                                              ║
║  GitHub Actions caches skyfield_cache/ so de421.bsp is     ║
║  downloaded only once, never re-fetched nightly.           ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────
import json
import math
import os
import sys
import argparse
import time
import io
import logging
from datetime import datetime, timedelta, timezone

# ── Third-party ───────────────────────────────────────────────
import requests
import pytz
import numpy as np

# ── Skyfield ──────────────────────────────────────────────────
from skyfield.api import load, wgs84, Star, Loader
from skyfield import almanac
from skyfield.framelib import ecliptic_frame
from skyfield.data import mpc as sk_mpc                               # comet orbital elements
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN  # for comet_orbit()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Env / secrets ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; CI injects secrets via environment variables directly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xtrobe")

# ─────────────────────────────────────────────────────────────
# NASA API KEY MANAGEMENT
# Two keys loaded from .env / environment; tried in order with
# automatic fallback so a rate-limit on key-1 won't kill the run.
# ─────────────────────────────────────────────────────────────

_NASA_KEYS: list[str] = []

def _init_nasa_keys() -> None:
    """Populate _NASA_KEYS from environment at startup."""
    for env_var in ("NASA_API_KEY_1", "NASA_API_KEY_2"):
        val = os.getenv(env_var, "").strip()
        if val and val not in _NASA_KEYS:
            _NASA_KEYS.append(val)
    if not _NASA_KEYS:
        log.warning("No NASA API keys found in environment — falling back to DEMO_KEY (30 req/hr).")
        _NASA_KEYS.append("DEMO_KEY")


def nasa_get(url: str, params: dict, timeout: int = 20) -> requests.Response | None:
    """
    GET url with NASA API key, automatically rotating to the next key
    on HTTP 429 (rate-limit) or 403 (forbidden / quota exceeded).
    Returns None if all keys are exhausted.
    """
    for idx, key in enumerate(_NASA_KEYS):
        p = {**params, "api_key": key}
        try:
            resp = requests.get(url, params=p, timeout=timeout)
            if resp.status_code in (429, 403):
                log.warning(f"NASA key #{idx+1} rate-limited ({resp.status_code}), trying next key...")
                time.sleep(2)
                continue
            return resp
        except requests.RequestException as exc:
            log.warning(f"NASA request error with key #{idx+1}: {exc}")
            continue
    log.error("All NASA API keys exhausted.")
    return None


# ─────────────────────────────────────────────────────────────
# PATHS & OUTPUT DIR
# ─────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
SKYFIELD_CACHE  = os.path.join(BASE_DIR, "skyfield_cache")  # cached by GitHub Actions

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SKYFIELD_CACHE, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SKYFIELD INIT — ephemeris is downloaded once, cached forever
# ─────────────────────────────────────────────────────────────

def init_skyfield():
    """
    Load timescale + ephemeris from local cache.
    The Loader checks SKYFIELD_CACHE first; it only downloads de421.bsp
    (~17 MB) when the file is absent.  GitHub Actions caches this folder
    between runs so nightly jobs never re-download.
    """
    loader = Loader(SKYFIELD_CACHE)
    ts  = loader.timescale()
    eph = loader("de421.bsp")
    log.info(f"Ephemeris loaded from cache: {SKYFIELD_CACHE}")
    return ts, eph


# ─────────────────────────────────────────────────────────────
# MIN ALTITUDE FOR VISIBILITY (degrees above horizon)
# ─────────────────────────────────────────────────────────────

MIN_ALT = {
    "lunar":    5,
    "eclipse":  5,
    "season":   0,
    "planet":  10,
    "meteor":  10,
    "deep_sky":20,
    "comet":   12,
    "asteroid":10,
    "variable":15,
    "atmosphere": 0,
}

SUN_NAUTICAL = -12
SUN_ASTRO    = -18

# ─────────────────────────────────────────────────────────────
# LOCATIONS
# ─────────────────────────────────────────────────────────────

LOCATIONS = {
    # ── INDIA ───────────────────────────────────────────────
    "IN-AP":{"name":"Andhra Pradesh",    "lat": 16.51,"lon": 80.52,"tz":"Asia/Kolkata","country":"IN"},
    "IN-AR":{"name":"Arunachal Pradesh", "lat": 27.08,"lon": 93.61,"tz":"Asia/Kolkata","country":"IN"},
    "IN-AS":{"name":"Assam",             "lat": 26.18,"lon": 91.74,"tz":"Asia/Kolkata","country":"IN"},
    "IN-BR":{"name":"Bihar",             "lat": 25.59,"lon": 85.14,"tz":"Asia/Kolkata","country":"IN"},
    "IN-CG":{"name":"Chhattisgarh",      "lat": 21.25,"lon": 81.63,"tz":"Asia/Kolkata","country":"IN"},
    "IN-GA":{"name":"Goa",               "lat": 15.49,"lon": 73.82,"tz":"Asia/Kolkata","country":"IN"},
    "IN-GJ":{"name":"Gujarat",           "lat": 23.02,"lon": 72.57,"tz":"Asia/Kolkata","country":"IN"},
    "IN-HR":{"name":"Haryana",           "lat": 30.73,"lon": 76.78,"tz":"Asia/Kolkata","country":"IN"},
    "IN-HP":{"name":"Himachal Pradesh",  "lat": 31.10,"lon": 77.17,"tz":"Asia/Kolkata","country":"IN"},
    "IN-JH":{"name":"Jharkhand",         "lat": 23.35,"lon": 85.33,"tz":"Asia/Kolkata","country":"IN"},
    "IN-KA":{"name":"Karnataka",         "lat": 12.97,"lon": 77.59,"tz":"Asia/Kolkata","country":"IN"},
    "IN-KL":{"name":"Kerala",            "lat":  8.52,"lon": 76.94,"tz":"Asia/Kolkata","country":"IN"},
    "IN-MP":{"name":"Madhya Pradesh",    "lat": 23.25,"lon": 77.41,"tz":"Asia/Kolkata","country":"IN"},
    "IN-MH":{"name":"Maharashtra",       "lat": 19.07,"lon": 72.87,"tz":"Asia/Kolkata","country":"IN"},
    "IN-MN":{"name":"Manipur",           "lat": 24.81,"lon": 93.94,"tz":"Asia/Kolkata","country":"IN"},
    "IN-ML":{"name":"Meghalaya",         "lat": 25.57,"lon": 91.88,"tz":"Asia/Kolkata","country":"IN"},
    "IN-MZ":{"name":"Mizoram",           "lat": 23.73,"lon": 92.72,"tz":"Asia/Kolkata","country":"IN"},
    "IN-NL":{"name":"Nagaland",          "lat": 25.67,"lon": 94.11,"tz":"Asia/Kolkata","country":"IN"},
    "IN-OD":{"name":"Odisha",            "lat": 20.30,"lon": 85.85,"tz":"Asia/Kolkata","country":"IN"},
    "IN-PB":{"name":"Punjab",            "lat": 30.73,"lon": 76.78,"tz":"Asia/Kolkata","country":"IN"},
    "IN-RJ":{"name":"Rajasthan",         "lat": 26.91,"lon": 75.79,"tz":"Asia/Kolkata","country":"IN"},
    "IN-SK":{"name":"Sikkim",            "lat": 27.33,"lon": 88.61,"tz":"Asia/Kolkata","country":"IN"},
    "IN-TN":{"name":"Tamil Nadu",        "lat": 13.08,"lon": 80.27,"tz":"Asia/Kolkata","country":"IN"},
    "IN-TS":{"name":"Telangana",         "lat": 17.38,"lon": 78.49,"tz":"Asia/Kolkata","country":"IN"},
    "IN-TR":{"name":"Tripura",           "lat": 23.83,"lon": 91.28,"tz":"Asia/Kolkata","country":"IN"},
    "IN-UP":{"name":"Uttar Pradesh",     "lat": 26.85,"lon": 80.95,"tz":"Asia/Kolkata","country":"IN"},
    "IN-UK":{"name":"Uttarakhand",       "lat": 30.32,"lon": 78.03,"tz":"Asia/Kolkata","country":"IN"},
    "IN-WB":{"name":"West Bengal",       "lat": 22.57,"lon": 88.36,"tz":"Asia/Kolkata","country":"IN"},
    "IN-DL":{"name":"Delhi",             "lat": 28.61,"lon": 77.21,"tz":"Asia/Kolkata","country":"IN"},
    "IN-JK":{"name":"Jammu & Kashmir",   "lat": 34.08,"lon": 74.79,"tz":"Asia/Kolkata","country":"IN"},
    "IN-LA":{"name":"Ladakh",            "lat": 34.17,"lon": 77.58,"tz":"Asia/Kolkata","country":"IN"},

    # ── USA ─────────────────────────────────────────────────
    "US-AL":{"name":"Alabama",           "lat": 32.36,"lon": -86.30,"tz":"America/Chicago","country":"US"},
    "US-AK":{"name":"Alaska",            "lat": 58.30,"lon":-134.42,"tz":"America/Anchorage","country":"US"},
    "US-AZ":{"name":"Arizona",           "lat": 33.45,"lon":-112.07,"tz":"America/Phoenix","country":"US"},
    "US-AR":{"name":"Arkansas",          "lat": 34.74,"lon": -92.33,"tz":"America/Chicago","country":"US"},
    "US-CA":{"name":"California",        "lat": 38.55,"lon":-121.47,"tz":"America/Los_Angeles","country":"US"},
    "US-CO":{"name":"Colorado",          "lat": 39.74,"lon":-104.98,"tz":"America/Denver","country":"US"},
    "US-CT":{"name":"Connecticut",       "lat": 41.76,"lon": -72.68,"tz":"America/New_York","country":"US"},
    "US-DE":{"name":"Delaware",          "lat": 39.16,"lon": -75.52,"tz":"America/New_York","country":"US"},
    "US-FL":{"name":"Florida",           "lat": 27.99,"lon": -81.76,"tz":"America/New_York","country":"US"},
    "US-GA":{"name":"Georgia",           "lat": 33.24,"lon": -83.44,"tz":"America/New_York","country":"US"},
    "US-HI":{"name":"Hawaii",            "lat": 20.80,"lon":-156.33,"tz":"Pacific/Honolulu","country":"US"},
    "US-ID":{"name":"Idaho",             "lat": 44.07,"lon":-114.74,"tz":"America/Boise","country":"US"},
    "US-IL":{"name":"Illinois",          "lat": 40.63,"lon": -89.40,"tz":"America/Chicago","country":"US"},
    "US-IN":{"name":"Indiana",           "lat": 40.27,"lon": -86.13,"tz":"America/Indiana/Indianapolis","country":"US"},
    "US-IA":{"name":"Iowa",              "lat": 42.08,"lon": -93.50,"tz":"America/Chicago","country":"US"},
    "US-KS":{"name":"Kansas",            "lat": 38.53,"lon": -96.73,"tz":"America/Chicago","country":"US"},
    "US-KY":{"name":"Kentucky",          "lat": 37.67,"lon": -84.67,"tz":"America/New_York","country":"US"},
    "US-LA":{"name":"Louisiana",         "lat": 31.17,"lon": -91.87,"tz":"America/Chicago","country":"US"},
    "US-ME":{"name":"Maine",             "lat": 45.37,"lon": -69.24,"tz":"America/New_York","country":"US"},
    "US-MD":{"name":"Maryland",          "lat": 39.05,"lon": -76.64,"tz":"America/New_York","country":"US"},
    "US-MA":{"name":"Massachusetts",     "lat": 42.23,"lon": -71.53,"tz":"America/New_York","country":"US"},
    "US-MI":{"name":"Michigan",          "lat": 44.35,"lon": -85.41,"tz":"America/Detroit","country":"US"},
    "US-MN":{"name":"Minnesota",         "lat": 46.28,"lon": -94.31,"tz":"America/Chicago","country":"US"},
    "US-MS":{"name":"Mississippi",       "lat": 32.74,"lon": -89.67,"tz":"America/Chicago","country":"US"},
    "US-MO":{"name":"Missouri",          "lat": 38.46,"lon": -92.29,"tz":"America/Chicago","country":"US"},
    "US-MT":{"name":"Montana",           "lat": 46.88,"lon":-110.36,"tz":"America/Denver","country":"US"},
    "US-NE":{"name":"Nebraska",          "lat": 41.13,"lon": -98.27,"tz":"America/Chicago","country":"US"},
    "US-NV":{"name":"Nevada",            "lat": 38.31,"lon":-117.06,"tz":"America/Los_Angeles","country":"US"},
    "US-NH":{"name":"New Hampshire",     "lat": 43.45,"lon": -71.57,"tz":"America/New_York","country":"US"},
    "US-NJ":{"name":"New Jersey",        "lat": 40.22,"lon": -74.76,"tz":"America/New_York","country":"US"},
    "US-NM":{"name":"New Mexico",        "lat": 34.31,"lon":-106.02,"tz":"America/Denver","country":"US"},
    "US-NY":{"name":"New York",          "lat": 42.16,"lon": -74.95,"tz":"America/New_York","country":"US"},
    "US-NC":{"name":"North Carolina",    "lat": 35.63,"lon": -79.81,"tz":"America/New_York","country":"US"},
    "US-ND":{"name":"North Dakota",      "lat": 47.53,"lon":-100.44,"tz":"America/Chicago","country":"US"},
    "US-OH":{"name":"Ohio",              "lat": 40.39,"lon": -82.76,"tz":"America/New_York","country":"US"},
    "US-OK":{"name":"Oklahoma",          "lat": 35.59,"lon": -97.49,"tz":"America/Chicago","country":"US"},
    "US-OR":{"name":"Oregon",            "lat": 43.93,"lon":-120.56,"tz":"America/Los_Angeles","country":"US"},
    "US-PA":{"name":"Pennsylvania",      "lat": 40.59,"lon": -77.21,"tz":"America/New_York","country":"US"},
    "US-RI":{"name":"Rhode Island",      "lat": 41.68,"lon": -71.56,"tz":"America/New_York","country":"US"},
    "US-SC":{"name":"South Carolina",    "lat": 33.90,"lon": -80.90,"tz":"America/New_York","country":"US"},
    "US-SD":{"name":"South Dakota",      "lat": 44.30,"lon":-100.23,"tz":"America/Chicago","country":"US"},
    "US-TN":{"name":"Tennessee",         "lat": 35.86,"lon": -86.35,"tz":"America/Chicago","country":"US"},
    "US-TX":{"name":"Texas",             "lat": 31.48,"lon": -99.33,"tz":"America/Chicago","country":"US"},
    "US-UT":{"name":"Utah",              "lat": 39.32,"lon":-111.09,"tz":"America/Denver","country":"US"},
    "US-VT":{"name":"Vermont",           "lat": 44.06,"lon": -72.67,"tz":"America/New_York","country":"US"},
    "US-VA":{"name":"Virginia",          "lat": 37.77,"lon": -78.17,"tz":"America/New_York","country":"US"},
    "US-WA":{"name":"Washington",        "lat": 47.38,"lon":-120.46,"tz":"America/Los_Angeles","country":"US"},
    "US-WV":{"name":"West Virginia",     "lat": 38.49,"lon": -80.95,"tz":"America/New_York","country":"US"},
    "US-WI":{"name":"Wisconsin",         "lat": 44.27,"lon": -89.62,"tz":"America/Chicago","country":"US"},
    "US-WY":{"name":"Wyoming",           "lat": 42.95,"lon":-107.55,"tz":"America/Denver","country":"US"},

    # ── NIGERIA ─────────────────────────────────────────────
    "NG-AB":{"name":"Abia",              "lat":  5.53,"lon":  7.49,"tz":"Africa/Lagos","country":"NG"},
    "NG-AD":{"name":"Adamawa",           "lat":  9.20,"lon": 12.49,"tz":"Africa/Lagos","country":"NG"},
    "NG-AK":{"name":"Akwa Ibom",         "lat":  5.03,"lon":  7.93,"tz":"Africa/Lagos","country":"NG"},
    "NG-AN":{"name":"Anambra",           "lat":  6.21,"lon":  7.07,"tz":"Africa/Lagos","country":"NG"},
    "NG-BA":{"name":"Bauchi",            "lat": 10.31,"lon":  9.84,"tz":"Africa/Lagos","country":"NG"},
    "NG-BY":{"name":"Bayelsa",           "lat":  4.92,"lon":  6.26,"tz":"Africa/Lagos","country":"NG"},
    "NG-BE":{"name":"Benue",             "lat":  7.73,"lon":  8.52,"tz":"Africa/Lagos","country":"NG"},
    "NG-BO":{"name":"Borno",             "lat": 11.85,"lon": 13.16,"tz":"Africa/Lagos","country":"NG"},
    "NG-CR":{"name":"Cross River",       "lat":  4.97,"lon":  8.34,"tz":"Africa/Lagos","country":"NG"},
    "NG-DE":{"name":"Delta",             "lat":  6.20,"lon":  6.74,"tz":"Africa/Lagos","country":"NG"},
    "NG-EB":{"name":"Ebonyi",            "lat":  6.33,"lon":  8.11,"tz":"Africa/Lagos","country":"NG"},
    "NG-ED":{"name":"Edo",               "lat":  6.34,"lon":  5.63,"tz":"Africa/Lagos","country":"NG"},
    "NG-EK":{"name":"Ekiti",             "lat":  7.62,"lon":  5.22,"tz":"Africa/Lagos","country":"NG"},
    "NG-EN":{"name":"Enugu",             "lat":  6.44,"lon":  7.50,"tz":"Africa/Lagos","country":"NG"},
    "NG-FC":{"name":"FCT Abuja",         "lat":  9.06,"lon":  7.50,"tz":"Africa/Lagos","country":"NG"},
    "NG-GO":{"name":"Gombe",             "lat": 10.29,"lon": 11.17,"tz":"Africa/Lagos","country":"NG"},
    "NG-IM":{"name":"Imo",               "lat":  5.49,"lon":  7.03,"tz":"Africa/Lagos","country":"NG"},
    "NG-JI":{"name":"Jigawa",            "lat": 11.76,"lon":  9.34,"tz":"Africa/Lagos","country":"NG"},
    "NG-KD":{"name":"Kaduna",            "lat": 10.52,"lon":  7.44,"tz":"Africa/Lagos","country":"NG"},
    "NG-KN":{"name":"Kano",              "lat": 12.00,"lon":  8.52,"tz":"Africa/Lagos","country":"NG"},
    "NG-KT":{"name":"Katsina",           "lat": 12.99,"lon":  7.60,"tz":"Africa/Lagos","country":"NG"},
    "NG-KB":{"name":"Kebbi",             "lat": 12.45,"lon":  4.20,"tz":"Africa/Lagos","country":"NG"},
    "NG-KO":{"name":"Kogi",              "lat":  7.80,"lon":  6.74,"tz":"Africa/Lagos","country":"NG"},
    "NG-KW":{"name":"Kwara",             "lat":  8.50,"lon":  4.55,"tz":"Africa/Lagos","country":"NG"},
    "NG-LA":{"name":"Lagos",             "lat":  6.52,"lon":  3.38,"tz":"Africa/Lagos","country":"NG"},
    "NG-NA":{"name":"Nasarawa",          "lat":  8.49,"lon":  8.52,"tz":"Africa/Lagos","country":"NG"},
    "NG-NI":{"name":"Niger",             "lat":  9.61,"lon":  6.56,"tz":"Africa/Lagos","country":"NG"},
    "NG-OG":{"name":"Ogun",              "lat":  7.16,"lon":  3.35,"tz":"Africa/Lagos","country":"NG"},
    "NG-ON":{"name":"Ondo",              "lat":  7.25,"lon":  5.19,"tz":"Africa/Lagos","country":"NG"},
    "NG-OS":{"name":"Osun",              "lat":  7.77,"lon":  4.56,"tz":"Africa/Lagos","country":"NG"},
    "NG-OY":{"name":"Oyo",               "lat":  7.38,"lon":  3.93,"tz":"Africa/Lagos","country":"NG"},
    "NG-PL":{"name":"Plateau",           "lat":  9.93,"lon":  8.89,"tz":"Africa/Lagos","country":"NG"},
    "NG-RI":{"name":"Rivers",            "lat":  4.82,"lon":  7.04,"tz":"Africa/Lagos","country":"NG"},
    "NG-SO":{"name":"Sokoto",            "lat": 13.06,"lon":  5.24,"tz":"Africa/Lagos","country":"NG"},
    "NG-TA":{"name":"Taraba",            "lat":  8.89,"lon": 11.37,"tz":"Africa/Lagos","country":"NG"},
    "NG-YO":{"name":"Yobe",              "lat": 11.75,"lon": 11.97,"tz":"Africa/Lagos","country":"NG"},
    "NG-ZA":{"name":"Zamfara",           "lat": 12.17,"lon":  6.66,"tz":"Africa/Lagos","country":"NG"},

    # ── PHILIPPINES ─────────────────────────────────────────
    "PH-NCR": {"name":"Metro Manila",           "lat": 14.60,"lon":121.00,"tz":"Asia/Manila","country":"PH"},
    "PH-CAR": {"name":"Cordillera (Baguio)",     "lat": 16.41,"lon":120.60,"tz":"Asia/Manila","country":"PH"},
    "PH-I":   {"name":"Ilocos (Laoag)",          "lat": 18.20,"lon":120.59,"tz":"Asia/Manila","country":"PH"},
    "PH-II":  {"name":"Cagayan Valley",          "lat": 17.61,"lon":121.72,"tz":"Asia/Manila","country":"PH"},
    "PH-III": {"name":"Central Luzon",           "lat": 15.48,"lon":120.71,"tz":"Asia/Manila","country":"PH"},
    "PH-IVA": {"name":"CALABARZON",              "lat": 14.10,"lon":121.08,"tz":"Asia/Manila","country":"PH"},
    "PH-IVB": {"name":"MIMAROPA",                "lat": 13.41,"lon":121.18,"tz":"Asia/Manila","country":"PH"},
    "PH-V":   {"name":"Bicol",                   "lat": 13.14,"lon":123.74,"tz":"Asia/Manila","country":"PH"},
    "PH-VI":  {"name":"Western Visayas (Iloilo)","lat": 10.72,"lon":122.56,"tz":"Asia/Manila","country":"PH"},
    "PH-VII": {"name":"Central Visayas (Cebu)",  "lat": 10.32,"lon":123.89,"tz":"Asia/Manila","country":"PH"},
    "PH-VIII":{"name":"Eastern Visayas",         "lat": 11.24,"lon":125.00,"tz":"Asia/Manila","country":"PH"},
    "PH-IX":  {"name":"Zamboanga Peninsula",     "lat":  7.83,"lon":123.43,"tz":"Asia/Manila","country":"PH"},
    "PH-X":   {"name":"Northern Mindanao",       "lat":  8.48,"lon":124.65,"tz":"Asia/Manila","country":"PH"},
    "PH-XI":  {"name":"Davao Region",            "lat":  7.07,"lon":125.61,"tz":"Asia/Manila","country":"PH"},
    "PH-XII": {"name":"SOCCSKSARGEN",            "lat":  6.50,"lon":124.85,"tz":"Asia/Manila","country":"PH"},
    "PH-XIII":{"name":"Caraga (Butuan)",         "lat":  8.95,"lon":125.54,"tz":"Asia/Manila","country":"PH"},
    "PH-BARMM":{"name":"Bangsamoro",             "lat":  7.22,"lon":124.25,"tz":"Asia/Manila","country":"PH"},

    # ── UNITED KINGDOM ──────────────────────────────────────
    "GB-ENG":{"name":"England (London)",        "lat": 51.51,"lon":  -0.13,"tz":"Europe/London","country":"GB"},
    "GB-SCT":{"name":"Scotland (Edinburgh)",    "lat": 55.95,"lon":  -3.19,"tz":"Europe/London","country":"GB"},
    "GB-WLS":{"name":"Wales (Cardiff)",         "lat": 51.48,"lon":  -3.18,"tz":"Europe/London","country":"GB"},
    "GB-NIR":{"name":"N. Ireland (Belfast)",    "lat": 54.60,"lon":  -5.93,"tz":"Europe/London","country":"GB"},
    "GB-NE": {"name":"NE England (Newcastle)",  "lat": 54.97,"lon":  -1.62,"tz":"Europe/London","country":"GB"},
    "GB-NW": {"name":"NW England (Manchester)", "lat": 53.48,"lon":  -2.24,"tz":"Europe/London","country":"GB"},
    "GB-YK": {"name":"Yorkshire (Leeds)",       "lat": 53.80,"lon":  -1.55,"tz":"Europe/London","country":"GB"},
    "GB-SW": {"name":"SW England (Exeter)",     "lat": 50.72,"lon":  -3.53,"tz":"Europe/London","country":"GB"},

    # ── GERMANY ─────────────────────────────────────────────
    "DE-BW":{"name":"Baden-Württemberg",        "lat": 48.78,"lon":  9.18,"tz":"Europe/Berlin","country":"DE"},
    "DE-BY":{"name":"Bavaria (Munich)",         "lat": 48.14,"lon": 11.58,"tz":"Europe/Berlin","country":"DE"},
    "DE-BE":{"name":"Berlin",                   "lat": 52.52,"lon": 13.40,"tz":"Europe/Berlin","country":"DE"},
    "DE-BB":{"name":"Brandenburg",              "lat": 52.40,"lon": 13.06,"tz":"Europe/Berlin","country":"DE"},
    "DE-HB":{"name":"Bremen",                   "lat": 53.08,"lon":  8.81,"tz":"Europe/Berlin","country":"DE"},
    "DE-HH":{"name":"Hamburg",                  "lat": 53.57,"lon":  9.99,"tz":"Europe/Berlin","country":"DE"},
    "DE-HE":{"name":"Hesse (Wiesbaden)",        "lat": 50.08,"lon":  8.24,"tz":"Europe/Berlin","country":"DE"},
    "DE-MV":{"name":"Mecklenburg-Vorpommern",   "lat": 53.63,"lon": 11.42,"tz":"Europe/Berlin","country":"DE"},
    "DE-NI":{"name":"Lower Saxony (Hanover)",   "lat": 52.37,"lon":  9.74,"tz":"Europe/Berlin","country":"DE"},
    "DE-NW":{"name":"NRW (Düsseldorf)",         "lat": 51.22,"lon":  6.78,"tz":"Europe/Berlin","country":"DE"},
    "DE-RP":{"name":"Rhineland-Palatinate",     "lat": 49.99,"lon":  8.27,"tz":"Europe/Berlin","country":"DE"},
    "DE-SL":{"name":"Saarland",                 "lat": 49.24,"lon":  6.99,"tz":"Europe/Berlin","country":"DE"},
    "DE-SN":{"name":"Saxony (Dresden)",         "lat": 51.05,"lon": 13.74,"tz":"Europe/Berlin","country":"DE"},
    "DE-ST":{"name":"Saxony-Anhalt",            "lat": 52.13,"lon": 11.62,"tz":"Europe/Berlin","country":"DE"},
    "DE-SH":{"name":"Schleswig-Holstein (Kiel)","lat": 54.32,"lon": 10.14,"tz":"Europe/Berlin","country":"DE"},
    "DE-TH":{"name":"Thuringia (Erfurt)",       "lat": 50.98,"lon": 11.03,"tz":"Europe/Berlin","country":"DE"},

    # ── CANADA ──────────────────────────────────────────────
    "CA-AB":{"name":"Alberta (Edmonton)",       "lat": 53.55,"lon":-113.49,"tz":"America/Edmonton","country":"CA"},
    "CA-BC":{"name":"British Columbia",         "lat": 53.73,"lon":-127.65,"tz":"America/Vancouver","country":"CA"},
    "CA-MB":{"name":"Manitoba (Winnipeg)",      "lat": 49.90,"lon": -97.14,"tz":"America/Winnipeg","country":"CA"},
    "CA-NB":{"name":"New Brunswick",            "lat": 46.56,"lon": -66.46,"tz":"America/Moncton","country":"CA"},
    "CA-NL":{"name":"Newfoundland",             "lat": 53.13,"lon": -57.66,"tz":"America/St_Johns","country":"CA"},
    "CA-NS":{"name":"Nova Scotia",              "lat": 45.19,"lon": -62.75,"tz":"America/Halifax","country":"CA"},
    "CA-NT":{"name":"Northwest Territories",    "lat": 64.83,"lon":-124.85,"tz":"America/Yellowknife","country":"CA"},
    "CA-NU":{"name":"Nunavut",                  "lat": 70.30,"lon": -83.13,"tz":"America/Iqaluit","country":"CA"},
    "CA-ON":{"name":"Ontario (Toronto)",        "lat": 51.25,"lon": -85.32,"tz":"America/Toronto","country":"CA"},
    "CA-PE":{"name":"Prince Edward Island",     "lat": 46.51,"lon": -63.42,"tz":"America/Halifax","country":"CA"},
    "CA-QC":{"name":"Quebec",                   "lat": 53.00,"lon": -70.00,"tz":"America/Toronto","country":"CA"},
    "CA-SK":{"name":"Saskatchewan",             "lat": 54.00,"lon":-106.11,"tz":"America/Regina","country":"CA"},
    "CA-YT":{"name":"Yukon",                    "lat": 63.00,"lon":-136.00,"tz":"America/Whitehorse","country":"CA"},

    # ── AUSTRALIA ───────────────────────────────────────────
    "AU-NSW":{"name":"New South Wales (Sydney)","lat":-33.87,"lon":151.21,"tz":"Australia/Sydney","country":"AU"},
    "AU-VIC":{"name":"Victoria (Melbourne)",    "lat":-37.81,"lon":144.96,"tz":"Australia/Melbourne","country":"AU"},
    "AU-QLD":{"name":"Queensland (Brisbane)",   "lat":-27.47,"lon":153.02,"tz":"Australia/Brisbane","country":"AU"},
    "AU-SA": {"name":"South Australia",         "lat":-34.93,"lon":138.60,"tz":"Australia/Adelaide","country":"AU"},
    "AU-WA": {"name":"Western Australia",       "lat":-31.95,"lon":115.86,"tz":"Australia/Perth","country":"AU"},
    "AU-TAS":{"name":"Tasmania (Hobart)",       "lat":-42.88,"lon":147.33,"tz":"Australia/Hobart","country":"AU"},
    "AU-NT": {"name":"Northern Territory",      "lat":-12.46,"lon":130.84,"tz":"Australia/Darwin","country":"AU"},
    "AU-ACT":{"name":"ACT (Canberra)",          "lat":-35.28,"lon":149.13,"tz":"Australia/Sydney","country":"AU"},

    # ── SOUTH AFRICA ────────────────────────────────────────
    "ZA-EC":{"name":"Eastern Cape",             "lat":-32.85,"lon": 27.43,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-FS":{"name":"Free State",               "lat":-29.12,"lon": 26.21,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-GP":{"name":"Gauteng (Johannesburg)",   "lat":-26.20,"lon": 28.04,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-KN":{"name":"KwaZulu-Natal",            "lat":-29.60,"lon": 30.38,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-LP":{"name":"Limpopo",                  "lat":-23.90,"lon": 29.45,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-MP":{"name":"Mpumalanga",               "lat":-25.47,"lon": 30.97,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-NC":{"name":"Northern Cape (Kimberley)","lat":-28.74,"lon": 24.77,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-NW":{"name":"North West",               "lat":-25.87,"lon": 25.64,"tz":"Africa/Johannesburg","country":"ZA"},
    "ZA-WC":{"name":"Western Cape (Cape Town)", "lat":-33.93,"lon": 18.42,"tz":"Africa/Johannesburg","country":"ZA"},
}

# Build country → [location_codes] index once at startup
COUNTRY_CODES: dict[str, list[str]] = {}
for _code, _loc in LOCATIONS.items():
    COUNTRY_CODES.setdefault(_loc["country"], []).append(_code)


# ─────────────────────────────────────────────────────────────
# CATALOGS
# ─────────────────────────────────────────────────────────────

MESSIER_CATALOG = [
    # id, name, type, ra_h, dec_d, mag, size_arcmin
    ("M1",  "Crab Nebula",          "SNR",  5.575,  22.01, 8.4,  7.0),
    ("M2",  "",                     "GC",  21.558,  -0.82, 6.5, 16.0),
    ("M3",  "",                     "GC",  13.703,  28.38, 6.2, 18.0),
    ("M4",  "",                     "GC",  16.393, -26.53, 5.6, 36.0),
    ("M5",  "",                     "GC",  15.310,   2.08, 5.6, 23.0),
    ("M6",  "Butterfly Cluster",    "OC",  17.667, -32.22, 4.2, 25.0),
    ("M7",  "Ptolemy Cluster",      "OC",  17.897, -34.82, 3.3, 80.0),
    ("M8",  "Lagoon Nebula",        "EN",  18.063, -24.38, 5.8, 90.0),
    ("M9",  "",                     "GC",  17.320, -18.51, 7.7,  9.0),
    ("M10", "",                     "GC",  16.950,  -4.10, 6.6, 15.0),
    ("M11", "Wild Duck Cluster",    "OC",  18.850,  -6.27, 5.8, 14.0),
    ("M12", "",                     "GC",  16.787,  -1.95, 6.7, 16.0),
    ("M13", "Hercules Cluster",     "GC",  16.695,  36.46, 5.8, 20.0),
    ("M14", "",                     "GC",  17.627,  -3.25, 7.6, 11.0),
    ("M15", "",                     "GC",  21.500,  12.17, 6.2, 18.0),
    ("M16", "Eagle Nebula",         "EN",  18.313, -13.79, 6.4, 35.0),
    ("M17", "Omega Nebula",         "EN",  18.347, -16.18, 6.0, 46.0),
    ("M18", "",                     "OC",  18.333, -17.13, 7.5,  9.0),
    ("M19", "",                     "GC",  17.043, -26.27, 6.8, 17.0),
    ("M20", "Trifid Nebula",        "EN",  18.043, -23.03, 6.3, 28.0),
    ("M21", "",                     "OC",  18.077, -22.50, 5.9, 13.0),
    ("M22", "",                     "GC",  18.607, -23.90, 5.1, 32.0),
    ("M23", "",                     "OC",  17.950, -19.02, 5.5, 27.0),
    ("M24", "Milky Way Star Cloud", "SC",  18.283, -18.55, 4.5,120.0),
    ("M25", "",                     "OC",  18.527, -19.23, 4.6, 32.0),
    ("M26", "",                     "OC",  18.755,  -9.39, 8.0,  8.0),
    ("M27", "Dumbbell Nebula",      "PN",  19.993,  22.72, 7.4,  8.0),
    ("M28", "",                     "GC",  18.408, -24.87, 6.8, 11.0),
    ("M29", "",                     "OC",  20.398,  38.51, 6.6,  7.0),
    ("M30", "",                     "GC",  21.673, -23.18, 7.2, 11.0),
    ("M31", "Andromeda Galaxy",     "Gx",   0.712,  41.27, 3.4,178.0),
    ("M32", "",                     "Gx",   0.712,  40.87, 8.7,  8.0),
    ("M33", "Triangulum Galaxy",    "Gx",   1.564,  30.66, 5.7, 73.0),
    ("M34", "",                     "OC",   2.703,  42.78, 5.5, 35.0),
    ("M35", "",                     "OC",   6.147,  24.33, 5.3, 28.0),
    ("M36", "",                     "OC",   5.603,  34.13, 6.0, 12.0),
    ("M37", "",                     "OC",   5.873,  32.55, 5.6, 24.0 ),
    ("M38", "",                     "OC",   5.480,  35.85, 6.4, 21.0),
    ("M39", "",                     "OC",  21.533,  48.43, 4.6, 32.0),
    ("M40", "Winnecke 4",           "Db",  12.370,  58.08, 8.4,  1.0),
    ("M41", "",                     "OC",   6.767, -20.72, 4.5, 38.0),
    ("M42", "Orion Nebula",         "EN",   5.588,  -5.39, 4.0, 85.0),
    ("M43", "De Mairan Nebula",     "EN",   5.592,  -5.27, 9.0, 20.0),
    ("M44", "Beehive Cluster",      "OC",   8.670,  19.98, 3.7, 95.0),
    ("M45", "Pleiades",             "OC",   3.790,  24.12, 1.6,110.0),
    ("M46", "",                     "OC",   7.697, -14.81, 6.0, 27.0),
    ("M47", "",                     "OC",   7.613, -14.50, 4.4, 30.0),
    ("M48", "",                     "OC",   8.233,  -5.80, 5.5, 54.0),
    ("M49", "",                     "Gx",  12.497,   8.00, 8.4,  9.0),
    ("M50", "",                     "OC",   7.050,  -8.38, 5.9, 16.0),
    ("M51", "Whirlpool Galaxy",     "Gx",  13.498,  47.20, 8.4, 11.0),
    ("M52", "",                     "OC",  23.403,  61.59, 6.9, 13.0),
    ("M53", "",                     "GC",  13.215,  18.17, 7.6, 13.0),
    ("M54", "",                     "GC",  18.917, -30.48, 7.6,  9.0),
    ("M55", "",                     "GC",  19.667, -30.97, 6.3, 19.0),
    ("M56", "",                     "GC",  19.277,  30.18, 8.3,  7.0),
    ("M57", "Ring Nebula",          "PN",  18.893,  33.03, 8.8,  1.5),
    ("M58", "",                     "Gx",  12.627,  11.82, 9.7,  6.0),
    ("M59", "",                     "Gx",  12.700,  11.65, 9.6,  5.0),
    ("M60", "",                     "Gx",  12.727,  11.55, 8.8,  7.0),
    ("M61", "",                     "Gx",  12.357,   4.47, 9.7,  6.0),
    ("M62", "",                     "GC",  17.020, -30.12, 6.5, 15.0),
    ("M63", "Sunflower Galaxy",     "Gx",  13.263,  42.03, 8.6, 12.0),
    ("M64", "Black Eye Galaxy",     "Gx",  12.945,  21.68, 8.5, 10.0),
    ("M65", "",                     "Gx",  11.312,  13.09, 9.3,  9.0),
    ("M66", "",                     "Gx",  11.337,  12.99, 8.9,  9.0),
    ("M67", "",                     "OC",   8.860,  11.82, 6.1, 25.0),
    ("M68", "",                     "GC",  12.657, -26.75, 7.8, 12.0),
    ("M69", "",                     "GC",  18.523, -32.35, 7.6,  9.0),
    ("M70", "",                     "GC",  18.720, -32.29, 7.9,  8.0),
    ("M71", "",                     "GC",  19.897,  18.78, 6.1,  7.0),
    ("M72", "",                     "GC",  20.890, -12.54, 9.3,  6.0),
    ("M73", "",                     "Ast", 20.978, -12.63, 9.0,  3.0),
    ("M74", "",                     "Gx",   1.610,  15.78, 9.4, 10.0),
    ("M75", "",                     "GC",  20.100, -21.92, 8.5,  6.0),
    ("M76", "Little Dumbbell",      "PN",   1.703,  51.57,10.1,  3.0),
    ("M77", "",                     "Gx",   2.712,  -0.01, 8.9,  7.0),
    ("M78", "",                     "RN",   5.780,   0.08, 8.3,  8.0),
    ("M79", "",                     "GC",   5.402, -24.52, 7.7,  9.0),
    ("M80", "",                     "GC",  16.280, -22.97, 7.3,  9.0),
    ("M81", "Bode's Galaxy",        "Gx",   9.927,  69.07, 6.9, 21.0),
    ("M82", "Cigar Galaxy",         "Gx",   9.928,  69.68, 8.4, 11.0),
    ("M83", "Southern Pinwheel",    "Gx",  13.617, -29.87, 7.5, 13.0),
    ("M84", "",                     "Gx",  12.421,  12.89, 9.1,  5.0),
    ("M85", "",                     "Gx",  12.424,  18.19, 9.1,  7.0),
    ("M86", "",                     "Gx",  12.437,  12.95, 8.9,  7.0),
    ("M87", "Virgo A",              "Gx",  12.514,  12.39, 8.6,  7.0),
    ("M88", "",                     "Gx",  12.533,  14.42, 9.6,  7.0),
    ("M89", "",                     "Gx",  12.594,  12.56, 9.8,  5.0),
    ("M90", "",                     "Gx",  12.614,  13.16, 9.5,  9.0),
    ("M91", "",                     "Gx",  12.591,  14.50, 9.7,  5.0),
    ("M92", "",                     "GC",  17.285,  43.13, 6.4, 14.0),
    ("M93", "",                     "OC",   7.742, -23.87, 6.0, 22.0),
    ("M94", "",                     "Gx",  12.852,  41.12, 8.2,  7.0),
    ("M95", "",                     "Gx",  10.700,  11.70, 9.7,  7.0),
    ("M96", "",                     "Gx",  10.780,  11.82, 9.2,  7.0),
    ("M97", "Owl Nebula",           "PN",  11.247,  55.02, 9.9,  3.4),
    ("M98", "",                     "Gx",  12.230,  14.90,10.1,  9.0),
    ("M99", "",                     "Gx",  12.313,  14.42, 9.9,  5.0),
    ("M100","",                     "Gx",  12.381,  15.82, 9.3,  7.0),
    ("M101","Pinwheel Galaxy",      "Gx",  14.053,  54.35, 7.9, 22.0),
    ("M102","",                     "Gx",  15.107,  55.77,10.0,  5.0),
    ("M103","",                     "OC",   1.553,  60.66, 7.4,  6.0),
    ("M104","Sombrero Galaxy",      "Gx",  12.666, -11.62, 8.0,  9.0),
    ("M105","",                     "Gx",  10.798,  12.58, 9.3,  5.0),
    ("M106","",                     "Gx",  12.317,  47.30, 8.4, 18.0),
    ("M107","",                     "GC",  16.542, -13.05, 7.9, 10.0),
    ("M108","",                     "Gx",  11.192,  55.67,10.0,  9.0),
    ("M109","",                     "Gx",  11.957,  53.38, 9.8,  8.0),
    ("M110","",                     "Gx",   0.673,  41.68, 8.7, 17.0),
]

CALDWELL_SOUTH = [
    ("C77",  "Centaurus A",           "Gx",  13.426, -43.02, 6.8,  21.0),
    ("C80",  "Omega Centauri",        "GC",  13.447, -47.48, 3.9,  36.0),
    ("C92",  "Eta Carinae Nebula",    "EN",  10.740, -59.87, 1.0, 120.0),
    ("C94",  "Jewel Box",             "OC",  12.897, -60.37, 4.2,  10.0),
    ("C99",  "Coalsack Nebula",       "DN",  12.550, -63.00, None,420.0),
    ("LMC",  "Large Magellanic Cloud","Irr",  5.383, -69.76, 0.9, 650.0),
    ("SMC",  "Small Magellanic Cloud","Irr",  0.875, -72.83, 2.7, 320.0),
]

ALGOL = {
    "ra_h":       3.136,
    "dec_d":     40.95,
    "period_days": 2.867315,
    "t0_jd":    2447041.852,
    "mag_min":    3.4,
    "mag_max":    2.1,
}

METEOR_SHOWERS = [
    # id, name, ra_h, dec_d, peak_month, peak_day, zhr, dur_days, speed_kms, parent
    ("QUA","Quadrantids",      15.3, 49.5, 1,  3, 120, 2, 41, "2003 EH1"),
    ("LYR","Lyrids",           18.1, 33.5, 4, 22,  18, 4, 49, "C/1861 G1 Thatcher"),
    ("ETA","Eta Aquariids",    22.5, -1.0, 5,  6,  50, 4, 66, "1P/Halley"),
    ("SDA","S. Delta Aquariids",22.7,-16.0,7, 29,  25, 5, 41, "96P/Machholz"),
    ("PER","Perseids",          3.1, 58.0, 8, 13, 100, 3, 59, "109P/Swift-Tuttle"),
    ("ORI","Orionids",          6.3, 15.5,10, 21,  25, 4, 66, "1P/Halley"),
    ("TAU","Taurids",           3.7, 14.0,11,  5,  10,20, 27, "2P/Encke"),
    ("LEO","Leonids",          10.1, 21.5,11, 17,  20, 2, 71, "55P/Tempel-Tuttle"),
    ("GEM","Geminids",          7.5, 32.5,12, 14, 150, 3, 35, "3200 Phaethon"),
    ("URS","Ursids",           14.5, 75.5,12, 22,  10, 2, 33, "8P/Tuttle"),
]


# ─────────────────────────────────────────────────────────────
# VISIBILITY HELPERS
# ─────────────────────────────────────────────────────────────

def _observer(eph, lat: float, lon: float):
    return eph["earth"] + wgs84.latlon(lat, lon)


def sun_alt_at(ts, eph, utc_dt: datetime, lat: float, lon: float) -> float:
    t   = ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
    obs = _observer(eph, lat, lon)
    alt, _, _ = obs.at(t).observe(eph["sun"]).apparent().altaz()
    return alt.degrees


def radec_alt_at(ts, eph, utc_dt: datetime, ra_h: float, dec_d: float,
                 lat: float, lon: float) -> float:
    t    = ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
    obs  = _observer(eph, lat, lon)
    star = Star(ra_hours=ra_h, dec_degrees=dec_d)
    alt, _, _ = obs.at(t).observe(star).apparent().altaz()
    return alt.degrees


def moon_illum_at(ts, eph, utc_dt: datetime) -> float:
    t    = ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
    e    = eph["earth"]
    sp   = e.at(t).observe(eph["sun"]).apparent()
    mp   = e.at(t).observe(eph["moon"]).apparent()
    sep  = sp.separation_from(mp).degrees
    return round((1 - math.cos(math.radians(sep))) / 2, 3)


def local_hhmm(utc_dt: datetime, tz_str: str) -> str:
    tz  = pytz.timezone(tz_str)
    loc = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz)
    return loc.strftime("%H:%M")


def dec_visible_from(dec_d: float, lat: float) -> bool:
    """Geometric check: object must have dec > lat − 90 to ever rise."""
    return dec_d > (lat - 90.0)


def visibility_record(ts, eph, utc_dt: datetime, ra_h, dec_d,
                      ev_cat: str, loc_code: str) -> dict:
    loc   = LOCATIONS[loc_code]
    lat   = loc["lat"]
    lon   = loc["lon"]
    tz    = loc["tz"]
    c     = loc["country"]
    month = utc_dt.month
    min_a = MIN_ALT.get(ev_cat, 10)

    # Hard geometric cut
    if ra_h is not None and dec_d is not None:
        if not dec_visible_from(dec_d, lat):
            return {"v": False, "alt": -99, "lt": local_hhmm(utc_dt, tz),
                    "dk": False, "cut": "geometry"}

    try:
        if ra_h is not None and dec_d is not None:
            alt = round(radec_alt_at(ts, eph, utc_dt, ra_h, dec_d, lat, lon), 1)
        else:
            alt = 90.0   # Global moment (solstice, lunar phase)
    except Exception:
        alt = -99.0

    try:
        s_alt = round(sun_alt_at(ts, eph, utc_dt, lat, lon), 1)
    except Exception:
        s_alt = 0.0

    # Explicitly cast to native Python floats and bools for JSON serialization
    dark    = bool(s_alt <= SUN_NAUTICAL)
    visible = bool(alt >= min_a and dark)

    rec = {"v": visible, "alt": float(alt), "lt": local_hhmm(utc_dt, tz),
           "dk": dark, "s_alt": float(s_alt)}

    if ev_cat == "meteor" and alt > 0:
        rec["zhr_f"] = round(math.sin(math.radians(max(alt, 0))), 2)

    # Regional context flags
    if c == "IN"  and month in (6, 7, 8, 9):              rec["monsoon"]       = True
    if c == "PH"  and month in (6, 7, 8, 9, 10, 11):      rec["typhoon"]       = True
    if c == "NG"  and month in (11, 12, 1):                rec["harmattan"]     = True
    if c in ("GB","DE") and month in (6,7) and abs(lat)>50:rec["short_night"]   = True
    if c == "CA"  and month in (6, 7) and lat > 55:        rec["midnight_sun"]  = True
    if c in ("GB","CA") and lat > 54 and month in (9,10,11,12,1,2,3):
        rec["aurora_possible"] = True

    return rec


# ─────────────────────────────────────────────────────────────
# EVENT GENERATORS
# ─────────────────────────────────────────────────────────────

def gen_lunar_phases(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year,  1,  1)
    t1 = ts.utc(year, 12, 31, 23, 59)
    times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    names = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]
    earth = eph["earth"]
    moon  = eph["moon"]
    full_by_month: dict = {}
    events = []

    for t, ph in zip(times, phases):
        utc_dt  = t.utc_datetime()
        name    = names[ph]
        subtype = None
        dist_km = None

        if ph == 2:  # Full Moon
            _, _, d = earth.at(t).observe(moon).apparent().radec()
            dist_km = round(d.km, 0)
            if dist_km < 362_000:
                name, subtype = "Supermoon (Full Moon)", "supermoon"
            elif dist_km > 404_000:
                name, subtype = "Micromoon (Full Moon)", "micromoon"
            key = (utc_dt.year, utc_dt.month)
            full_by_month[key] = full_by_month.get(key, 0) + 1
            if full_by_month[key] == 2:
                name, subtype = "Blue Moon (2nd Full Moon)", "blue_moon"

        events.append({
            "id":          f"LUN_{ph}_{utc_dt.strftime('%Y%m%d')}",
            "category":    "lunar",
            "type":        name,
            "subtype":     subtype,
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        None,
            "dec_d":       None,
            "distance_km": dist_km,
            "magnitude":   -12.6 if ph == 2 else None,
            "global_event":True,
            "description": _lunar_desc(ph, subtype, dist_km),
        })
    return events


def _lunar_desc(ph: int, sub, dist) -> str:
    base = {
        0: "New Moon — no Moon in sky, ideal for deep-sky observing.",
        1: "First Quarter — half-lit Moon in evening sky.",
        2: "Full Moon — bright all night.",
        3: "Last Quarter — Moon rises near midnight.",
    }[ph]
    if sub == "supermoon":
        return f"Supermoon at {int(dist):,} km — 14% larger, 30% brighter than average. {base}"
    if sub == "micromoon":
        return f"Micromoon at {int(dist):,} km — slightly smaller than average. {base}"
    if sub == "blue_moon":
        return f"Blue Moon — second Full Moon in one calendar month. {base}"
    return base


def gen_lunar_eclipses(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year,  1,  1)
    t1 = ts.utc(year, 12, 31, 23, 59)
    times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    earth = eph["earth"]
    moon  = eph["moon"]
    sun   = eph["sun"]
    events = []

    for t, ph in zip(times, phases):
        if ph != 2:
            continue
        utc_dt = t.utc_datetime()
        e_pos  = earth.at(t)
        s_app  = e_pos.observe(sun).apparent()
        m_app  = e_pos.observe(moon).apparent()
        sep_deg = s_app.separation_from(m_app).degrees

        _, _, m_dist_r = e_pos.observe(moon).apparent().radec()
        _, _, s_dist_r = e_pos.observe(sun).apparent().radec()
        dist_moon_km = m_dist_r.km
        dist_sun_km  = s_dist_r.km

        r_sun_km   = 696_000
        r_earth_km = 6_371
        r_moon_km  = 1_737.4

        umbra_angle    = math.degrees(math.atan((r_sun_km - r_earth_km) / dist_sun_km))
        umbra_radius   = r_earth_km - math.tan(math.radians(umbra_angle)) * dist_moon_km
        pen_angle      = math.degrees(math.atan((r_sun_km + r_earth_km) / dist_sun_km))
        pen_radius     = r_earth_km + math.tan(math.radians(pen_angle))  * dist_moon_km

        miss_angle_deg = abs(sep_deg - 180.0)
        miss_km        = math.tan(math.radians(miss_angle_deg)) * dist_moon_km

        if miss_km < umbra_radius + r_moon_km:
            if miss_km + r_moon_km < umbra_radius:
                eclipse_type = "Total Lunar Eclipse"
                duration_min = max(1, int(120 * (1 - miss_km / umbra_radius)))
            else:
                eclipse_type = "Partial Lunar Eclipse"
                duration_min = max(1, int(80 * (1 - miss_km / (umbra_radius + r_moon_km))))
        elif miss_km < pen_radius + r_moon_km:
            eclipse_type = "Penumbral Lunar Eclipse"
            duration_min = max(1, int(200 * (1 - miss_km / (pen_radius + r_moon_km))))
        else:
            continue

        events.append({
            "id":          f"LEC_{utc_dt.strftime('%Y%m%d')}",
            "category":    "eclipse",
            "type":        eclipse_type,
            "subtype":     "lunar_eclipse",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        None,
            "dec_d":       None,
            "duration_min":duration_min,
            "magnitude":   -12.6,
            "global_event":False,
            "description": f"{eclipse_type} — duration ~{duration_min} min. Visible wherever Moon is above horizon.",
        })
    return events


def gen_solar_eclipses(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year,  1,  1)
    t1 = ts.utc(year, 12, 31, 23, 59)
    times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    earth = eph["earth"]
    moon  = eph["moon"]
    sun   = eph["sun"]
    events = []

    for t, ph in zip(times, phases):
        if ph != 0:
            continue
        utc_dt  = t.utc_datetime()
        e_pos   = earth.at(t)
        s_app   = e_pos.observe(sun).apparent()
        m_app   = e_pos.observe(moon).apparent()
        sep_deg = s_app.separation_from(m_app).degrees

        _, _, m_dist_r = e_pos.observe(moon).apparent().radec()
        _, _, s_dist_r = e_pos.observe(sun).apparent().radec()
        dist_moon_km = m_dist_r.km
        dist_sun_km  = s_dist_r.km
        r_sun_km  = 696_000
        r_moon_km = 1_737.4
        r_earth_km = 6_371

        ang_sun_deg  = math.degrees(math.atan(r_sun_km  / dist_sun_km))
        ang_moon_deg = math.degrees(math.atan(r_moon_km / dist_moon_km))

        if sep_deg > ang_sun_deg + ang_moon_deg + 0.5:
            continue

        miss_km = math.tan(math.radians(sep_deg)) * dist_sun_km

        if ang_moon_deg >= ang_sun_deg:
            eclipse_type = "Total Solar Eclipse"
        elif miss_km < r_earth_km:
            eclipse_type = "Annular Solar Eclipse"
        else:
            eclipse_type = "Partial Solar Eclipse"

        mag = round(ang_moon_deg / ang_sun_deg, 3)
        events.append({
            "id":          f"SEC_{utc_dt.strftime('%Y%m%d')}",
            "category":    "eclipse",
            "type":        eclipse_type,
            "subtype":     "solar_eclipse",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        None,
            "dec_d":       None,
            "magnitude":   mag,
            "global_event":False,
            "description": f"{eclipse_type} — magnitude {mag}. Visible along central path.",
        })
    return events


def gen_seasons(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year,  1,  1)
    t1 = ts.utc(year, 12, 31, 23, 59)
    times, idxs = almanac.find_discrete(t0, t1, almanac.seasons(eph))
    names = [
        "March Equinox (Spring NH / Autumn SH)",
        "June Solstice (Summer NH / Winter SH)",
        "September Equinox (Autumn NH / Spring SH)",
        "December Solstice (Winter NH / Summer SH)",
    ]
    events = []
    for t, idx in zip(times, idxs):
        utc_dt = t.utc_datetime()
        events.append({
            "id":          f"SEA_{idx}_{year}",
            "category":    "season",
            "type":        names[idx],
            "subtype":     "equinox" if idx in (0, 2) else "solstice",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        None,
            "dec_d":       None,
            "global_event":True,
            "description": f"{names[idx]} — exact moment Sun crosses ecliptic milestone.",
        })
    return events


def _ecl_lon(eph, ts, body_key: str, t) -> float:
    pos = eph["earth"].at(t).observe(eph[body_key]).apparent()
    _, lon, _ = pos.frame_latlon(ecliptic_frame)
    return lon.degrees % 360.0


def _sun_ecl_lon(eph, ts, t) -> float:
    pos = eph["earth"].at(t).observe(eph["sun"]).apparent()
    _, lon, _ = pos.frame_latlon(ecliptic_frame)
    return lon.degrees % 360.0


def gen_planetary_events(ts, eph, year: int) -> list[dict]:
    events = []
    earth  = eph["earth"]

    # Outer planet oppositions
    outer = [
        ("MARS",    "mars",               2, -2.9),
        ("JUPITER", "jupiter barycenter", 2, -2.9),
        ("SATURN",  "saturn barycenter",  3,  0.5),
        ("URANUS",  "uranus barycenter",  3,  5.7),
        ("NEPTUNE", "neptune barycenter", 3,  7.8),
    ]
    for label, bkey, step, ref_mag in outer:
        diffs, dates = [], []
        d   = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
        while d < end:
            t    = ts.from_datetime(d)
            diff = abs((_ecl_lon(eph, ts, bkey, t) - _sun_ecl_lon(eph, ts, t) + 180) % 360 - 180)
            diffs.append(diff)
            dates.append(d)
            d += timedelta(days=step)

        for i in range(1, len(diffs) - 1):
            if diffs[i] < diffs[i-1] and diffs[i] < diffs[i+1] and diffs[i] < 3.0:
                utc_dt = dates[i]
                t2 = ts.from_datetime(utc_dt)
                ra_, dec_, dist_ = earth.at(t2).observe(eph[bkey]).apparent().radec()
                events.append({
                    "id":          f"OPP_{label}_{utc_dt.strftime('%Y%m%d')}",
                    "category":    "planet",
                    "type":        f"{label.title()} at Opposition",
                    "subtype":     "opposition",
                    "planet":      label,
                    "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ra_h":        round(ra_.hours, 3),
                    "dec_d":       round(dec_.degrees, 2),
                    "distance_km": round(dist_.km, 0),
                    "magnitude":   ref_mag,
                    "global_event":True,
                    "description": (f"{label.title()} at opposition — visible all night, "
                                    f"closest/brightest. Magnitude ~{ref_mag}."),
                })

    # Inner planet elongations
    inner = [("MERCURY", "mercury", -0.4), ("VENUS", "venus", -4.0)]
    for label, bkey, ref_mag in inner:
        elongs, dates = [], []
        d   = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
        while d < end:
            t     = ts.from_datetime(d)
            elong = (_ecl_lon(eph, ts, bkey, t) - _sun_ecl_lon(eph, ts, t) + 180) % 360 - 180
            elongs.append(elong)
            dates.append(d)
            d += timedelta(days=1)

        for i in range(1, len(elongs) - 1):
            if elongs[i] > elongs[i-1] and elongs[i] > elongs[i+1] and elongs[i] > 10:
                utc_dt = dates[i]
                t2 = ts.from_datetime(utc_dt)
                ra_, dec_, _ = earth.at(t2).observe(eph[bkey]).apparent().radec()
                events.append({
                    "id":             f"ELO_{label}_E_{utc_dt.strftime('%Y%m%d')}",
                    "category":       "planet",
                    "type":           f"{label.title()} Max Eastern Elongation",
                    "subtype":        "elongation_east",
                    "planet":         label,
                    "utc":            utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ra_h":           round(ra_.hours, 3),
                    "dec_d":          round(dec_.degrees, 2),
                    "elongation_deg": round(elongs[i], 1),
                    "magnitude":      ref_mag,
                    "global_event":   True,
                    "description":    (f"{label.title()} at max eastern elongation "
                                       f"({round(elongs[i],1)}°) — best evening visibility."),
                })
            if elongs[i] < elongs[i-1] and elongs[i] < elongs[i+1] and elongs[i] < -10:
                utc_dt = dates[i]
                t2 = ts.from_datetime(utc_dt)
                ra_, dec_, _ = earth.at(t2).observe(eph[bkey]).apparent().radec()
                events.append({
                    "id":             f"ELO_{label}_W_{utc_dt.strftime('%Y%m%d')}",
                    "category":       "planet",
                    "type":           f"{label.title()} Max Western Elongation",
                    "subtype":        "elongation_west",
                    "planet":         label,
                    "utc":            utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ra_h":           round(ra_.hours, 3),
                    "dec_d":          round(dec_.degrees, 2),
                    "elongation_deg": round(abs(elongs[i]), 1),
                    "magnitude":      ref_mag,
                    "global_event":   True,
                    "description":    (f"{label.title()} at max western elongation "
                                       f"({round(abs(elongs[i]),1)}°) — best morning visibility."),
                })

    # Planetary conjunctions (all pairs < 2°)
    all_planets = [
        ("MERCURY","mercury"),
        ("VENUS",  "venus"),
        ("MARS",   "mars"),
        ("JUPITER","jupiter barycenter"),
        ("SATURN", "saturn barycenter"),
    ]
    pairs = [(all_planets[i], all_planets[j])
             for i in range(len(all_planets))
             for j in range(i+1, len(all_planets))]

    for (l1, b1), (l2, b2) in pairs:
        in_conj  = False
        min_sep  = 999.0
        min_date = None
        d   = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
        while d < end:
            t  = ts.from_datetime(d)
            p1 = earth.at(t).observe(eph[b1]).apparent()
            p2 = earth.at(t).observe(eph[b2]).apparent()
            sep = p1.separation_from(p2).degrees
            if sep < 2.0:
                in_conj = True
                if sep < min_sep:
                    min_sep, min_date = sep, d
            else:
                if in_conj and min_date:
                    t2 = ts.from_datetime(min_date)
                    ra_, dec_, _ = earth.at(t2).observe(eph[b1]).apparent().radec()
                    events.append({
                        "id":             f"CONJ_{l1}_{l2}_{min_date.strftime('%Y%m%d')}",
                        "category":       "planet",
                        "type":           f"{l1.title()}-{l2.title()} Conjunction",
                        "subtype":        "conjunction",
                        "bodies":         [l1, l2],
                        "utc":            min_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "ra_h":           round(ra_.hours, 3),
                        "dec_d":          round(dec_.degrees, 2),
                        "separation_deg": round(min_sep, 2),
                        "global_event":   True,
                        "description":    (f"{l1.title()} and {l2.title()} only "
                                           f"{round(min_sep,2)}° apart — fit in one binocular view."),
                    })
                in_conj, min_sep, min_date = False, 999.0, None
            d += timedelta(days=1)

    return events


def gen_meteor_showers(ts, eph, year: int) -> list[dict]:
    events = []
    for sid, name, ra_h, dec_d, pm, pd, zhr, dur, spd, parent in METEOR_SHOWERS:
        peak_utc = datetime(year, pm, pd, 4, 0, tzinfo=timezone.utc)
        moon_pct = moon_illum_at(ts, eph, peak_utc)
        events.append({
            "id":                f"MTR_{sid}_{year}",
            "category":          "meteor",
            "type":              f"{name} Meteor Shower",
            "subtype":           "meteor_shower",
            "shower_id":         sid,
            "utc":               peak_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end":           (peak_utc + timedelta(days=dur)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":              ra_h,
            "dec_d":             dec_d,
            "zhr":               zhr,
            "speed_kms":         spd,
            "parent_body":       parent,
            "moon_illumination": moon_pct,
            "moon_interference": moon_pct > 0.50,
            "global_event":      False,
            "description":       (f"{name} peak — up to {zhr} meteors/hr from dark site. "
                                  f"Speed: {spd} km/s. Parent: {parent}. "
                                  f"Moon: {int(moon_pct*100)}% — "
                                  f"{'⚠ interference' if moon_pct > 0.5 else '✓ dark sky'}."),
        })
    return events


def gen_algol_minima(year: int) -> list[dict]:
    ref_jd = ALGOL["t0_jd"]
    period = ALGOL["period_days"]
    y_start = (datetime(year,    1, 1, tzinfo=timezone.utc).timestamp() / 86400) + 2440587.5
    y_end   = (datetime(year+1, 1, 1, tzinfo=timezone.utc).timestamp() / 86400) + 2440587.5
    n0 = math.ceil( (y_start - ref_jd) / period)
    n1 = math.floor((y_end   - ref_jd) / period)
    events = []
    for n in range(n0, n1 + 1):
        jd     = ref_jd + n * period
        ts_sec = (jd - 2440587.5) * 86400
        utc_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        if utc_dt.year != year:
            continue
        events.append({
            "id":          f"ALG_{utc_dt.strftime('%Y%m%d_%H%M')}",
            "category":    "variable",
            "type":        "Algol Minimum (Beta Persei)",
            "subtype":     "eclipsing_binary",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        ALGOL["ra_h"],
            "dec_d":       ALGOL["dec_d"],
            "mag_at_min":  ALGOL["mag_min"],
            "mag_at_max":  ALGOL["mag_max"],
            "duration_hrs":10,
            "global_event":False,
            "description": ("Algol dims from mag 2.1 → 3.4 over ~5 hours — "
                            "watch the star fade and recover with the naked eye."),
        })
    return events


def gen_zodiacal_light(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year, 12, 31)
    times, idxs = almanac.find_discrete(t0, t1, almanac.seasons(eph))

    equinox: dict[str, datetime] = {}
    for t, idx in zip(times, idxs):
        if idx == 0: equinox["vernal"]   = t.utc_datetime()
        if idx == 2: equinox["autumnal"] = t.utc_datetime()

    events = []
    if "vernal" in equinox:
        ve = equinox["vernal"]
        events.append({
            "id":          f"ZOD_NH_SPRING_{year}",
            "category":    "atmosphere",
            "type":        "Zodiacal Light — Spring Evening Window (NH)",
            "subtype":     "zodiacal_light",
            "utc":         (ve - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end":     (ve + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":None, "dec_d":None, "global_event":False, "hemisphere":"north",
            "description": "Zodiacal light visible in west after twilight — need dark sky, no Moon, clear western horizon.",
        })

    if "autumnal" in equinox:
        ae = equinox["autumnal"]
        events.append({
            "id":          f"ZOD_NH_AUTUMN_{year}",
            "category":    "atmosphere",
            "type":        "Zodiacal Light — Autumn Morning Window (NH)",
            "subtype":     "zodiacal_light",
            "utc":         (ae - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end":     (ae + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":None, "dec_d":None, "global_event":False, "hemisphere":"north",
            "description": "Zodiacal light visible in east before dawn. Best Sept-Oct from NH.",
        })
        events.append({
            "id":          f"ZOD_SH_SPRING_{year}",
            "category":    "atmosphere",
            "type":        "Zodiacal Light — Spring Evening Window (SH)",
            "subtype":     "zodiacal_light",
            "utc":         (ae - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end":     (ae + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":None, "dec_d":None, "global_event":False, "hemisphere":"south",
            "description": "Zodiacal light spring window for AU, ZA, NG, PH.",
        })
    return events


def gen_deep_sky_highlights() -> tuple[list, list]:
    messier    = [{"id":m[0],"name":m[1],"type":m[2],"ra_h":m[3],
                   "dec_d":m[4],"mag":m[5],"size_am":m[6]} for m in MESSIER_CATALOG]
    caldwell_s = [{"id":c[0],"name":c[1],"type":c[2],"ra_h":c[3],
                   "dec_d":c[4],"mag":c[5],"size_am":c[6]} for c in CALDWELL_SOUTH]
    return messier, caldwell_s


# ─────────────────────────────────────────────────────────────
# DAILY — ASTEROIDS (NASA NeoWs) with dual-key fallback
# ─────────────────────────────────────────────────────────────

def fetch_asteroid_week(start: str, end: str) -> list[dict]:
    """
    Fetch close-approach NEOs for a 7-day window.
    Uses nasa_get() which automatically rotates between NASA_API_KEY_1
    and NASA_API_KEY_2 on rate-limit errors.
    """
    resp = nasa_get(
        "https://api.nasa.gov/neo/rest/v1/feed",
        {"start_date": start, "end_date": end},
    )
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "N/A"
        log.warning(f"NeoWs failed for {start}→{end} (HTTP {code})")
        return []

    events = []
    for date_key, neos in resp.json().get("near_earth_objects", {}).items():
        for neo in neos:
            cad     = neo["close_approach_data"][0]
            dist_ld = float(cad["miss_distance"]["lunar"])
            if dist_ld > 15:
                continue
            d_m   = neo["estimated_diameter"]["meters"]
            avg_d = (d_m["estimated_diameter_min"] + d_m["estimated_diameter_max"]) / 2
            try:
                utc_dt = datetime.strptime(cad["close_approach_date_full"], "%Y-%b-%d %H:%M")
            except Exception:
                utc_dt = datetime.strptime(date_key, "%Y-%m-%d")
            events.append({
                "id":          f"AST_{neo['id']}_{utc_dt.strftime('%Y%m%d')}",
                "category":    "asteroid",
                "type":        "Asteroid Close Approach",
                "subtype":     "near_earth_object",
                "name":        neo["name"],
                "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":        None,
                "dec_d":       None,
                "distance_ld": round(dist_ld, 2),
                "diameter_m":  round(avg_d, 0),
                "hazardous":   bool(neo.get("is_potentially_hazardous_asteroid", False)),
                "global_event":True,
                "description": (f"{neo['name']} passes Earth at {round(dist_ld,2)} lunar distances "
                                f"({round(dist_ld*384400,0):,.0f} km). "
                                f"Estimated diameter: {round(avg_d,0):.0f} m."),
            })
    return events


def fetch_all_asteroids_year(year: int) -> list[dict]:
    """Fetch all 52 weeks; pause 1.2 s between requests to stay within rate limits."""
    events: list[dict] = []
    d   = datetime(year, 1, 1)
    end = datetime(year, 12, 28)
    while d <= end:
        s = d.strftime("%Y-%m-%d")
        e = (d + timedelta(days=6)).strftime("%Y-%m-%d")
        log.info(f"  Asteroids {s} → {e}")
        try:
            events.extend(fetch_asteroid_week(s, e))
        except Exception as ex:
            log.warning(f"  Asteroid week {s} failed: {ex}")
        time.sleep(1.2)
        d += timedelta(days=7)

    # Deduplicate
    seen: set[str] = set()
    unique = [ev for ev in events if not (ev["id"] in seen or seen.add(ev["id"]))]
    log.info(f"  Total unique asteroid events: {len(unique)}")
    return unique


# ─────────────────────────────────────────────────────────────
# DAILY — COMETS (MPC orbital elements, correct Skyfield API)
# ─────────────────────────────────────────────────────────────



def fetch_comets_with_positions(ts, eph) -> list[dict]:
    """
    Downloads MPC comet orbital elements using an enterprise-grade retry
    session to bypass academic firewalls and intermittent timeouts.
    """
    events: list[dict] = []
    mpc_url = "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt"
    
    # 1. Setup a persistent session with Exponential Backoff
    session = requests.Session()
    retry = Retry(
        total=5,           # Try up to 5 times
        read=5,            # Retry on read timeouts
        connect=5,         # Retry on connection drops
        backoff_factor=2,  # Wait 2s, 4s, 8s, 16s between attempts
        status_forcelist=[403, 429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        log.info("  Fetching comets from MPC (with exponential backoff protection)...")
        # Extended base timeout to 45 seconds for slow servers
        resp = session.get(mpc_url, headers=headers, timeout=45)
        resp.raise_for_status()
        
        f = io.BytesIO(resp.content)
        comets_df = sk_mpc.load_comets_dataframe(f)
    except Exception as exc:
        log.error(f"  All MPC fetch attempts failed or timed out: {exc}")
        return events

    now_utc = datetime.now(timezone.utc)
    t_now   = ts.from_datetime(now_utc)
    earth   = eph["earth"]
    sun     = eph["sun"]

    for _, row in comets_df.iterrows():
        try:
            comet       = sun + sk_mpc.comet_orbit(row, ts, GM_SUN)
            astrometric = earth.at(t_now).observe(comet)
            ra, dec, dist = astrometric.radec()

            if dist.au > 5.0:
                continue  # Too faint / too far

            designation = str(row.get("designation", "Unknown"))
            events.append({
                "id":                  f"CMT_{designation.replace(' ','_')[:20]}",
                "category":            "comet",
                "type":                "Bright Comet Visible",
                "subtype":             "comet",
                "name":                designation,
                "utc":                 now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":                round(float(ra.hours), 3),
                "dec_d":               round(float(dec.degrees), 2),
                "distance_au":         round(float(dist.au), 3),
                "magnitude_estimate":  "check latest reports",
                "global_event":        False,
                "description":         (f"Comet {designation} currently at "
                                        f"RA {round(float(ra.hours),2)}h, "
                                        f"Dec {round(float(dec.degrees),1)}°, "
                                        f"dist {round(float(dist.au),2)} AU."),
            })
        except Exception:
            continue

    log.info(f"  Comets with real positions: {len(events)}")
    return events


# ─────────────────────────────────────────────────────────────
# VISIBILITY MATRIX
# ─────────────────────────────────────────────────────────────

def compute_visibility(ts, eph, events: list[dict],
                       country_filter: list[str] | None = None) -> dict:
    locs = {k: v for k, v in LOCATIONS.items()
            if country_filter is None or v["country"] in country_filter}
    log.info(f"  Computing {len(events)} events × {len(locs)} locations = {len(events)*len(locs)} checks")

    vis: dict = {}
    for i, ev in enumerate(events):
        ev_id  = ev["id"]
        ra_h   = ev.get("ra_h")
        dec_d  = ev.get("dec_d")
        cat    = ev.get("category", "planet")
        try:
            utc_dt = datetime.strptime(ev["utc"], "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue

        block: dict = {}
        for loc_code in locs:
            try:
                rec = visibility_record(ts, eph, utc_dt, ra_h, dec_d, cat, loc_code)
                if ev.get("moon_interference"):
                    rec["moon_warn"] = True
                block[loc_code] = rec
            except Exception as exc:
                block[loc_code] = {"v": False, "alt": -99, "lt": None,
                                   "dk": False, "err": str(exc)[:40]}
        vis[ev_id] = block

        if (i + 1) % 30 == 0:
            log.info(f"    {i+1}/{len(events)} events processed")

    return vis


# ─────────────────────────────────────────────────────────────
# OUTPUT BUILDERS
# ─────────────────────────────────────────────────────────────

def build_static(ts, eph, year: int) -> list[dict]:
    log.info(f"=== BUILDING STATIC JSON for {year} ===")
    all_events: list[dict] = []

    log.info("Lunar phases...")
    all_events += gen_lunar_phases(ts, eph, year)

    log.info("Lunar eclipses (computed)...")
    all_events += gen_lunar_eclipses(ts, eph, year)

    log.info("Solar eclipses (computed)...")
    all_events += gen_solar_eclipses(ts, eph, year)

    log.info("Seasons...")
    all_events += gen_seasons(ts, eph, year)

    log.info("Planetary events (scanning full year — may take 1–2 min)...")
    all_events += gen_planetary_events(ts, eph, year)

    log.info("Meteor showers...")
    all_events += gen_meteor_showers(ts, eph, year)

    log.info("Algol minima...")
    all_events += gen_algol_minima(year)

    log.info("Zodiacal light windows...")
    all_events += gen_zodiacal_light(ts, eph, year)

    all_events.sort(key=lambda e: e.get("utc", ""))
    log.info(f"Total events generated: {len(all_events)}")

    messier, caldwell_s = gen_deep_sky_highlights()

    doc = {
        "schema":    "xtrobe-2.1",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year":      year,
        "count":     len(all_events),
        "events":    all_events,
        "catalog": {
            "messier":        messier,
            "caldwell_south": caldwell_s,
            "algol":          ALGOL,
        },
        "locations_meta": {
            code: {"name": loc["name"], "lat": loc["lat"],
                   "lon":  loc["lon"],  "country": loc["country"]}
            for code, loc in LOCATIONS.items()
        },
    }

    path = os.path.join(OUTPUT_DIR, "static_events.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")
    return all_events


def build_visibility_files(ts, eph, events: list[dict],
                           country_limit: str | None = None) -> None:
    cc = {country_limit: COUNTRY_CODES.get(country_limit, [])} \
         if country_limit else COUNTRY_CODES

    for country, codes in cc.items():
        log.info(f"=== VISIBILITY {country} ({len(codes)} locations) ===")
        vis = compute_visibility(ts, eph, events, country_filter=[country])
        doc = {
            "schema":    "xtrobe-2.1",
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "country":   country,
            "locations": {c: LOCATIONS[c] for c in codes},
            "visibility":vis,
        }
        path = os.path.join(OUTPUT_DIR, f"visibility_{country}.json")
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")


def build_daily(ts, eph, year: int) -> None:
    log.info("=== BUILDING DAILY UPDATE JSON ===")

    log.info("Fetching all asteroid close approaches (full year, 52 weeks)...")
    asteroids = fetch_all_asteroids_year(year)

    log.info("Fetching comets with real RA/Dec from MPC...")
    comets = fetch_comets_with_positions(ts, eph)

    daily_events = asteroids + comets
    log.info(f"Daily events: {len(asteroids)} asteroids + {len(comets)} comets")

    vis = compute_visibility(ts, eph, daily_events) if daily_events else {}

    doc = {
        "schema":    "xtrobe-2.1",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "events":    daily_events,
        "visibility":vis,
    }
    path = os.path.join(OUTPUT_DIR, "daily_update.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="XTROBE Calendar Generator v2.1 — fully dynamic, zero hardcoded dates"
    )
    p.add_argument("--type",    choices=["static", "daily", "both"], default="both",
                   help="What to generate (default: both)")
    p.add_argument("--year",    type=int, default=datetime.now(timezone.utc).year,
                   help="Year to generate (default: current year)")
    p.add_argument("--country", default=None,
                   help="Limit visibility output to one country code, e.g. IN")
    args = p.parse_args()

    # Init NASA keys from env (.env file or GitHub Actions secrets)
    _init_nasa_keys()
    log.info(f"NASA API keys available: {len(_NASA_KEYS)}")

    log.info("Loading Skyfield ephemeris (cached in skyfield_cache/ — downloads once only)...")
    ts, eph = init_skyfield()

    if args.type in ("static", "both"):
        events = build_static(ts, eph, args.year)
        build_visibility_files(ts, eph, events, country_limit=args.country)

    if args.type in ("daily", "both"):
        build_daily(ts, eph, args.year)

    log.info("=== COMPLETE ===")
    log.info(f"Output files written to: {OUTPUT_DIR}/")
    log.info("""
Frontend integration:
  1. Load static_events.json  — once per year
  2. Load visibility_XX.json  — once per year (XX = 2-letter country code)
  3. User picks state/region  → filter visibility[eventId][stateCode].v === true
  4. Display .lt (local time), .alt (altitude °), .dk (dark sky flag)
  5. Load daily_update.json   — every night (asteroids + comets only)

GitHub Actions cron (nightly):
  0 1 * * *  → runs --type daily, commits output/ to data branch
  0 2 1 1 *  → runs --type static on Jan 1 each year
""")


if __name__ == "__main__":
    main()
