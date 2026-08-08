# XTROBE Calendar Engine — Developer Documentation v5.8

> **System Architecture:** Clean Two-File Decoupled Architecture  
> **Output Files:** `output/yearly_events.json` (Static Yearly — 776 Events) & `output/daily_events.json` (Dynamic Daily — 4 Telemetry Streams)  
> **Execution Schedule (31-Minute Staggered IST Timeline):**  
>   - `daily_update.yml`: Runs on **git push**, **daily at 12:00 AM Midnight IST** (`18:30 UTC`), and manual dispatch.  
>   - `yearly_update.yml`: Runs **twice per year at 03:37 AM IST** (**Dec 25** for upcoming year & **June 25** for mid-year refresh).  

---

## Table of Contents
1. [Master 31-Minute Staggered Schedule (All Repositories in IST)](#1-master-31-minute-staggered-schedule-all-repositories-in-ist)
2. [Why Two Files? (Architectural Rationale)](#2-why-two-files-architectural-rationale)
3. [Complete Event Breakdown (Yearly vs. Daily)](#3-complete-event-breakdown-yearly-vs-daily)
4. [JSON Schema Specifications](#4-json-schema-specifications)
5. [Dynamic Daily Open API Data Streams](#5-dynamic-daily-open-api-data-streams)
6. [Frontend Integration Guide (Merging 2 Files on Client)](#6-frontend-integration-guide-merging-2-files-on-client)

---

## 1. Master 31-Minute Staggered Schedule (All Repositories in IST)

To prevent GitHub Actions concurrency limits and API rate-limiting, **all workflow jobs across all 3 active repositories are staggered exactly 31 minutes apart starting from 12:00 AM Midnight IST**:

| Step | IST Time | UTC Time | Cron Expression | Workflow File | Repository | Frequency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **12:00 AM IST** | `18:30 UTC` | `30 18 * * *` | `daily_update.yml` | `Calender-astronomical` | **Daily** |
| **2** | **12:31 AM IST** | `19:01 UTC` | `1 19 * * *` | `fetch-satellites.yml` | `Satellites` | **Daily** |
| **3** | **01:02 AM IST** | `19:32 UTC` | `32 19 * * *` | `update_events.yml` | `xtrobe-ingestion` | **Daily** |
| **4** | **01:33 AM IST** | `20:03 UTC` | `3 20 * * *` | `update_reports.yml` | `xtrobe-ingestion` | **Daily** |
| **5** | **02:04 AM IST** | `20:34 UTC` | `34 20 */15 * *` | `update_reference_data.yml` | `xtrobe-ingestion` | **Every 15 Days** |
| **6** | **02:35 AM IST** | `21:05 UTC` | `5 21 */15 * *` | `update_expeditions.yml` | `xtrobe-ingestion` | **Every 15 Days** |
| **7** | **03:06 AM IST** | `21:36 UTC` | `36 21 */15 * *` | `update_docking_events.yml` | `xtrobe-ingestion` | **Every 15 Days** |
| **8** | **03:37 AM IST** | `22:07 UTC` | `7 22 24 6,12 *` | `yearly_update.yml` | `Calender-astronomical` | **Dec 25 & June 25** |
| **9** | **+31m offset** | `8 */5 * * *` | `8 */5 * * *` | `update_launches.yml` | `xtrobe-ingestion` | **Every 5 Hours** |
| **10**| **XX:45 IST** | `15 * * * *` | `15 * * * *` | `update_articles.yml` | `xtrobe-ingestion` | **Hourly** |
| **11**| **XX:16 IST** | `46 * * * *` | `46 * * * *` | `update_updates.yml` | `xtrobe-ingestion` | **Hourly** |

---

## 2. Complete Event Breakdown (Yearly vs. Daily)

### Yearly Events (`output/yearly_events.json` — 776 Events)
* **Lunar Events (159 events):** Phases, Supermoons, Micromoons, Blue Moons, Perigee, Apogee, Moon-Planet/Star Conjunctions ($\le 3^\circ$).
* **Variable Stars (308 events):** Algol minima, Delta Cephei, Beta Lyrae, Eta Aquilae, Zeta Geminorum + BJD Rømer delay correction.
* **Stellar Occultations (261 events):** Asteroid occultations of stars from IOTA/RASC XML.
* **Planetary Alignments (28 events):** Oppositions (Mars, Jupiter, Saturn, Uranus, Neptune), Elongations (Mercury, Venus), 21 Planet Pairs, Planet-Star Conjunctions.
* **Meteor Showers (10 events):** Major annual showers matched by solar longitude $\lambda$ + Moon glare warnings.
* **Eclipses (4 events):** Solar (with NASA WGS84 path coordinates) & Lunar Eclipses.
* **Seasons (4 events):** March/Sept Equinoxes, June/Dec Solstices.
* **Atmosphere (2 events):** Zodiacal Light observation windows.

### Daily Dynamic Events (`output/daily_events.json` — 4 Streams)
1. **`asteroid`**: Near-Earth Asteroid close flybys ($\le 20.0\text{ LD}$).
2. **`comet`**: Currently active bright comets ($\le \text{mag } 12.0$).
3. **`space_weather`**: NOAA Space Weather Alerts (Geomagnetic Storm Warnings, Aurora Borealis Kp $\ge 4$ alerts, Solar Flare warnings).
4. **`fireball`**: Super-bright atmospheric meteor fireball / bolide impact detections from US sensors.

---

## 3. Frontend Integration Guide

Frontend clients fetch both files and merge them seamlessly:

```javascript
async function loadFullCalendar() {
  const [yearlyRes, dailyRes] = await Promise.all([
    fetch("output/yearly_events.json"),
    fetch("output/daily_events.json")
  ]);

  const yearlyData = await yearlyRes.json();
  const dailyData  = await dailyRes.json();

  const allEvents = [...yearlyData.events, ...dailyData.events];
  allEvents.sort((a, b) => new Date(a.utc) - new Date(b.utc));

  console.log(`Loaded ${allEvents.length} total events (${yearlyData.count} yearly + ${dailyData.count} dynamic daily)`);
}
```
