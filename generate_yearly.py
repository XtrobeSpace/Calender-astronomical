#!/usr/bin/env python3
"""
XTROBE Calendar Engine — Yearly Event Generator v5.2

Computes all ~750+ deterministic, predictable astronomical events for a given year 
with sub-minute ephemeris precision.

Usage:
  python generate_yearly.py --year 2026
  
Output:
  output/yearly_events.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ── Third-party ───────────────────────────────────────────────
import astronomy as ae
import numpy as np
from skyfield import almanac
from skyfield.api import Star
from skyfield.framelib import ecliptic_frame
from skyfield.searchlib import find_maxima, find_minima

# ── Shared Library ────────────────────────────────────────────
from shared import (
    ALGOL, BASE_DIR, CALDWELL_SOUTH, MESSIER_CATALOG, METEOR_SHOWERS_V3,
    OUTPUT_DIR, SKYFIELD_CACHE, _safe_round, barycentric_correction,
    http, init_skyfield, log, moon_illum_at, safe_json_extract
)

# ─────────────────────────────────────────────────────────────
# NASA SOLAR ECLIPSE WGS84 DATABASE
# ─────────────────────────────────────────────────────────────
NASA_ECLIPSES = {
  "20260217":{"date":"2026 Feb 17","type":"Annular Solar Eclipse","magnitude":0.963,"duration":"02m20s","region":"s Argentina & Chile, s Africa, Antarctica\n[Annular: Antarctica]","path_center":[[-71.9533,136.6417],[-73.8617,121.735],[-73.76,113.1983],[-73.22,106.9417],[-72.4717,102.13],[-71.6117,98.3567],[-70.685,95.365],[-69.7167,92.905],[-68.7267,91.0733],[-67.7217,89.555],[-66.7083,88.355],[-65.69,87.425],[-64.6667,86.725],[-63.6417,86.2283],[-62.6133,85.915],[-61.5817,85.7717],[-60.5433,85.7933],[-59.495,85.9783],[-58.4333,86.3333],[-57.3517,86.875],[-56.2433,87.64],[-55.0917,88.6833],[-53.8667,90.1333],[-52.4967,92.3133],[-50.4717,97.56],[-50.115,99.0317]],"path_north":[[-73.93,144.6717],[-75.6383,104.2117],[-74.8117,98.8683],[-73.885,94.7317],[-72.9017,91.4317],[-71.8867,88.8333],[-70.8567,86.7633],[-69.82,85.1133],[-68.7817,83.7967],[-67.7467,82.755],[-66.7133,81.9433],[-65.685,81.325],[-64.6617,80.8767],[-63.6417,80.575],[-62.625,80.4067],[-61.6083,80.3633],[-60.5933,80.4383],[-59.575,80.6283],[-58.5533,80.9333],[-57.525,81.3583],[-56.4833,81.915],[-55.4267,82.62],[-54.345,83.4983],[-53.2267,84.6033],[-52.0533,86.005],[-47.49,96.4033]],"path_south":[[-69.125,128.1567],[-70.3067,120.2767],[-70.2917,112.6667],[-69.7567,107.6233],[-69.0217,103.8467],[-68.18,100.9217],[-67.2733,98.63],[-66.325,96.8367],[-65.345,95.45],[-64.3433,94.41],[-63.3217,93.67],[-62.2833,93.2067],[-61.2233,93.0067],[-60.1383,93.0717],[-59.02,93.4333],[-57.8533,94.1517],[-56.5983,95.385],[-55.1383,97.6733],[-53.4983,102.6933]],"path_latitude":-64.7167,"path_longitude":86.7533,"time_utc":"2026-02-17T12:11:54Z"},
  "20260812":{"date":"2026 Aug 12","type":"Total Solar Eclipse","magnitude":1.039,"duration":"02m18s","region":"n N. America, w Africa, Europe \n[Total: Arctic, Greenland, Iceland, Spain]","path_center":[[75.0783,113.4517],[82.275,112.4867],[85.295,104.215],[87.2783,81.525],[87.8233,33.0],[86.835,-1.6383],[85.4033,-15.1817],[83.9317,-21.1867],[82.495,-24.2717],[81.11,-25.9917],[79.7733,-26.9817],[78.4833,-27.54],[77.2333,-27.825],[76.0183,-27.9283],[74.8367,-27.905],[73.6833,-27.7883],[72.5567,-27.6033],[71.45,-27.3617],[70.365,-27.0783],[69.2983,-26.76],[68.2467,-26.41],[67.21,-26.0317],[66.185,-25.63],[65.1717,-25.205],[64.1683,-24.7567],[63.1717,-24.2867],[62.1833,-23.7933],[61.2,-23.2767],[60.2217,-22.7367],[59.245,-22.17],[58.2717,-21.5733],[57.2967,-20.9467],[56.3217,-20.2867],[55.3433,-19.5883],[54.3617,-18.8467],[53.3717,-18.0567],[52.3717,-17.2117],[51.36,-16.3033],[50.3333,-15.3167],[49.285,-14.2383],[48.2117,-13.0483],[47.1017,-11.715],[45.9433,-10.19],[44.7133,-8.3983],[43.3717,-6.1883],[41.8167,-3.185],[39.4083,2.95],[38.68,5.415]],"path_north":[[75.1733,108.69],[75.9367,108.7583],[82.1633,103.2167],[84.85,90.395],[86.3433,65.8233],[86.545,32.7283],[85.72,8.375],[84.4817,-4.81],[83.1317,-12.0017],[81.775,-16.2167],[80.4417,-18.8417],[79.1417,-20.5383],[77.875,-21.6567],[76.6417,-22.3933],[75.44,-22.865],[74.2667,-23.145],[73.1167,-23.285],[71.9917,-23.3133],[70.885,-23.2583],[69.7983,-23.1317],[68.7267,-22.9483],[67.67,-22.7133],[66.6267,-22.4367],[65.5933,-22.12],[64.5717,-21.7683],[63.5567,-21.3817],[62.5483,-20.9617],[61.5467,-20.5083],[60.5483,-20.0217],[59.5533,-19.5],[58.56,-18.9433],[57.565,-18.3483],[56.5683,-17.7117],[55.5683,-17.0283],[54.5617,-16.295],[53.5467,-15.5033],[52.52,-14.6467],[51.4783,-13.7117],[50.4167,-12.685],[49.33,-11.5467],[48.2083,-10.2667],[47.0383,-8.8017],[45.8017,-7.0767],[44.4567,-4.9483],[42.9083,-2.085],[40.665,3.295],[39.7083,6.34]],"path_south":[[74.9133,117.96],[85.3217,119.4233],[87.7533,108.4317],[89.0667,38.1483],[87.7883,-19.5067],[86.1417,-29.2167],[84.565,-32.2467],[83.0717,-33.4167],[81.65,-33.8383],[80.2917,-33.8967],[78.9867,-33.76],[77.7267,-33.505],[76.5067,-33.1783],[75.3233,-32.805],[74.1683,-32.4],[73.0433,-31.9717],[71.94,-31.5267],[70.86,-31.0683],[69.7967,-30.6],[68.7533,-30.12],[67.7233,-29.6317],[66.7067,-29.1333],[65.7033,-28.625],[64.71,-28.1067],[63.7267,-27.5767],[62.75,-27.0333],[61.78,-26.4767],[60.8167,-25.905],[59.8567,-25.3167],[58.9,-24.7067],[57.945,-24.0767],[56.99,-23.4217],[56.0367,-22.7383],[55.0783,-22.025],[54.1183,-21.275],[53.1517,-20.485],[52.1767,-19.6467],[51.1933,-18.755],[50.195,-17.7983],[49.1817,-16.765],[48.1467,-15.6383],[47.0833,-14.3967],[45.9833,-13.0083],[44.8317,-11.42],[43.6067,-9.5517],[42.2633,-7.2367],[40.6833,-4.04],[37.69,4.54]],"path_latitude":65.225,"path_longitude":-25.2283,"time_utc":"2026-08-12T17:45:54Z"},
  "20270206":{"date":"2027 Feb 06","type":"Annular Solar Eclipse","magnitude":0.928,"duration":"07m51s","region":"S. America, Antarctica, w & s Africa\n[Annular: Chile, Argentina, Atlantic]","path_center":[[-39.085,-131.34],[-41.33,-122.8433],[-42.7767,-116.2433],[-43.5967,-111.7417],[-44.155,-108.0933],[-44.5567,-104.9467],[-44.8483,-102.1417],[-45.0567,-99.5917],[-45.2017,-97.2417],[-45.2917,-95.055],[-45.3383,-93.0067],[-45.345,-91.075],[-45.3183,-89.245],[-45.2617,-87.5083],[-45.1783,-85.8517],[-45.07,-84.2717],[-44.9417,-82.7567],[-44.7917,-81.305],[-44.625,-79.91],[-44.44,-78.5683],[-44.2417,-77.2767],[-44.0283,-76.03],[-43.8,-74.8283],[-43.5617,-73.6667],[-43.31,-72.545],[-43.0483,-71.4583],[-42.7767,-70.4067],[-42.495,-69.3867],[-42.2033,-68.3983],[-41.905,-67.44],[-41.5967,-66.5083],[-41.2817,-65.605],[-40.96,-64.7267],[-40.63,-63.8717],[-40.295,-63.04],[-39.9533,-62.23],[-39.605,-61.4417],[-39.25,-60.6733],[-38.89,-59.925],[-38.5267,-59.195],[-38.1567,-58.4817],[-37.7817,-57.785],[-37.4017,-57.1033],[-37.0167,-56.4383],[-36.6283,-55.7867],[-36.235,-55.15],[-35.8383,-54.5267],[-35.4367,-53.915],[-35.03,-53.315],[-34.6217,-52.7267],[-34.2083,-52.15],[-33.7917,-51.5833],[-33.37,-51.025],[-32.9467,-50.4767],[-32.5183,-49.9367],[-32.0867,-49.405],[-31.6517,-48.8817],[-31.2133,-48.365],[-30.7717,-47.855],[-30.3267,-47.35],[-29.8783,-46.8517],[-29.4267,-46.36],[-28.9717,-45.8717],[-28.5133,-45.3867],[-28.05,-44.9067],[-27.585,-44.43],[-27.1167,-43.9567],[-26.6433,-43.485],[-26.1667,-43.015],[-25.6883,-42.5467],[-25.205,-42.08],[-24.7167,-41.6133],[-24.2267,-41.1467],[-23.7317,-40.68],[-23.2333,-40.2133],[-22.7317,-39.7433],[-22.2233,-39.2717],[-21.7133,-38.7967],[-21.1983,-38.32],[-20.6783,-37.8383],[-20.1533,-37.3517],[-19.625,-36.8617],[-19.09,-36.3633],[-18.55,-35.86],[-18.0067,-35.3483],[-17.455,-34.8267],[-16.9,-34.2983],[-16.3367,-33.7567],[-15.7683,-33.205],[-15.1917,-32.64],[-14.61,-32.0617],[-14.0183,-31.4667],[-13.42,-30.855],[-12.8133,-30.225],[-12.1967,-29.5733],[-11.5717,-28.8983],[-10.935,-28.1983],[-10.2867,-27.4683],[-9.6267,-26.7083],[-8.9533,-25.9117],[-8.2633,-25.0733],[-7.5583,-24.19],[-6.8333,-23.2533],[-6.0883,-22.2567],[-5.32,-21.1883],[-4.5233,-20.0367],[-3.6933,-18.7833],[-2.825,-17.405],[-1.905,-15.8683],[-0.9217,-14.1217],[0.15,-12.0817],[1.3567,-9.5833],[2.81,-6.2333],[5.1183,0.1233],[6.195,3.7117]],"path_north":[[-37.6467,-130.35],[-40.8533,-117.8333],[-41.8317,-112.9717],[-42.4783,-109.1867],[-42.9417,-105.98],[-43.2833,-103.1517],[-43.535,-100.5983],[-43.715,-98.255],[-43.84,-96.0817],[-43.9183,-94.05],[-43.955,-92.14],[-43.9567,-90.3317],[-43.9283,-88.6167],[-43.8717,-86.9833],[-43.79,-85.425],[-43.6867,-83.9317],[-43.5633,-82.5017],[-43.42,-81.1267],[-43.2617,-79.805],[-43.085,-78.5317],[-42.895,-77.3033],[-42.6917,-76.1183],[-42.475,-74.9717],[-42.2467,-73.865],[-42.0067,-72.7917],[-41.7567,-71.7533],[-41.4967,-70.7467],[-41.2267,-69.77],[-40.9483,-68.8217],[-40.6617,-67.9017],[-40.3667,-67.0067],[-40.0633,-66.1367],[-39.755,-65.29],[-39.4383,-64.465],[-39.115,-63.6633],[-38.785,-62.8817],[-38.45,-62.1183],[-38.1083,-61.375],[-37.7617,-60.65],[-37.4083,-59.9417],[-37.0517,-59.2483],[-36.6883,-58.5717],[-36.3217,-57.91],[-35.9483,-57.2633],[-35.5717,-56.63],[-35.19,-56.0083],[-34.805,-55.4],[-34.415,-54.8033],[-34.02,-54.2183],[-33.6217,-53.6433],[-33.22,-53.08],[-32.8133,-52.525],[-32.4033,-51.9783],[-31.9883,-51.4417],[-31.5717,-50.9133],[-31.15,-50.3917],[-30.725,-49.8783],[-30.295,-49.3717],[-29.8633,-48.87],[-29.4267,-48.375],[-28.9867,-47.885],[-28.5433,-47.4017],[-28.0967,-46.9217],[-27.6467,-46.445],[-27.1917,-45.9733],[-26.735,-45.5033],[-26.2733,-45.0383],[-25.8083,-44.5733],[-25.3383,-44.1117],[-24.8667,-43.6517],[-24.39,-43.1917],[-23.9083,-42.7333],[-23.4233,-42.275],[-22.935,-41.815],[-22.4433,-41.355],[-21.9467,-40.8933],[-21.445,-40.43],[-20.9383,-39.9633],[-20.4283,-39.4933],[-19.9133,-39.0217],[-19.3933,-38.5433],[-18.87,-38.0617],[-18.34,-37.5733],[-17.8033,-37.0783],[-17.2633,-36.5767],[-16.7167,-36.0667],[-16.1633,-35.5467],[-15.605,-35.0183],[-15.04,-34.4783],[-14.4667,-33.9267],[-13.8883,-33.3617],[-13.3,-32.78],[-12.705,-32.1833],[-12.1,-31.57],[-11.4867,-30.935],[-10.865,-30.2783],[-10.2317,-29.5983],[-9.5867,-28.8917],[-8.9283,-28.155],[-8.2583,-27.3833],[-7.5733,-26.575],[-6.8717,-25.725],[-6.1517,-24.825],[-5.4133,-23.87],[-4.6517,-22.85],[-3.8617,-21.7567],[-3.0433,-20.57],[-2.1883,-19.2767],[-1.2883,-17.845],[-0.3333,-16.24],[0.6967,-14.3967],[1.8333,-12.2133],[3.1367,-9.4683],[4.7933,-5.5383],[7.6583,2.9817]],"path_south":[[-40.5317,-132.3833],[-43.4667,-120.7133],[-44.6117,-114.9383],[-45.3167,-110.6517],[-45.8017,-107.085],[-46.1483,-103.9667],[-46.395,-101.165],[-46.565,-98.6033],[-46.675,-96.235],[-46.7317,-94.0267],[-46.7467,-91.9517],[-46.7233,-89.9933],[-46.6667,-88.1367],[-46.5833,-86.3733],[-46.4717,-84.6917],[-46.3383,-83.085],[-46.1817,-81.5467],[-46.0067,-80.07],[-45.815,-78.6533],[-45.6067,-77.2917],[-45.3833,-75.98],[-45.145,-74.715],[-44.895,-73.495],[-44.6317,-72.3183],[-44.3583,-71.18],[-44.0733,-70.0783],[-43.78,-69.0133],[-43.4767,-67.9817],[-43.1633,-66.9817],[-42.8433,-66.0133],[-42.515,-65.0733],[-42.18,-64.16],[-41.8383,-63.2733],[-41.4883,-62.41],[-41.1333,-61.5717],[-40.7717,-60.7567],[-40.405,-59.9617],[-40.0317,-59.1883],[-39.655,-58.435],[-39.2717,-57.7],[-38.885,-56.9817],[-38.4917,-56.2817],[-38.095,-55.5967],[-37.695,-54.9283],[-37.29,-54.275],[-36.88,-53.635],[-36.4667,-53.0083],[-36.05,-52.3933],[-35.63,-51.7917],[-35.205,-51.2017],[-34.7783,-50.6217],[-34.3467,-50.0517],[-33.9117,-49.4917],[-33.475,-48.9417],[-33.0333,-48.4],[-32.5883,-47.865],[-32.14,-47.3383],[-31.69,-46.8183],[-31.235,-46.305],[-30.7783,-45.7983],[-30.3183,-45.2967],[-29.8533,-44.8],[-29.3867,-44.3083],[-28.9167,-43.82],[-28.4433,-43.335],[-27.9667,-42.8533],[-27.4883,-42.3733],[-27.005,-41.8967],[-26.5183,-41.42],[-26.0283,-40.9467],[-25.535,-40.4717],[-25.0383,-39.9983],[-24.5367,-39.5233],[-24.0333,-39.0467],[-23.525,-38.57],[-23.0117,-38.09],[-22.4967,-37.6083],[-21.9767,-37.1217],[-21.4517,-36.6317],[-20.9217,-36.1367],[-20.3883,-35.635],[-19.85,-35.1283],[-19.3067,-34.615],[-18.7583,-34.0917],[-18.2033,-33.56],[-17.6433,-33.02],[-17.0783,-32.4667],[-16.505,-31.9033],[-15.9267,-31.325],[-15.34,-30.7317],[-14.7467,-30.1217],[-14.145,-29.4933],[-13.535,-28.845],[-12.9167,-28.175],[-12.2867,-27.48],[-11.6483,-26.7583],[-10.9967,-26.005],[-10.3317,-25.2167],[-9.655,-24.39],[-8.9617,-23.52],[-8.2517,-22.6],[-7.5217,-21.6217],[-6.77,-20.5767],[-5.9917,-19.4533],[-5.185,-18.235],[-4.3417,-16.9],[-3.455,-15.42],[-2.5117,-13.7517],[-1.4917,-11.8233],[-0.36,-9.5067],[0.9617,-6.5233],[2.745,-1.88],[4.7183,4.4583]],"path_latitude":-31.3033,"path_longitude":-48.4683,"time_utc":"2027-02-06T15:59:36Z"},
  "20270802":{"date":"2027 Aug 02","type":"Total Solar Eclipse","magnitude":1.079,"duration":"06m23s","region":"Africa, Europe, Mid East, w & s Asia\n[Total:Morocco, Spain, Algeria, Libya, Egypt, Saudi Arabia, Yemen, Somalia]","path_center":[[27.9617,-44.4767],[30.4883,-36.2333],[32.1183,-30.09],[33.085,-25.9033],[33.77,-22.5183],[34.2883,-19.6017],[34.6883,-17.0033],[35.0017,-14.64],[35.2433,-12.46],[35.4283,-10.43],[35.565,-8.5217],[35.66,-6.7217],[35.7183,-5.01],[35.7433,-3.3817],[35.7383,-1.8233],[35.7083,-0.33],[35.6517,1.105],[35.575,2.4867],[35.4767,3.8183],[35.36,5.105],[35.225,6.35],[35.0733,7.555],[34.905,8.725],[34.7233,9.8583],[34.5267,10.96],[34.3183,12.0317],[34.0967,13.075],[33.8633,14.09],[33.6183,15.08],[33.3617,16.045],[33.0967,16.9867],[32.82,17.9067],[32.535,18.805],[32.24,19.6833],[31.9367,20.5433],[31.6233,21.385],[31.3033,22.2083],[30.975,23.0167],[30.6383,23.8083],[30.295,24.585],[29.9433,25.3467],[29.585,26.0967],[29.22,26.8317],[28.8483,27.5567],[28.4683,28.2683],[28.0833,28.97],[27.6917,29.66],[27.2933,30.3417],[26.8883,31.0133],[26.4783,31.6767],[26.0617,32.3333],[25.6383,32.98],[25.21,33.6217],[24.775,34.2567],[24.335,34.885],[23.8883,35.5083],[23.4367,36.1267],[22.9783,36.7417],[22.5133,37.3517],[22.0433,37.9583],[21.5667,38.5633],[21.085,39.1667],[20.5967,39.7683],[20.1017,40.37],[19.6017,40.97],[19.095,41.5717],[18.5817,42.1733],[18.0617,42.7767],[17.5333,43.3833],[17.0,43.9933],[16.46,44.6067],[15.9117,45.225],[15.3567,45.85],[14.7933,46.48],[14.2217,47.1183],[13.6433,47.765],[13.055,48.4217],[12.4583,49.09],[11.8517,49.7683],[11.235,50.4617],[10.61,51.1717],[9.9717,51.8967],[9.3233,52.6417],[8.6617,53.4067],[7.9883,54.1967],[7.3,55.0133],[6.595,55.8583],[5.8767,56.7367],[5.1383,57.6517],[4.3817,58.61],[3.6017,59.615],[2.7983,60.6767],[1.9683,61.8033],[1.1083,63.005],[0.21,64.295],[-0.7283,65.695],[-1.7183,67.2267],[-2.7717,68.9317],[-3.9067,70.86],[-5.155,73.1083],[-6.575,75.8517],[-8.3067,79.52],[-11.1017,86.42],[-12.4833,90.4417]],"path_north":[[28.8,-44.9433],[30.7217,-38.8933],[32.77,-31.5833],[33.8617,-27.0667],[34.62,-23.4917],[35.1883,-20.44],[35.6283,-17.735],[35.9717,-15.2817],[36.24,-13.0233],[36.445,-10.9217],[36.6,-8.95],[36.7083,-7.0867],[36.78,-5.32],[36.815,-3.6367],[36.82,-2.0283],[36.7967,-0.4867],[36.7467,0.9933],[36.6733,2.42],[36.58,3.795],[36.465,5.1233],[36.3317,6.4067],[36.1817,7.65],[36.0133,8.8567],[35.8317,10.025],[35.6333,11.1617],[35.4233,12.265],[35.1983,13.34],[34.9617,14.385],[34.7133,15.405],[34.455,16.3983],[34.1833,17.3667],[33.9033,18.3117],[33.6133,19.2367],[33.3117,20.1383],[33.0033,21.0217],[32.685,21.885],[32.3583,22.73],[32.0233,23.5583],[31.68,24.37],[31.33,25.165],[30.9717,25.9467],[30.605,26.7117],[30.2333,27.465],[29.8533,28.205],[29.4667,28.9333],[29.0733,29.6483],[28.675,30.3533],[28.2683,31.0483],[27.8567,31.7333],[27.4367,32.4083],[27.0133,33.0767],[26.5817,33.735],[26.145,34.3867],[25.7033,35.0317],[25.2533,35.67],[24.8,36.3033],[24.3383,36.93],[23.8733,37.5533],[23.4,38.1717],[22.9233,38.7883],[22.4383,39.4],[21.9483,40.01],[21.4517,40.6183],[20.95,41.2267],[20.4417,41.8333],[19.9267,42.44],[19.4067,43.0483],[18.8783,43.6583],[18.345,44.27],[17.8033,44.885],[17.255,45.5033],[16.7,46.1267],[16.1367,46.7567],[15.5667,47.3917],[14.9883,48.035],[14.4017,48.685],[13.8067,49.3467],[13.2033,50.0183],[12.59,50.7033],[11.9667,51.4017],[11.3333,52.115],[10.6883,52.845],[10.0333,53.595],[9.3633,54.365],[8.6817,55.16],[7.9867,55.9817],[7.275,56.8333],[6.5483,57.7183],[5.8017,58.6417],[5.0367,59.6067],[4.2483,60.6217],[3.435,61.695],[2.595,62.8333],[1.7233,64.0483],[0.8133,65.3583],[-0.14,66.7783],[-1.145,68.34],[-2.2183,70.0817],[-3.38,72.0633],[-4.6633,74.3917],[-6.1383,77.2733],[-7.9917,81.2667],[-11.6333,90.8383]],"path_south":[[27.1217,-44.0233],[30.035,-34.225],[31.4083,-28.79],[32.2733,-24.8717],[32.8967,-21.6483],[33.37,-18.8517],[33.7367,-16.35],[34.0217,-14.07],[34.2417,-11.9633],[34.4067,-9.9983],[34.5283,-8.1517],[34.6083,-6.4067],[34.655,-4.75],[34.67,-3.17],[34.6583,-1.66],[34.62,-0.2117],[34.56,1.18],[34.4783,2.5183],[34.3767,3.8117],[34.2567,5.06],[34.12,6.2667],[33.9683,7.4367],[33.8017,8.5717],[33.62,9.6733],[33.425,10.7433],[33.2183,11.7833],[32.9983,12.7967],[32.7683,13.7833],[32.5267,14.7467],[32.275,15.685],[32.0117,16.6],[31.74,17.495],[31.46,18.37],[31.17,19.2267],[30.8717,20.0633],[30.5667,20.8833],[30.2517,21.6883],[29.93,22.475],[29.6,23.2483],[29.2617,24.0067],[28.9183,24.7517],[28.5667,25.485],[28.2083,26.205],[27.8433,26.9133],[27.4717,27.6117],[27.0933,28.2983],[26.71,28.975],[26.3183,29.6433],[25.9217,30.3033],[25.52,30.955],[25.11,31.6],[24.695,32.2367],[24.275,32.8667],[23.8467,33.4917],[23.415,34.11],[22.975,34.725],[22.5317,35.335],[22.08,35.94],[21.6233,36.5433],[21.1617,37.1433],[20.6933,37.74],[20.2183,38.3367],[19.7367,38.9317],[19.25,39.525],[18.7567,40.12],[18.2567,40.715],[17.7517,41.3117],[17.2383,41.91],[16.7183,42.5117],[16.1917,43.115],[15.6583,43.725],[15.1183,44.3383],[14.57,44.9567],[14.0133,45.5833],[13.45,46.2167],[12.8767,46.86],[12.295,47.5117],[11.705,48.175],[11.1067,48.85],[10.4967,49.54],[9.8767,50.2433],[9.2467,50.965],[8.605,51.705],[7.95,52.4667],[7.2833,53.2517],[6.6017,54.0617],[5.9067,54.9],[5.1933,55.7733],[4.4633,56.6817],[3.7133,57.6317],[2.9433,58.63],[2.15,59.6817],[1.3283,60.7967],[0.4783,61.985],[-0.4083,63.26],[-1.3333,64.64],[-2.3083,66.1483],[-3.345,67.8183],[-4.4567,69.7033],[-5.675,71.8833],[-7.0467,74.5133],[-8.6867,77.9383],[-11.04,83.5317],[-13.3317,90.05]],"path_latitude":25.505,"path_longitude":33.1833,"time_utc":"2027-08-02T10:06:38Z"}
}

class SeparationFinder:
    def __init__(self, earth, body_a, body_b, step_days=1.0):
        self.earth = earth
        self.body_a = body_a
        self.body_b = body_b
        self.step_days = step_days

    def __call__(self, t):
        p1 = self.earth.at(t).observe(self.body_a).apparent()
        p2 = self.earth.at(t).observe(self.body_b).apparent()
        return p1.separation_from(p2).degrees

VARIABLE_STARS_META = {
    "DCEP": {
        "ident": "del Cep",
        "name": "Delta Cephei Maximum",
        "subtype": "cepheid_variable",
        "ra_h": 22.4862,
        "dec_d": 58.4153,
        "default_t0": 2436075.415,
        "default_period": 5.366341,
        "default_mag_max": 3.48,
        "default_mag_min": 4.37,
        "cache_name": "aavso_deltacep_cache.xml",
        "desc_template": "Delta Cephei peak maximum brightness. Prototype Cepheid variable magnitude {mag_max} to {mag_min} over {period_days} days."
    },
    "BLYR": {
        "ident": "bet Lyr",
        "name": "Beta Lyrae Minimum",
        "subtype": "eclipsing_binary",
        "ra_h": 18.8347,
        "dec_d": 33.3631,
        "default_t0": 2436793.48,
        "default_period": 12.93095,
        "quadratic_coeff": 0.00000386,
        "default_mag_max": 3.25,
        "default_mag_min": 4.36,
        "cache_name": "aavso_betalyr_cache.xml",
        "desc_template": "Beta Lyrae primary eclipse minimum. Interacting binary magnitude {mag_max} to {mag_min} over {period_days} days."
    },
    "EAQL": {
        "ident": "eta Aql",
        "name": "Eta Aquilae Maximum",
        "subtype": "cepheid_variable",
        "ra_h": 19.8746,
        "dec_d": 1.0056,
        "default_t0": 2442752.628,
        "default_period": 7.176641,
        "default_mag_max": 3.48,
        "default_mag_min": 4.39,
        "cache_name": "aavso_etaaquilae_cache.xml",
        "desc_template": "Eta Aquilae peak maximum brightness. Cepheid variable magnitude {mag_max} to {mag_min} over {period_days} days."
    },
    "ZGEM": {
        "ident": "zet Gem",
        "name": "Zeta Geminorum Maximum",
        "subtype": "cepheid_variable",
        "ra_h": 7.0685,
        "dec_d": 20.5703,
        "default_t0": 2442749.626,
        "default_period": 10.15073,
        "default_mag_max": 3.62,
        "default_mag_min": 4.18,
        "cache_name": "aavso_zetagem_cache.xml",
        "desc_template": "Zeta Geminorum peak maximum brightness. Cepheid variable magnitude {mag_max} to {mag_min} over {period_days} days."
    }
}

# ─────────────────────────────────────────────────────────────
# YEARLY EVENT GENERATORS
# ─────────────────────────────────────────────────────────────

def _classify_full_moon(ts, eph, t_full) -> str | None:
    earth, moon = eph["earth"], eph["moon"]
    def dist_fn(t): return earth.at(t).observe(moon).apparent().distance().km
    dist_fn.step_days = 1.0

    t_lo = ts.tt_jd(t_full.tt - 16)
    t_hi = ts.tt_jd(t_full.tt + 16)

    peri_t, peri_d = find_minima(t_lo, t_hi, dist_fn, epsilon=1.0/24)
    apo_t,  apo_d  = find_maxima(t_lo, t_hi, dist_fn, epsilon=1.0/24)

    if len(peri_t) == 0 or len(apo_t) == 0: return None
    perigee_km = min(peri_d)
    apogee_km  = max(apo_d)
    rng = apogee_km - perigee_km
    if rng <= 0: return None

    full_km = dist_fn(t_full)
    frac = (full_km - perigee_km) / rng
    if frac <= 0.10: return "supermoon"
    if frac >= 0.90: return "micromoon"
    return None

def gen_lunar_phases(ts, eph, year: int) -> list[dict]:
    t0 = ts.utc(year - 1, 12, 31)
    t1 = ts.utc(year + 1,  1,  2)
    times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    names = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]
    earth, moon = eph["earth"], eph["moon"]
    full_by_month = {}
    events = []

    for t, ph in zip(times, phases):
        utc_dt = t.utc_datetime()
        if utc_dt.year != year:
            continue

        name = names[ph]
        subtype, dist_km, blue_moon_note = None, None, None

        astrometric = earth.at(t).observe(moon).apparent()
        ra_, dec_, d_ = astrometric.radec()
        ra_h  = round(ra_.hours, 3)
        dec_d = round(dec_.degrees, 2)

        if ph == 2:  # Full Moon
            dist_km = round(d_.km, 0)
            sub_label = _classify_full_moon(ts, eph, t)
            if sub_label == "supermoon":
                name, subtype = "Supermoon (Full Moon)", "supermoon"
            elif sub_label == "micromoon":
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
            "ra_h":        ra_h,
            "dec_d":       dec_d,
            "distance_km": dist_km,
            "magnitude":   -12.6 if ph == 2 else None,
            "global_event":True,
            "precision":   "exact",
            "method":      "skyfield_almanac_de440s",
            "description": f"{name} at UTC {utc_dt.strftime('%H:%M')}.",
        })
    return events

def gen_eclipses(ts, eph, year: int) -> list[dict]:
    events = []
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    t = ae.Time.Make(year, 1, 1, 0, 0, 0)

    for _ in range(6):
        le = ae.SearchLunarEclipse(t)
        peak_dt = le.peak.Utc().replace(tzinfo=timezone.utc)
        if peak_dt.year > year: break
        if peak_dt.year == year and le.kind != ae.EclipseKind.Invalid:
            t_sf = ts.from_datetime(peak_dt)
            ra_, dec_, _ = earth.at(t_sf).observe(moon).apparent().radec()
            kind_name = {
                ae.EclipseKind.Total: "Total Lunar Eclipse",
                ae.EclipseKind.Partial: "Partial Lunar Eclipse",
                ae.EclipseKind.Penumbral: "Penumbral Lunar Eclipse",
            }.get(le.kind, "Lunar Eclipse")

            events.append({
                "id":           f"LEC_{peak_dt.strftime('%Y%m%d')}",
                "category":     "eclipse",
                "type":         kind_name,
                "subtype":      "lunar_eclipse",
                "utc":          peak_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":         round(ra_.hours, 3),
                "dec_d":        round(dec_.degrees, 2),
                "obscuration":  _safe_round(le.obscuration, 4),
                "global_event": False,
                "precision":    "exact",
                "method":       "astronomy_engine_shadow_search",
                "description":  f"{kind_name} peak instant.",
            })
        t = le.peak.AddDays(10.0)

    t = ae.Time.Make(year, 1, 1, 0, 0, 0)
    for _ in range(6):
        se = ae.SearchGlobalSolarEclipse(t)
        peak_dt = se.peak.Utc().replace(tzinfo=timezone.utc)
        if peak_dt.year > year: break
        if peak_dt.year == year and se.kind != ae.EclipseKind.Invalid:
            date_key = peak_dt.strftime('%Y%m%d')
            t_sf = ts.from_datetime(peak_dt)
            ra_, dec_, _ = earth.at(t_sf).observe(sun).apparent().radec()
            nasa_data = NASA_ECLIPSES.get(date_key)
            kind_name = {
                ae.EclipseKind.Total: "Total Solar Eclipse",
                ae.EclipseKind.Annular: "Annular Solar Eclipse",
                ae.EclipseKind.Partial: "Partial Solar Eclipse",
            }.get(se.kind, "Solar Eclipse")

            if nasa_data:
                events.append({
                    "id":             f"SEC_{date_key}",
                    "category":       "eclipse",
                    "type":           kind_name,
                    "subtype":        "solar_eclipse",
                    "utc":            nasa_data.get("time_utc", peak_dt.strftime("%Y-%m-%dT%H:%M:%SZ")),
                    "ra_h":           round(ra_.hours, 3),
                    "dec_d":          round(dec_.degrees, 2),
                    "obscuration":    nasa_data.get("magnitude", _safe_round(se.obscuration, 4)),
                    "path_latitude":  nasa_data.get("path_latitude"),
                    "path_longitude": nasa_data.get("path_longitude"),
                    "path_center":    nasa_data.get("path_center", []),
                    "path_north":     nasa_data.get("path_north", []),
                    "path_south":     nasa_data.get("path_south", []),
                    "global_event":   False,
                    "precision":      "exact",
                    "method":         "nasa_authoritative_bulletin",
                    "description":    f"{kind_name} — NASA WGS84 centerline and path boundaries.",
                })
        t = se.peak.AddDays(10.0)

    return events

def gen_seasons(ts, eph, year: int) -> list[dict]:
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31, 23, 59)
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
            "id":           f"SEA_{idx}_{year}",
            "category":     "season",
            "type":         names[idx],
            "subtype":      "equinox" if idx in (0, 2) else "solstice",
            "utc":          utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":         None,
            "dec_d":        None,
            "global_event": True,
            "precision":    "exact",
            "method":       "skyfield_almanac_de440s",
            "description":  f"{names[idx]} — exact moment Sun crosses ecliptic milestone.",
        })
    return events

def gen_planetary_events(ts, eph, year: int) -> list[dict]:
    events = []
    earth = eph["earth"]
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31, 23, 59)
    EPS = 2.0 / 1440.0

    # Oppositions
    outer = [
        ("MARS", "mars barycenter", -2.9),
        ("JUPITER", "jupiter barycenter", -2.9),
        ("SATURN", "saturn barycenter", 0.5),
        ("URANUS", "uranus barycenter", 5.7),
        ("NEPTUNE", "neptune barycenter", 7.8),
    ]
    for label, bkey, ref_mag in outer:
        def opp_fn(t):
            plon = earth.at(t).observe(eph[bkey]).apparent().frame_latlon(ecliptic_frame)[1].degrees % 360.0
            slon = earth.at(t).observe(eph["sun"]).apparent().frame_latlon(ecliptic_frame)[1].degrees % 360.0
            return np.abs((plon - slon) % 360.0 - 180.0)
        opp_fn.step_days = 5.0

        times_opp, diffs_opp = find_minima(t0, t1, opp_fn, epsilon=EPS)
        for t2, dv in zip(times_opp, diffs_opp):
            if dv > 3.0: continue
            utc_dt = t2.utc_datetime()
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
                "precision":   "exact",
                "method":      "skyfield_root_finding",
                "description": f"{label.title()} at opposition — visible all night.",
            })

    # Elongations
    inner = [("MERCURY", "mercury", -0.4), ("VENUS", "venus", -4.0)]
    for label, bkey, ref_mag in inner:
        def elong_fn(t):
            plon = earth.at(t).observe(eph[bkey]).apparent().frame_latlon(ecliptic_frame)[1].degrees % 360.0
            slon = earth.at(t).observe(eph["sun"]).apparent().frame_latlon(ecliptic_frame)[1].degrees % 360.0
            return (plon - slon + 180.0) % 360.0 - 180.0
        elong_fn.step_days = 5.0

        times_e, vals_e = find_maxima(t0, t1, elong_fn, epsilon=EPS)
        for t2, ev_ in zip(times_e, vals_e):
            if ev_ < 10: continue
            utc_dt = t2.utc_datetime()
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
                "elongation_deg": round(float(ev_), 1),
                "magnitude":      ref_mag,
                "global_event":   True,
                "precision":      "exact",
                "method":         "skyfield_root_finding",
                "description":    f"{label.title()} at max eastern elongation ({round(float(ev_),1)}°).",
            })

    # Planetary Conjunctions (21 pairs)
    all_planets = [
        ("MERCURY","mercury"), ("VENUS","venus"),
        ("MARS","mars barycenter"), ("JUPITER","jupiter barycenter"),
        ("SATURN","saturn barycenter"), ("URANUS","uranus barycenter"),
        ("NEPTUNE","neptune barycenter")
    ]
    pairs = [(all_planets[i], all_planets[j]) for i in range(len(all_planets)) for j in range(i+1, len(all_planets))]

    for (l1, b1), (l2, b2) in pairs:
        finder = SeparationFinder(earth, eph[b1], eph[b2], step_days=5.0)
        times_c, seps_c = find_minima(t0, t1, finder, epsilon=EPS)
        for t2, sep_v in zip(times_c, seps_c):
            if sep_v >= 2.0: continue
            utc_dt = t2.utc_datetime()
            ra_, dec_, _ = earth.at(t2).observe(eph[b1]).apparent().radec()
            events.append({
                "id":             f"CONJ_{l1}_{l2}_{utc_dt.strftime('%Y%m%d')}",
                "category":       "planet",
                "type":           f"{l1.title()}-{l2.title()} Conjunction",
                "subtype":        "conjunction",
                "bodies":         [l1, l2],
                "utc":            utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":           round(ra_.hours, 3),
                "dec_d":          round(dec_.degrees, 2),
                "separation_deg": round(float(sep_v), 2),
                "global_event":   True,
                "precision":      "exact",
                "method":         "skyfield_root_finding",
                "description":    f"{l1.title()} and {l2.title()} pass within {round(float(sep_v),2)}°.",
            })

    return events

def gen_meteor_showers(ts, eph, year: int) -> list[dict]:
    events = []
    search_start = ae.Time.Make(year, 1, 1, 0, 0, 0)
    for sid, name, ra_h, dec_d, lam, unc_deg, zhr, dur, spd, parent in METEOR_SHOWERS_V3:
        try:
            t_peak_ae = ae.SearchSunLongitude(lam, search_start, 365.0)
            peak_utc  = t_peak_ae.Utc().replace(tzinfo=timezone.utc)
            if peak_utc.year != year: continue
            moon_pct  = moon_illum_at(ts, eph, peak_utc)

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
                "lambda_peak_deg":   lam,
                "zhr":               zhr,
                "speed_kms":         spd,
                "parent_body":       parent,
                "moon_illumination": moon_pct,
                "moon_interference": moon_pct > 0.50,
                "global_event":      False,
                "precision":         "predicted",
                "method":            "iau_mdc_solar_longitude",
                "description":       f"{name} peak at solar longitude {lam}°. Moon: {int(moon_pct*100)}%.",
            })
        except Exception:
            continue
    return events

def gen_algol_minima(ts, eph, year: int) -> list[dict]:
    events = []
    JD_UNIX_EPOCH = 2440587.5
    J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    ref_jd = ALGOL["t0_jd"]
    period = ALGOL["period_days"]
    y_start_jd = (datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0) + JD_UNIX_EPOCH
    y_end_jd   = (datetime(year+1, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0) + JD_UNIX_EPOCH

    n0 = math.ceil((y_start_jd - ref_jd) / period)
    n1 = math.floor((y_end_jd - ref_jd) / period)

    for n in range(n0, n1 + 1):
        bjd_jd = ref_jd + n * period
        bjd_dt = (J2000_UTC + timedelta(days=bjd_jd - 2451545.0)).replace(microsecond=0)
        bjd_delta_days = barycentric_correction(ts, eph, bjd_dt, ALGOL["ra_h"], ALGOL["dec_d"])
        utc_dt = (bjd_dt - timedelta(days=bjd_delta_days)).replace(microsecond=0)

        if utc_dt.year != year: continue

        events.append({
            "id":                 f"ALG_{utc_dt.strftime('%Y%m%d_%H%M')}",
            "category":           "variable",
            "type":               "Algol Minimum (Beta Persei)",
            "subtype":            "eclipsing_binary",
            "utc":                utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bjd_roemer_utc":     bjd_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bjd_correction_min": round(bjd_delta_days * 1440.0, 1),
            "ra_h":               ALGOL["ra_h"],
            "dec_d":              ALGOL["dec_d"],
            "mag_at_min":         ALGOL["mag_min"],
            "mag_at_max":         ALGOL["mag_max"],
            "global_event":       False,
            "precision":          "exact",
            "method":             "linear_ephemeris_bjd_corrected",
            "description":        "Algol mid-minimum — deepest eclipse point.",
        })
    return events

def gen_expanded_variable_stars(ts, eph, year: int) -> list[dict]:
    events = []
    JD_UNIX_EPOCH = 2440587.5
    J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    y_start_jd = (datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0) + JD_UNIX_EPOCH
    y_end_jd   = (datetime(year+1, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0) + JD_UNIX_EPOCH

    for key, meta in VARIABLE_STARS_META.items():
        epoch, period = meta["default_t0"], meta["default_period"]
        mag_max, mag_min = meta["default_mag_max"], meta["default_mag_min"]
        quad_c = meta.get("quadratic_coeff", 0.0)

        n0 = math.ceil((y_start_jd - epoch) / period)
        n1 = math.floor((y_end_jd - epoch) / period)

        for n in range(n0, n1 + 1):
            bjd_jd = epoch + n * period + quad_c * n * n
            bjd_dt = (J2000_UTC + timedelta(days=bjd_jd - 2451545.0)).replace(microsecond=0)
            bjd_delta_days = barycentric_correction(ts, eph, bjd_dt, meta["ra_h"], meta["dec_d"])
            utc_dt = (bjd_dt - timedelta(days=bjd_delta_days)).replace(microsecond=0)

            if utc_dt.year != year: continue

            events.append({
                "id":                 f"VAR_{key}_{utc_dt.strftime('%Y%m%d_%H%M')}",
                "category":           "variable",
                "type":               meta["name"],
                "subtype":            meta["subtype"],
                "utc":                utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bjd_roemer_utc":     bjd_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bjd_correction_min": round(bjd_delta_days * 1440.0, 1),
                "ra_h":               meta["ra_h"],
                "dec_d":              meta["dec_d"],
                "mag_max":            mag_max,
                "mag_min":            mag_min,
                "period_days":        round(period, 5),
                "global_event":       False,
                "precision":          "exact",
                "method":             "quadratic_ephemeris" if quad_c else "linear_ephemeris",
                "description":        meta["desc_template"].format(mag_max=mag_max, mag_min=mag_min, period_days=round(period,2)),
            })
    return events

def gen_zodiacal_light(ts, eph, year: int) -> list[dict]:
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31)
    times, idxs = almanac.find_discrete(t0, t1, almanac.seasons(eph))
    equinox = {}
    for t, idx in zip(times, idxs):
        if idx == 0: equinox["vernal"]   = t.utc_datetime()
        if idx == 2: equinox["autumnal"] = t.utc_datetime()

    events = []
    if "vernal" in equinox:
        ve = equinox["vernal"]
        events.append({
            "id": f"ZOD_NH_SPRING_{year}", "category": "atmosphere",
            "type": "Zodiacal Light — Spring Evening Window (NH)", "subtype": "zodiacal_light",
            "utc": (ve - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end": (ve + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h": None, "dec_d": None, "global_event": False, "hemisphere": "north",
            "precision": "window", "description": "Zodiacal light in west after twilight."
        })
    if "autumnal" in equinox:
        aeq = equinox["autumnal"]
        events.append({
            "id": f"ZOD_NH_AUTUMN_{year}", "category": "atmosphere",
            "type": "Zodiacal Light — Autumn Morning Window (NH)", "subtype": "zodiacal_light",
            "utc": (aeq - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_end": (aeq + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h": None, "dec_d": None, "global_event": False, "hemisphere": "north",
            "precision": "window", "description": "Zodiacal light in east before dawn."
        })
    return events

def gen_lunar_conjunctions(ts, eph, year: int) -> list[dict]:
    events = []
    earth, moon = eph["earth"], eph["moon"]
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31, 23, 59)
    EPS = 2.0 / 1440.0

    targets = [
        ("Venus", eph["venus"]), ("Mars", eph["mars barycenter"]),
        ("Jupiter", eph["jupiter barycenter"]), ("Saturn", eph["saturn barycenter"]),
        ("Pleiades (M45)", Star(ra_hours=3.7914, dec_degrees=24.105)),
        ("Spica", Star(ra_hours=13.4199, dec_degrees=-11.1614)),
        ("Antares", Star(ra_hours=16.4901, dec_degrees=-26.4319)),
        ("Aldebaran", Star(ra_hours=4.5987, dec_degrees=16.5092)),
        ("Regulus", Star(ra_hours=10.1395, dec_degrees=11.9672)),
        ("Pollux", Star(ra_hours=7.7553, dec_degrees=28.0261)),
    ]

    for label, target in targets:
        finder = SeparationFinder(earth, moon, target, step_days=0.5)
        times, seps = find_minima(t0, t1, finder, epsilon=EPS)
        for t, sep_v in zip(times, seps):
            if sep_v > 3.0: continue
            utc_dt = t.utc_datetime()
            ra_, dec_, _ = earth.at(t).observe(moon).apparent().radec()
            moon_illum = moon_illum_at(ts, eph, utc_dt)

            events.append({
                "id":               f"LCONJ_{label.split()[0].upper()}_{utc_dt.strftime('%Y%m%d')}",
                "category":         "lunar",
                "type":             f"Moon-{label} Conjunction",
                "subtype":          "conjunction",
                "target":           label,
                "utc":              utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ra_h":             round(ra_.hours, 3),
                "dec_d":            round(dec_.degrees, 2),
                "separation_deg":   round(float(sep_v), 2),
                "moon_illumination": round(float(moon_illum), 3),
                "global_event":     False,
                "precision":        "exact",
                "method":           "skyfield_root_finding",
                "description":      f"Moon passes within {sep_v:.2f}° of {label}.",
            })
    return events

def gen_planet_star_conjunctions(ts, eph, year: int) -> list[dict]:
    events = []
    earth = eph["earth"]
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31, 23, 59)
    EPS = 2.0 / 1440.0

    planets = [
        ("Venus", eph["venus"]), ("Mars", eph["mars barycenter"]), ("Jupiter", eph["jupiter barycenter"])
    ]
    stars = [
        ("Spica", Star(ra_hours=13.4199, dec_degrees=-11.1614), 0.98),
        ("Antares", Star(ra_hours=16.4901, dec_degrees=-26.4319), 1.06),
        ("Regulus", Star(ra_hours=10.1395, dec_degrees=11.9672), 1.35),
        ("Aldebaran", Star(ra_hours=4.5987, dec_degrees=16.5092), 0.87),
        ("Pollux", Star(ra_hours=7.7553, dec_degrees=28.0261), 1.14),
    ]

    for p_name, p_body in planets:
        for s_name, s_star, s_mag in stars:
            finder = SeparationFinder(earth, p_body, s_star, step_days=2.0)
            times, seps = find_minima(t0, t1, finder, epsilon=EPS)
            for t, sep_v in zip(times, seps):
                if sep_v >= 1.5: continue
                utc_dt = t.utc_datetime()
                ra_, dec_, _ = earth.at(t).observe(p_body).apparent().radec()
                events.append({
                    "id":             f"PCONJ_{p_name.upper()}_{s_name.upper()}_{utc_dt.strftime('%Y%m%d')}",
                    "category":       "planet",
                    "type":           f"{p_name}-{s_name} Conjunction",
                    "subtype":        "conjunction",
                    "planet_name":    p_name,
                    "star_name":      s_name,
                    "star_magnitude": s_mag,
                    "utc":            utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ra_h":           round(ra_.hours, 3),
                    "dec_d":          round(dec_.degrees, 2),
                    "separation_deg": round(float(sep_v), 2),
                    "global_event":   True,
                    "precision":      "exact",
                    "method":         "skyfield_root_finding",
                    "description":    f"{p_name} passes within {sep_v:.2f}° of star {s_name}.",
                })
    return events

def gen_lunar_apogee_perigee(ts, eph, year: int) -> list[dict]:
    t0, t1 = ts.utc(year, 1, 1), ts.utc(year, 12, 31, 23, 59)
    earth, moon = eph["earth"], eph["moon"]
    def dist_fn(t): return earth.at(t).observe(moon).apparent().distance().km
    dist_fn.step_days = 1.0

    times_peri, dists_peri = find_minima(t0, t1, dist_fn, epsilon=2.0/1440.0)
    times_apo,  dists_apo  = find_maxima(t0, t1, dist_fn, epsilon=2.0/1440.0)

    events = []
    for t, dist in zip(times_peri, dists_peri):
        utc_dt = t.utc_datetime()
        ra_, dec_, _ = earth.at(t).observe(moon).apparent().radec()
        events.append({
            "id":          f"LUN_PERIGEE_{utc_dt.strftime('%Y%m%d')}",
            "category":    "lunar",
            "type":        "Lunar Perigee",
            "subtype":     "perigee",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        round(ra_.hours, 3),
            "dec_d":       round(dec_.degrees, 2),
            "distance_km": round(float(dist), 0),
            "global_event":True,
            "precision":   "exact",
            "method":      "skyfield_find_minima",
            "description": f"Lunar Perigee — Moon closest at {int(round(float(dist), 0)):,} km.",
        })
    for t, dist in zip(times_apo, dists_apo):
        utc_dt = t.utc_datetime()
        ra_, dec_, _ = earth.at(t).observe(moon).apparent().radec()
        events.append({
            "id":          f"LUN_APOGEE_{utc_dt.strftime('%Y%m%d')}",
            "category":    "lunar",
            "type":        "Lunar Apogee",
            "subtype":     "apogee",
            "utc":         utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ra_h":        round(ra_.hours, 3),
            "dec_d":       round(dec_.degrees, 2),
            "distance_km": round(float(dist), 0),
            "global_event":True,
            "precision":   "exact",
            "method":      "skyfield_find_maxima",
            "description": f"Lunar Apogee — Moon furthest at {int(round(float(dist), 0)):,} km.",
        })
    return events

def gen_iota_occultations(year: int) -> list[dict]:
    events = []
    url = f"https://occultations.org/publications/rasc/{year}/All{year}.xml"
    _CACHE = os.path.join(SKYFIELD_CACHE, f"All{year}.xml")

    xml_text = None
    try:
        resp = http.get(url, timeout=15)
        resp.raise_for_status()
        xml_text = resp.text
        with open(_CACHE, "w", encoding="utf-8") as f: f.write(xml_text)
    except Exception:
        if os.path.exists(_CACHE):
            try:
                with open(_CACHE, "r", encoding="utf-8") as f: xml_text = f.read()
            except Exception: pass

    if not xml_text: return events

    try:
        root = ET.fromstring(xml_text)
        for ev in root.findall("Event"):
            try:
                star_text     = getattr(ev.find("Star"),     "text", None)
                obj_text      = getattr(ev.find("Object"),   "text", None)
                elements_text = getattr(ev.find("Elements"), "text", None)
                if not star_text or not obj_text or not elements_text: continue

                star_parts, obj_parts, elements_parts = star_text.split(","), obj_text.split(","), elements_text.split(",")
                if len(star_parts) < 5 or len(obj_parts) < 11 or len(elements_parts) < 6: continue

                star_name, star_mag = star_parts[0].strip(), float(star_parts[4])
                ast_num, ast_name, duration = obj_parts[0].strip(), obj_parts[1].strip(), float(obj_parts[10])

                if star_mag < 8.0 and duration > 2.0:
                    y_ev, m_ev, d_ev, frac_h = int(elements_parts[2]), int(elements_parts[3]), int(elements_parts[4]), float(elements_parts[5])
                    utc_dt = datetime(y_ev, m_ev, d_ev, tzinfo=timezone.utc) + timedelta(hours=frac_h)
                    utc_dt = (utc_dt + timedelta(seconds=0.5)).replace(microsecond=0)

                    events.append({
                        "id":               f"OCC_{ast_num}_{utc_dt.strftime('%Y%m%d')}",
                        "category":         "occultation",
                        "type":             f"Stellar Occultation by {ast_name}",
                        "subtype":          "occultation",
                        "utc":              utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "ra_h":             round(float(star_parts[1]), 3),
                        "dec_d":            round(float(star_parts[2]), 2),
                        "star_name":        star_name,
                        "star_magnitude":   star_mag,
                        "asteroid_name":    ast_name,
                        "asteroid_number":  ast_num,
                        "duration_seconds": duration,
                        "global_event":     False,
                        "precision":        "predicted",
                        "method":           "iota_rasc_predictions",
                        "description":      f"Star {star_name} (mag {star_mag}) occulted by ({ast_num}) {ast_name} for {duration:.1f}s.",
                    })
            except Exception: continue
    except Exception: pass
    return events

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="XTROBE Yearly Astronomical Event Generator v5.2")
    p.add_argument("--year", type=int, default=datetime.now(timezone.utc).year, help="Target year")
    args = p.parse_args()

    log.info(f"=== GENERATING FULL YEARLY CALENDAR FOR {args.year} ===")
    ts, eph = init_skyfield()

    all_events = []
    log.info("Computing Lunar Phases...")
    all_events += gen_lunar_phases(ts, eph, args.year)
    
    log.info("Computing Eclipses...")
    all_events += gen_eclipses(ts, eph, args.year)
    
    log.info("Computing Seasons...")
    all_events += gen_seasons(ts, eph, args.year)
    
    log.info("Computing Planetary Oppositions, Elongations & Conjunctions...")
    all_events += gen_planetary_events(ts, eph, args.year)
    
    log.info("Computing Meteor Shower Peaks...")
    all_events += gen_meteor_showers(ts, eph, args.year)
    
    log.info("Computing Algol Minima...")
    all_events += gen_algol_minima(ts, eph, args.year)
    
    log.info("Computing Expanded Variable Stars (Delta Cephei, Beta Lyrae, etc)...")
    all_events += gen_expanded_variable_stars(ts, eph, args.year)
    
    log.info("Computing Zodiacal Light Windows...")
    all_events += gen_zodiacal_light(ts, eph, args.year)
    
    log.info("Computing Lunar Conjunctions...")
    all_events += gen_lunar_conjunctions(ts, eph, args.year)
    
    log.info("Computing Planet-Star Conjunctions...")
    all_events += gen_planet_star_conjunctions(ts, eph, args.year)
    
    log.info("Computing Lunar Apogee & Perigee...")
    all_events += gen_lunar_apogee_perigee(ts, eph, args.year)
    
    log.info("Computing IOTA Major Stellar Occultations...")
    all_events += gen_iota_occultations(args.year)

    all_events.sort(key=lambda e: e.get("utc", ""))
    log.info(f"Total Yearly Events Computed: {len(all_events)}")

    messier    = [{"id":m[0],"name":m[1],"type":m[2],"ra_h":m[3],"dec_d":m[4],"mag":m[5],"size_am":m[6]} for m in MESSIER_CATALOG]
    caldwell_s = [{"id":c[0],"name":c[1],"type":c[2],"ra_h":c[3],"dec_d":c[4],"mag":c[5],"size_am":c[6]} for c in CALDWELL_SOUTH]

    doc = {
        "schema":    "xtrobe-yearly-5.2",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year":      args.year,
        "count":     len(all_events),
        "events":    all_events,
        "catalog":   {"messier": messier, "caldwell_south": caldwell_s, "algol": ALGOL}
    }

    out_path = os.path.join(OUTPUT_DIR, "yearly_events.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))

    log.info(f"Saved Yearly Events JSON to: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
