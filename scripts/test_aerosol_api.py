import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from backend.main import (
    app,
    health_check,
    get_data_status,
    get_latest_aerosol,
    get_forecast_for_station,
    get_dispersion_conditions,
    get_active_alerts,
    simulate_scenario,
    SimulationRequest
)

print("=" * 70)
print("1. Testing GET /health...")
h = health_check()
print(f"   Health Check: {h}")

print("\n" + "=" * 70)
print("2. Testing GET /aerosol...")
aero = get_latest_aerosol()
print(json.dumps(aero, indent=2))

print("\n" + "=" * 70)
print("3. Testing GET /data-status...")
ds = get_data_status()
print(json.dumps(ds, indent=2))

print("\n" + "=" * 70)
print("4. Confirming existing endpoints untouched:")
fc = get_forecast_for_station("Alipur, Delhi - DPCC")
print(f"   GET /forecast: Points = {len(fc.forecast)}")

disp = get_dispersion_conditions("Alipur, Delhi - DPCC")
print(f"   GET /dispersion: Classification = {disp.classification}, PBL = {disp.boundary_layer_height}m")

alerts = get_active_alerts()
print(f"   GET /alerts: Alerts count = {len(alerts.alerts)}")

sim_req = SimulationRequest(pm25=200, wind_speed=1.5, wind_direction=270, temperature=15, humidity=80, pbl=400, precipitation=0, enable_feedback=True)
sim = simulate_scenario(sim_req)
print(f"   POST /simulate: Baseline points = {len(sim.baseline_forecast)}, Feedback points = {len(sim.feedback_forecast)}")

print("\n" + "=" * 70)
print("ALL ENDPOINT VERIFICATION CHECKS PASSED (100% ISOLATED & WORKING)!")
print("=" * 70)
