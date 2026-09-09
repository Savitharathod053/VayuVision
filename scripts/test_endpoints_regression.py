"""Quick regression test for VayuDrishti endpoints after V3 model swap."""
import urllib.request, json

BASE = "http://localhost:8000"
STATIONS = [
    "Anand%20Vihar%2C%20Delhi%20-%20DPCC",
    "R%20K%20Puram%2C%20Delhi%20-%20DPCC",
    "Mandir%20Marg%2C%20New%20Delhi%20-%20DPCC",
]
ENDPOINTS = ["/forecast/", "/explain/", "/dispersion/"]

all_ok = True
results = []

for station in STATIONS:
    for ep in ENDPOINTS:
        url = BASE + ep + station
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                status = r.getcode()
                body = json.loads(r.read().decode())
                ok = (status == 200)
                notes = []
                if ep == "/forecast/":
                    fc = body.get("forecast", [])
                    notes.append("forecast_hours=" + str(len(fc)))
                    if fc:
                        notes.append("first_pm25=" + str(fc[0].get("predicted_pm25")))
                        notes.append("has_ci=" + str("expected_low" in fc[0]))
                elif ep == "/explain/":
                    factors = body.get("top_factors", [])
                    notes.append("top_factors=" + str(len(factors)))
                    if factors:
                        notes.append("top=" + str(factors[0].get("feature")))
                        notes.append("shap=" + str(round(factors[0].get("shap_value", 0), 2)))
                elif ep == "/dispersion/":
                    notes.append("risk_index=" + str(body.get("dispersion_risk_index")))
                    notes.append("category=" + str(body.get("risk_category", "?")))
                results.append((status, ok, ep.rstrip("/"), station[:30].replace("%20", " "), ", ".join(notes)))
        except Exception as e:
            results.append((0, False, ep.rstrip("/"), station[:30].replace("%20", " "), str(e)[:70]))
            all_ok = False

print(f"{'HTTP':>4}  {'OK':>5}  {'Endpoint':<14} {'Station':<32} Notes")
print("-" * 100)
for status, ok, ep, st, notes in results:
    mark = "OK" if ok else "FAIL"
    print(f"{status:>4}  {mark:>5}  {ep:<14} {st:<32} {notes}")

print()
all_passed = all_ok and all(r[1] for r in results)
print("=== ALL PASS ===" if all_passed else "=== FAILURES DETECTED ===")
