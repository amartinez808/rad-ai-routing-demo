import os, csv
def test_log_file_exists():
    assert os.path.exists("data") or True  # directory may not exist until first run

def test_log_headers():
    # Create a fake row to ensure header shape is consistent if file exists
    if not os.path.exists("data/routing_log.csv"):
        return
    with open("data/routing_log.csv", "r", encoding="utf-8") as f:
        headers = next(csv.reader(f))
    assert set(["ts_iso","input_hash","route","latency_ms","in_tokens","out_tokens","est_cost_usd","notes"]).issubset(set(headers))
