import requests
import sys

session_id = input("Paste session ID: ")

resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={
        "obstacles_enabled": True,
        "obstacles_count": 1000,
        "obstacle_density": 100.0,
    }
)
print(resp.status_code, resp.json())