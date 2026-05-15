import requests
# replace the session ID with the one printed in the first terminal
session_id = "e184967d-627c-43f0-b6d7-555917560fa7"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
