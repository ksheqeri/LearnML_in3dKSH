import requests
# replace the session ID with the one printed in the first terminal
session_id = "55767882-6541-4be9-beca-cac92c65607e"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
