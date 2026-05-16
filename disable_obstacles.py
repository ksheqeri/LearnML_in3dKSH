import requests
# replace the session ID with the one printed in the first terminal
session_id = "9cb12f83-d4f7-4dde-a9ca-17df9eacc457"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
