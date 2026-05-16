import requests
# replace the session ID with the one printed in the first terminal
session_id = "daa91c3a-fb51-4be2-abbc-4172a40f2ae9"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
