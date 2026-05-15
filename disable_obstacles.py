import requests
# replace the session ID with the one printed in the first terminal
session_id = "d44d446a-afc2-4ebe-a823-63b19d538540"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
