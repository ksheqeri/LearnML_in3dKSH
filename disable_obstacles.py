import requests
# replace the session ID with the one printed in the first terminal
session_id = "e56d48df-1afc-4495-8d33-b65927bf8157"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
