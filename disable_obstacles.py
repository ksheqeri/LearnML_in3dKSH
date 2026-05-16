import requests
# replace the session ID with the one printed in the first terminal
session_id = "ec07f632-2382-4be8-8894-52b195794e15"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
