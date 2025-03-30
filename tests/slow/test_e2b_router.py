import requests
import json

# 📍 Your API endpoint
API_URL = "http://localhost:8000/execute_batch"

# 📜 Full Python scripts as strings
scripts = [
    {
        "code": """
print("1")
"""
    }
]

# 🔁 Wrap scripts into request payload
payload = {"scripts": scripts, "language": "python"}

# 🚀 Send POST request
response = requests.post(API_URL, json=payload)

# 📋 Print results
if response.ok:
    results = response.json()
    for i, result in enumerate(results):
        print(f"\n🔹 Script #{i+1}")
        if result["error"]:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Result: {result['result']}")
else:
    print("❌ Request failed:", response.status_code)
    print(response.text)
