import requests
import json

API_URL = 'http://localhost:8000'

# 1. Test health check
print('Testing /health endpoint...')
try:
    response = requests.get(f'{API_URL}/health')
    print(f'Status: {response.status_code}')
    print(f'Response: {response.json()}\n')
except Exception as e:
    print(f"Health check failed: {e}\n")

# 2. Test predictions with scenarios
test_cases = [
    {
        'name': 'Normal Operation',
        'data': {
            'temperature': 70.0,
            'vibration': 0.4,
            'pressure': 95.0,
            'rpm': 1500.0,
            'age_days': 100
        }
    },
    {
        'name': 'High Risk Scenario',
        'data': {
            'temperature': 95.0,
            'vibration': 0.9,
            'pressure': 135.0,
            'rpm': 1500.0,
            'age_days': 320
        }
    }
]

for test in test_cases:
    print(f"Testing scenario: {test['name']}...")
    try:
        response = requests.post(
            f'{API_URL}/predict',
            json=test['data']
        )
        if response.status_code == 200:
            result = response.json()
            print(f'  Will Fail:   {result["will_fail"]}')
            print(f'  Probability: {result["probability"]:.3f}')
            print(f'  Latency:     {result["latency_ms"]} ms')
            print(f'  Recommendation: {result["recommendation"]}\n')
        else:
            print(f"  Failed with status code: {response.status_code}\n")
    except Exception as e:
        print(f"  Prediction request failed: {e}\n")
        