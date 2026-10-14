import json
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def get_json(url, payload=None):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main():
    status, home = get_json(f"{BASE_URL}/")
    assert status == 200
    assert home["expected_features"] == 30
    print("Home endpoint: PASS")

    # Synthetic inputs for an educational API smoke test.
    features = [1.0] * 30
    status, result = get_json(
        f"{BASE_URL}/predict",
        {"features": features},
    )

    assert status == 200
    assert result["predicted_class"] in {"malignant", "benign"}
    assert result["educational_use_only"] is True

    if "probabilities" in result:
        probabilities = result["probabilities"]
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1.0) < 0.001

    print("Prediction endpoint: PASS")
    print("Docker API smoke tests passed")


if __name__ == "__main__":
    main()
