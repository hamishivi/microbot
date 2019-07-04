from starlette.testclient import TestClient

from main import app

client = TestClient(app)

# just check home page loads
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

# ensure response for three modes
# we require the startup to happen first before
# we can test
def test_chat_response_friend():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello!", "mode": "friend"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Hi!"

def test_chat_response_professional():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello!", "mode": "professional"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Hello."

def test_chat_response_comic():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello!", "mode": "comic"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Hey."
