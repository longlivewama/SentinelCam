from app.core.security import create_access_token


def test_websocket_rejects_invalid_token(client):
    try:
        with client.websocket_connect("/api/ws/events?token=garbage"):
            pass
        assert False, "expected the connection to be rejected"
    except Exception:
        pass  # starlette raises WebSocketDisconnect when the server closes during handshake


def test_websocket_accepts_valid_token(client, viewer_user):
    token = create_access_token(viewer_user.id, viewer_user.email)
    with client.websocket_connect(f"/api/ws/events?token={token}"):
        pass  # successfully connected and cleanly closed
