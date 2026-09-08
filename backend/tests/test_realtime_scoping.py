"""
Regression tests for the realtime WebSocket channel, which used to send
every published event to every connected client. That meant a viewer with
the alerts page open received other users' `upload.progress`,
`upload.completed`, and `alert.created` payloads - the last of which
carries `source_name`, i.e. the other user's original video filename -
silently undoing the REST layer's authorization.

These exercise the delivery filter directly rather than through a live
socket: what matters is *who is selected as a recipient*, and driving that
through real WebSocket handshakes would test asyncio plumbing instead.
"""
from app.services.realtime import RealtimeBroadcaster, Subscriber, _may_receive


def _drain(coro):
    """Runs an already-created coroutine to completion without an event
    loop. The bodies here only append to a list, so they never suspend."""
    try:
        coro.send(None)
    except StopIteration:
        pass
    return None


def _subscriber(user_id: int, is_operator: bool = False) -> Subscriber:
    # The websocket object is never touched by the filter under test.
    return Subscriber(websocket=object(), user_id=user_id, is_operator=is_operator)


def test_shared_events_reach_every_subscriber():
    """Camera alerts and camera status are shared infrastructure - every
    authenticated user can already list every camera over REST."""
    assert _may_receive(_subscriber(1), owner_user_id=None)
    assert _may_receive(_subscriber(2), owner_user_id=None)
    assert _may_receive(_subscriber(3, is_operator=True), owner_user_id=None)


def test_owner_receives_their_own_upload_events():
    assert _may_receive(_subscriber(7), owner_user_id=7)


def test_other_users_do_not_receive_someone_elses_upload_events():
    assert not _may_receive(_subscriber(8), owner_user_id=7)


def test_operators_receive_every_users_upload_events():
    assert _may_receive(_subscriber(9, is_operator=True), owner_user_id=7)


def test_publish_selects_only_permitted_subscribers(monkeypatch):
    """End-to-end through publish(): the owner and an operator are picked,
    an unrelated viewer is not."""
    broadcaster = RealtimeBroadcaster()
    broadcaster._loop = object()  # non-None so publish proceeds

    owner = _subscriber(1)
    bystander = _subscriber(2)
    operator = _subscriber(3, is_operator=True)
    broadcaster._subscribers = {owner, bystander, operator}

    sent_to = []

    async def _record(subscriber, message):
        sent_to.append(subscriber)

    broadcaster._safe_send = _record
    # publish() hands the coroutine to the event loop; here we just run it
    # inline so the recipient list is populated synchronously.
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", lambda coro, loop: _drain(coro))

    broadcaster.publish("alert.created", {"source_name": "private.mp4"}, owner_user_id=1)

    assert set(sent_to) == {owner, operator}
    assert bystander not in sent_to


def test_websocket_rejects_a_token_for_a_deactivated_user(client, db, make_user_factory):
    from app.core.security import create_access_token

    user = make_user_factory("deactivated@example.com")
    token = create_access_token(user.id, user.email)
    user.is_active = False
    db.commit()

    try:
        with client.websocket_connect(f"/api/ws/events?token={token}"):
            pass
        raise AssertionError("expected the connection to be rejected")
    except AssertionError:
        raise
    except Exception:
        pass  # starlette raises when the server closes during the handshake
