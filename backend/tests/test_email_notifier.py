from app.services.notifications.email_notifier import EmailNotifier


class _FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=10):
        self.host = host
        self.port = port
        self.starttls_called = False
        self.login_called_with = None
        self.sent = None
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, user, password):
        self.login_called_with = (user, password)

    def sendmail(self, from_addr, to_addrs, message):
        self.sent = (from_addr, to_addrs, message)


def test_starttls_called_when_use_tls_enabled(monkeypatch):
    import app.config as config_module
    import app.services.notifications.email_notifier as mod

    monkeypatch.setattr(config_module.settings, "SMTP_USE_TLS", True)
    monkeypatch.setattr(config_module.settings, "SMTP_USER", "")
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(mod.smtplib, "SMTP", _FakeSMTP)

    assert EmailNotifier().send("subject", "body", ["to@example.com"]) is True
    assert _FakeSMTP.instances[0].starttls_called is True


def test_starttls_skipped_when_use_tls_disabled(monkeypatch):
    """Needed for local/e2e SMTP catchers (e.g. Mailpit) that don't
    support STARTTLS at all - calling it unconditionally would break them."""
    import app.config as config_module
    import app.services.notifications.email_notifier as mod

    monkeypatch.setattr(config_module.settings, "SMTP_USE_TLS", False)
    monkeypatch.setattr(config_module.settings, "SMTP_USER", "")
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(mod.smtplib, "SMTP", _FakeSMTP)

    assert EmailNotifier().send("subject", "body", ["to@example.com"]) is True
    assert _FakeSMTP.instances[0].starttls_called is False


def test_login_skipped_when_no_smtp_user(monkeypatch):
    import app.config as config_module
    import app.services.notifications.email_notifier as mod

    monkeypatch.setattr(config_module.settings, "SMTP_USE_TLS", False)
    monkeypatch.setattr(config_module.settings, "SMTP_USER", "")
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(mod.smtplib, "SMTP", _FakeSMTP)

    EmailNotifier().send("subject", "body", ["to@example.com"])
    assert _FakeSMTP.instances[0].login_called_with is None


def test_send_returns_false_with_no_recipients():
    assert EmailNotifier().send("subject", "body", []) is False
