"""Тесты whitelist-функции is_normal_disconnect.

Любой false-positive (нормальный disconnect классифицирован как сбой) ->
ложный СРОЧНО-лид в Bitrix.
Любой false-negative (реальный сбой классифицирован как disconnect) ->
пропустим реальную проблему.
"""
from backend.services.disconnect_policy import is_normal_disconnect


class TestNormalDisconnectsWhitelisted:
    """Эти должны возвращать True — НЕ триггерить fatal_fallback."""

    def test_starlette_runtime_error_receive_after_close(self):
        exc = RuntimeError(
            'Cannot call "receive" once a disconnect message has been received.'
        )
        assert is_normal_disconnect(exc) is True

    def test_starlette_runtime_error_send_after_close(self):
        exc = RuntimeError(
            'Cannot call "send" once a close message has been sent.'
        )
        assert is_normal_disconnect(exc) is True

    def test_broken_pipe_error(self):
        assert is_normal_disconnect(BrokenPipeError("Broken pipe")) is True

    def test_connection_reset_error(self):
        assert is_normal_disconnect(ConnectionResetError("Connection reset")) is True

    def test_generic_connection_closed_message(self):
        assert is_normal_disconnect(Exception("Connection closed by peer")) is True

    def test_websocket_not_connected_message(self):
        assert is_normal_disconnect(RuntimeError("WebSocket is not connected.")) is True

    def test_going_away_code(self):
        assert is_normal_disconnect(Exception("code=1001, reason=Going Away")) is True

    def test_closedresourceerror_by_name(self):
        # Имитируем anyio.ClosedResourceError без реального anyio
        class ClosedResourceError(Exception):
            pass
        assert is_normal_disconnect(ClosedResourceError()) is True

    def test_websocketdisconnect_by_name(self):
        # Имитируем starlette WebSocketDisconnect без импорта starlette
        class WebSocketDisconnect(Exception):
            pass
        assert is_normal_disconnect(WebSocketDisconnect()) is True


class TestRealFailuresNotWhitelisted:
    """Реальные сбои — должны возвращать False (→ fatal_fallback)."""

    def test_value_error(self):
        assert is_normal_disconnect(ValueError("bad payload")) is False

    def test_key_error(self):
        assert is_normal_disconnect(KeyError("missing_key")) is False

    def test_type_error(self):
        assert is_normal_disconnect(TypeError("wrong type")) is False

    def test_attribute_error(self):
        assert is_normal_disconnect(AttributeError("no attr")) is False

    def test_generic_runtime_error_unrelated(self):
        assert is_normal_disconnect(RuntimeError("something else broke")) is False

    def test_empty_exception(self):
        assert is_normal_disconnect(Exception("")) is False

    def test_os_error_enoent(self):
        assert is_normal_disconnect(OSError("[Errno 2] No such file")) is False

    def test_zero_division(self):
        assert is_normal_disconnect(ZeroDivisionError("division by zero")) is False
