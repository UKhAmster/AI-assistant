"""Политика различения «нормальное закрытие WebSocket» vs «реальный сбой бота».

`_fatal_fallback` триггерит СРОЧНО-лид в Bitrix. Он должен срабатывать ТОЛЬКО
на реальные сбои — не на рядовые disconnect'ы (закрытие вкладки, ребут
контейнера, моргание сети). Эта функция — whitelist-фильтр.
"""
from __future__ import annotations

# Сигнатуры нормального закрытия по тексту исключения (case-insensitive).
_NORMAL_DISCONNECT_MSG_MARKERS: tuple[str, ...] = (
    "disconnect message has been received",   # starlette при receive() после close
    'cannot call "receive"',                  # starlette после close (общий)
    'cannot call "send"',                     # starlette после close
    "connection closed",                      # websockets lib
    "websocket is not connected",             # starlette guard
    "broken pipe",                            # OS-level write после close
    "going away",                             # WS close code 1001
    "no close frame received",                # abrupt client disconnect
)

# Имена классов-исключений (case-insensitive) которые заведомо = normal close.
_NORMAL_DISCONNECT_EXC_NAMES: frozenset[str] = frozenset({
    "websocketdisconnect",     # starlette.websockets.WebSocketDisconnect
    "connectionclosed",        # websockets.exceptions.ConnectionClosed
    "connectionclosedok",      # 1000 / 1001 normal close
    "connectionclosederror",   # abnormal close
    "connectionreseterror",    # TCP RST
    "connectionabortederror",  # OSError ConnectionAborted
    "connectionaborted",       # OSError ConnectionAborted (alt name)
    "closedresourceerror",     # anyio.ClosedResourceError
    "brokenpipeerror",         # OSError BrokenPipe
    "endofstream",             # anyio.EndOfStream
})


def is_normal_disconnect(exc: BaseException) -> bool:
    """True если exc — нормальное закрытие соединения, False если реальный сбой.

    Проверяет по имени класса И по тексту сообщения. False-positive приводит
    к ложному СРОЧНО-лиду; false-negative пропустит реальную ошибку. Потому
    whitelist консервативен — все маркеры должны быть чётко связаны с сетевым
    закрытием.
    """
    if type(exc).__name__.lower() in _NORMAL_DISCONNECT_EXC_NAMES:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _NORMAL_DISCONNECT_MSG_MARKERS)
