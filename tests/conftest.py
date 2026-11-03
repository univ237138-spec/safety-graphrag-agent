"""테스트는 외부 네트워크에 연결하지 않는다 (로컬 Neo4j 일치 검사만 localhost 허용)."""
import socket

import pytest

LOCAL = ("localhost", "127.0.0.1", "::1")


def guard(original, host_of):
    def call(*args, **kwargs):
        address = host_of(args)
        if (address[0] if isinstance(address, tuple) else address) not in LOCAL:
            raise AssertionError("테스트에서 외부 네트워크 연결 금지")
        return original(*args, **kwargs)
    return call


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import llm

    original_env = llm.env
    monkeypatch.setattr(llm, "env", lambda name, default=None:
                        default if name == "OPENAI_API_KEY" else original_env(name, default))
    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect, lambda a: a[1]))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(socket.socket.connect_ex, lambda a: a[1]))
    monkeypatch.setattr(socket, "create_connection", guard(socket.create_connection, lambda a: a[0]))
    monkeypatch.setattr(socket, "getaddrinfo", guard(socket.getaddrinfo, lambda a: a[0]))
