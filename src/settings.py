"""설정은 safety-agent/.env 에서만 읽는다 (시스템 환경변수는 보지 않음).

다른 프로젝트의 NEO4J_* 환경변수가 실수로 섞이지 않게 하려는 것.
값은 절대 출력하지 않는다.
"""
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"


def env(name, default=None):
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name and value.strip():
                return value.strip()
    return default
