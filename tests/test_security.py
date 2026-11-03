"""보안 점검: 키 유출 방지 + 다른 프로젝트 접속 정보와 분리. 값은 출력하지 않는다."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import load_neo4j  # noqa: E402
import settings  # noqa: E402

SECRET_KEYS = ("DATA_GO_KR_KEY", "OPENAI_API_KEY", "NEO4J_PASSWORD")
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def publishable_files():
    """git에 올라갈 수 있는 파일 (추적 중 + 추적 전이지만 무시되지 않은 것)."""
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    return [ROOT / n for n in names if n]


def test_env_is_ignored():
    assert git("check-ignore", ".env").strip() == ".env"
    assert not any(p.name == ".env" for p in publishable_files())


def test_no_secret_values_in_publishable_files():
    secrets = [v for v in (settings.env(k) for k in SECRET_KEYS) if v and len(v) >= 8]
    leaked = []
    for path in publishable_files():
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", "ignore")
        if any(s in text for s in secrets) or KEY_PATTERN.search(text):
            leaked.append(path.name)
    assert not leaked, f"키로 보이는 값이 들어간 파일: {leaked}"


@pytest.mark.parametrize("uri", ["neo4j+s://abcd1234.databases.neo4j.io", "bolt://192.168.0.10:7687"])
def test_loader_refuses_non_local_neo4j(monkeypatch, uri):
    monkeypatch.setattr(load_neo4j, "env", lambda name, default=None: uri if name == "NEO4J_URI" else default)
    with pytest.raises(SystemExit):
        load_neo4j.local_uri()


def test_settings_ignore_system_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://cloud.databases.neo4j.io")
    assert "databases.neo4j.io" not in (settings.env("NEO4J_URI") or "")


def test_no_reference_to_other_project():
    for path in publishable_files():
        if path.suffix in (".py", ".yml", ".toml", ".cfg") and path.name != Path(__file__).name:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "recipe-agent" not in text and "NEO4J_CREDENTIALS_FILE" not in text, path.name
