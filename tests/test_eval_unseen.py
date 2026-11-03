"""저장된 미지 질문 결과 채점과 앱의 이어받기 규칙을 검증한다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_unseen import report, run, score, summary, update_carry


def test_score_synthetic_file(tmp_path):
    gold = {"single": [
        {"id": "U1", "question": "프레스 교체", "equipment": ["생산설비(프레스)"], "situations": ["교체"]},
        {"id": "U2", "question": "점심", "equipment": [], "situations": [], "guidance": "need_context"}],
        "multi": [{"id": "M1", "turns": ["지게차", "로봇 점검"], "first_guidance": "need_situation",
                   "equipment": ["산업용로봇"], "situations": ["점검"], "not_equipment": ["지게차"]}]}
    results = {"single": [
        {"이름": "아무 이름", "질문": "점심", "조건": {"설비": [], "작업": []}, "안내": "need_context", "사례": 0},
        {"질문": "프레스 교체", "조건": {"설비": ["생산설비(프레스)"], "작업": ["교체"]}, "사례": 2}],
        "multi": [{"대화": [
            {"질문": "지게차", "안내": "need_situation"},
            {"질문": "로봇 점검", "조건": {"설비": ["지게차", "산업용로봇"], "작업": ["점검"]}, "사례": 1}]}]}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    scored = score(path, gold)
    assert scored["totals"]["all"] == {"passed": 2, "total": 3}
    assert scored["totals"]["not_equipment"] == {"passed": 0, "total": 1}
    assert scored["items"][2]["first_guidance"] is True
    assert "all: 2/3" in summary(scored) and "| M1 |" in report(scored)
    results["single"][0]["조건"]["작업"] = ["추측한 작업"]
    assert score(results, gold)["items"][1]["situations"] is False
    assert score({"single": [], "multi": []}, gold)["totals"]["all"]["passed"] == 0


def test_carry_matches_app():
    conditions = {"targets": [], "situations": ["작업"]}
    out = {"conditions": conditions, "guidance": {"type": "need_equipment"}}
    assert update_carry(out, 0) == (conditions, 1)
    assert update_carry(out, 1) == (None, 0)
    assert update_carry({**out, "llm_error": "오류"}, 0) == (None, 0)
    assert update_carry({**out, "guidance": {"type": "need_context"}}, 0) == (None, 0)


def test_run_with_fake_agent_and_ids():
    class Agent:
        def __init__(self):
            self.carries = []

        def ask(self, question, carry=None):
            self.carries.append(carry)
            return {"conditions": {"targets": [], "situations": ["작업"]},
                    "guidance": {"type": "need_equipment"}}

    agent = Agent()
    gold = {"single": [], "multi": [{"id": "M1", "turns": ["첫째", "둘째", "셋째"]}]}
    results = run(gold, agent)
    assert results["multi"][0]["id"] == "M1"
    assert [bool(c) for c in agent.carries] == [False, True, False]
    assert [t["이어받음"] for t in results["multi"][0]["대화"]] == [False, True, False]
