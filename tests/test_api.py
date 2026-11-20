"""API 冒烟测试:页面、示例、计划、续排、失败、校验错误。"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
client = TestClient(app)


def example_body():
    def load(name):
        return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))

    return {
        "topology": load("topology.json"),
        "routes": load("routes.json"),
        "requirements": load("requirements.json"),
    }


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "换表轮次规划器" in r.text


def test_example_endpoint():
    r = client.get("/api/example")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"topology", "routes", "requirements"}


def test_plan_ok():
    r = client.post("/api/plan", json={**example_body(), "switched": []})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert len(d["rounds"]) == 3
    assert sorted(r0 for rnd in d["rounds"] for r0 in rnd) == sorted(
        ["R1", "R2", "R3", "R4", "R5"]
    )


def test_plan_replan_with_switched():
    r = client.post("/api/plan", json={**example_body(), "switched": ["R4", "R5"]})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["switched"] == ["R4", "R5"]
    assert len(d["rounds"]) == 2


def test_plan_failure_shape():
    body = example_body()
    body["requirements"]["requirements"][0]["allowed_egress"] = ["E2"]
    r = client.post("/api/plan", json={**body, "switched": []})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "failed"
    assert d["reason"] == "current_unsafe"
    assert d["failure_state"] == []
    assert d["first_failures"] == []  # 当前不通时不混入首次故障组合
    f0 = d["failures"][0]
    assert f0["verdict"] == "wrong_egress"
    assert f0["verdict_zh"] == "错出口"
    assert f0["path"][0]["router"] == "R1"


def test_plan_first_failure_groups_shape():
    """无法安全换完时:返回去重的首次故障组合,每组带次序与故障路径。"""
    body = example_body()
    body["routes"]["R4"]["after"] = [
        {"prefix": "10.1.0.0/16", "next_hop": "10.0.14.1"}
    ]
    r = client.post("/api/plan", json={**body, "switched": []})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "failed"
    assert d["reason"] == "unreachable"
    groups = d["first_failures"]
    assert len(groups) > 1
    assert len({tuple(g["switched"]) for g in groups}) == len(groups)
    for g in groups:
        assert g["order"][-1] == g["trigger"]
        assert sorted(g["switched"]) == sorted(set(d["switched"]) | set(g["order"]))
        assert g["failures"]
        for f in g["failures"]:
            assert f["verdict"] in {"blackhole", "loop", "wrong_egress"}
            assert f["path"] and f["path"][0]["router"] == f["ingress"]
            assert f["target_address"]
    # 兼容字段取组数最少的那一组
    assert d["failure_state"] == min((g["switched"] for g in groups), key=len)


def test_plan_ok_reports_minimal_rounds_with_start():
    body = example_body()
    r = client.post("/api/plan", json={**body, "switched": ["R4"]})
    d = r.json()
    assert d["status"] == "ok"
    assert d["first_failures"] == []
    assert d["failure_state"] is None
    assert len(d["rounds"]) == 3


def test_validation_error_returns_400():
    body = example_body()
    body["requirements"]["requirements"][0]["allowed_egress"] = ["NOPE"]
    r = client.post("/api/plan", json={**body, "switched": []})
    assert r.status_code == 400
    assert "不存在的出口" in r.json()["detail"]
