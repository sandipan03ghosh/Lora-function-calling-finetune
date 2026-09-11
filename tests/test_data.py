from ftpipe.config import DataConfig
from ftpipe.data import build_splits, valid_row


def _tool(name, params):
    return {"name": name, "parameters": {p: {"type": "str"} for p in params}}


def test_valid_row_accepts_good():
    tools = [_tool("w", ["city"])]
    calls = [{"name": "w", "arguments": {"city": "Paris"}}]
    assert valid_row(tools, calls) is True


def test_valid_row_rejects_unknown_tool():
    assert valid_row([_tool("w", ["city"])], [{"name": "x", "arguments": {}}]) is False


def test_valid_row_rejects_unknown_arg():
    tools = [_tool("w", ["city"])]
    assert valid_row(tools, [{"name": "w", "arguments": {"country": "FR"}}]) is False


def test_valid_row_rejects_empty_calls():
    assert valid_row([_tool("w", ["city"])], []) is False


def _rows(n):
    out = []
    for i in range(n):
        tool = f"tool_{i % 20}"
        out.append(
            {
                "id": i,
                "query": f"please run task number {i}",
                "tools": [_tool(tool, ["a"])],
                "gold_calls": [{"name": tool, "arguments": {"a": str(i)}}],
            }
        )
    return out


def test_build_splits_ratios_and_no_leak():
    cfg = DataConfig(max_examples=200, ood_tool_fraction=0.2, seed=1)
    res = build_splits(_rows(400), cfg)
    sizes = res["stats"]["split_sizes"]

    # in-distribution rows are capped at max_examples and split ~80/10/10
    id_total = sizes["train"] + sizes["val"] + sizes["test_id"]
    assert id_total <= cfg.max_examples
    assert abs(sizes["train"] / id_total - 0.8) < 0.05

    # held-out tools never appear in train
    heldout = set(res["stats"]["heldout_tools_sample"])
    train_tools = {c["name"] for r in res["splits"]["train"] for c in r["gold_calls"]}
    assert not (train_tools & heldout)
    assert sizes["test_ood"] > 0


def test_build_splits_is_deterministic():
    cfg = DataConfig(max_examples=100, seed=7)
    a = build_splits(_rows(300), cfg)["splits"]["train"]
    b = build_splits(_rows(300), cfg)["splits"]["train"]
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_build_splits_dedupes_queries():
    rows = _rows(50) + _rows(50)  # exact duplicates
    res = build_splits(rows, DataConfig(max_examples=1000, seed=3))
    assert res["stats"]["n_after_dedup"] == 50
