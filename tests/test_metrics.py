from ftpipe.metrics import head_to_head, parse_calls, score_example, score_records


def test_parse_clean_array():
    out = parse_calls('[{"name": "get_time", "arguments": {"city": "Paris"}}]')
    assert out == [{"name": "get_time", "arguments": {"city": "Paris"}}]


def test_parse_prose_wrapped():
    text = 'Sure! Here you go:\n```json\n[{"name": "a", "arguments": {}}]\n```\nHope that helps.'
    assert parse_calls(text) == [{"name": "a", "arguments": {}}]


def test_parse_single_object_promoted_to_list():
    assert parse_calls('{"name": "a", "arguments": {"x": 1}}') == [
        {"name": "a", "arguments": {"x": 1}}
    ]


def test_parse_garbage_returns_none():
    assert parse_calls("I cannot help with that.") is None
    assert parse_calls("") is None
    assert parse_calls(None) is None


def test_parse_empty_array():
    assert parse_calls("[]") == []


def test_score_perfect():
    gold = [{"name": "w", "arguments": {"city": "Paris"}}]
    s = score_example(parse_calls('[{"name":"w","arguments":{"city":"Paris"}}]'), gold, {"w"})
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0
    assert s["exact_calls"] == 1 and s["hallucinated"] is False


def test_score_wrong_tool_and_hallucination():
    gold = [{"name": "w", "arguments": {"city": "Paris"}}]
    s = score_example([{"name": "nope", "arguments": {}}], gold, {"w", "t"})
    assert s["fn"] == 1 and s["fp"] == 1
    assert s["exact_calls"] == 0 and s["hallucinated"] is True


def test_score_right_tool_wrong_arg():
    gold = [{"name": "w", "arguments": {"city": "Paris"}}]
    s = score_example([{"name": "w", "arguments": {"city": "Lyon"}}], gold, {"w"})
    assert s["tp"] == 1 and s["exact_calls"] == 0
    assert s["arg_matched"] == 0 and s["arg_total"] == 1


def test_invalid_json_scored_as_wrong():
    gold = [{"name": "w", "arguments": {}}]
    s = score_example(parse_calls("garbage"), gold, {"w"})
    assert s["valid_json"] is False and s["exact_calls"] == 0 and s["fn"] == 1


def test_score_records_splits():
    records = [
        {"id": 1, "query": "q", "tools": [{"name": "w"}], "gold_calls": [{"name": "w", "arguments": {}}],
         "split_kind": "id", "prediction": '[{"name":"w","arguments":{}}]'},
        {"id": 2, "query": "q", "tools": [{"name": "w"}], "gold_calls": [{"name": "w", "arguments": {}}],
         "split_kind": "ood", "prediction": "nonsense"},
    ]
    out = score_records(records)
    assert out["by_split"]["id"]["exact_call_match"] == 1.0
    assert out["by_split"]["ood"]["exact_call_match"] == 0.0
    assert out["by_split"]["ood"]["json_validity"] == 0.0


def test_head_to_head_buckets():
    base = [{"id": 1, "query": "q", "tools": [{"name": "w"}], "gold_calls": [{"name": "w", "arguments": {}}],
             "prediction": "bad"}]
    ft = [{"id": 1, "query": "q", "tools": [{"name": "w"}], "gold_calls": [{"name": "w", "arguments": {}}],
           "prediction": '[{"name":"w","arguments":{}}]'}]
    h = head_to_head(base, ft)
    assert h["n_helped"] == 1 and h["n_hurt"] == 0
