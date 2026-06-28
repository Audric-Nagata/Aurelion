import pytest
from tools.utils import extract_code, parse_json_block


def test_extract_code_with_block():
    result = extract_code(
        "Some text\n```python\ndef foo():\n    pass\n```\nmore text"
    )
    assert result == "def foo():\n    pass"


def test_extract_code_without_block():
    result = extract_code("just plain text")
    assert result == "just plain text"


def test_parse_json_block_found():
    result = parse_json_block('Here is {"key": "value", "num": 42} the end')
    assert result == {"key": "value", "num": 42}


def test_parse_json_block_no_match():
    result = parse_json_block("no json here")
    assert result == {}
