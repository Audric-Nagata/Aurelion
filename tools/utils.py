import json
import re


def extract_code(text: str) -> str:
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def parse_json_block(text: str) -> dict:
    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
