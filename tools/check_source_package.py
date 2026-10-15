"""Validate examples and parse Python helpers without model files or SDKs."""
# SPDX-License-Identifier: MIT
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for script in (root / 'tools').glob('*.py'):
    ast.parse(script.read_text(encoding='utf-8-sig'), filename=str(script))
for example in (root / 'examples').glob('*.json'):
    json.loads(example.read_text(encoding='utf-8-sig'))
request = json.loads((root / 'examples/request.json').read_text(encoding='utf-8'))
assert request['cpu_workers'] == 4 and request['body_workers'] == 1
assert request['gpu_head'] is False
assert all(isinstance(t, int) and 0 <= t < 262144 for t in request['tokens'])
print('Python helpers and public JSON examples validated')
