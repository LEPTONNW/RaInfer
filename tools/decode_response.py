"""Decode a saved @DFLASH2 result using local tokenizer assets; no downloads."""
# SPDX-License-Identifier: MIT
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    found = False
    for line in args.input.read_text(encoding='utf-8').splitlines():
        if line.startswith('@DFLASH2 '):
            event = json.loads(line[len('@DFLASH2 '):])
            if event.get('event') == 'result':
                print(tokenizer.decode(event['report']['tokens'], skip_special_tokens=True))
                found = True
    if not found:
        parser.error('No @DFLASH2 result record found')


if __name__ == '__main__':
    main()
