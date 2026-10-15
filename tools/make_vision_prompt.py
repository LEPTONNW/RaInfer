"""Format a local Gemma chat template with one mtmd media marker; no downloads."""
# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if '<__media__>' in args.prompt:
        parser.error('Supply text only; the helper adds the image marker')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    formatted = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '<__media__>\n' + args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    args.output.write_text(formatted, encoding='utf-8')
    print('Wrote one-image prompt using local tokenizer assets')


if __name__ == '__main__':
    main()
