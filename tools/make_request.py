"""Create one JSONL worker request using local Gemma tokenizer assets or IDs."""
# SPDX-License-Identifier: MIT
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tokens-file', type=Path, help='Whitespace-separated token IDs')
    source.add_argument('--tokenizer-dir', type=Path, help='Local HF tokenizer directory; requires transformers')
    parser.add_argument('--prompt', help='User text for the local tokenizer chat template')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--ngram', action='store_true', help='Enable the app baseline n-gram option')
    args = parser.parse_args()
    if args.tokenizer_dir:
        if args.prompt is None:
            parser.error('--tokenizer-dir requires --prompt')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
        tokens = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
    else:
        if args.prompt is not None:
            parser.error('--prompt is only used with --tokenizer-dir')
        tokens = [int(value) for value in args.tokens_file.read_text(encoding='utf-8').split()]
    if not tokens or len(tokens) > 2044 or any(t < 0 or t >= 262144 for t in tokens):
        parser.error('Expected 1..2044 prompt IDs in the Gemma vocabulary')
    if args.id < 0 or args.id >= 2**64 or not 1 <= args.max_new_tokens <= 2048:
        parser.error('Invalid request ID or output budget')
    request = dict(id=args.id, tokens=tokens, max_new_tokens=args.max_new_tokens,
                   cpu_workers=4, body_workers=1, fast_argmax=True,
                   compact_logits=True, full_vocabulary=False, ngram=args.ngram,
                   gpu_head=False, burst=False, fixed_affinity=False, profile=False)
    args.output.write_text(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Wrote {len(tokens)} prompt tokens to {args.output}')


if __name__ == '__main__':
    main()
