"""Compare recorded target-only/speculative results; never runs a device test."""
# SPDX-License-Identifier: MIT
import argparse
import array
import json
import sys
from pathlib import Path


def read_report(path):
    text = path.read_text(encoding='utf-8-sig')
    records = [json.loads(line[len('@DFLASH2 '):]) for line in text.splitlines()
               if line.startswith('@DFLASH2 ')]
    if records:
        results = [item['report'] for item in records if item.get('event') == 'result']
        if len(results) != 1:
            raise ValueError('Expected one result per comparison file')
        return results[0]
    value = json.loads(text)
    return value.get('report', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline-logits', type=Path)
    parser.add_argument('--candidate-logits', type=Path)
    parser.add_argument('--expected-image-rows', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if bool(args.baseline_logits) != bool(args.candidate_logits):
        parser.error('Supply both native U16 logit dumps or neither')
    a, b = read_report(args.baseline), read_report(args.candidate)
    if not a.get('tokens') or not b.get('tokens'):
        raise ValueError('Missing generated output tokens')
    checks = {'output_tokens_equal': a['tokens'] == b['tokens'],
              'prompt_count_equal': a['prompt_tokens'] == b['prompt_tokens']}
    if 'predictions' in a and 'predictions' in b:
        checks['target_predictions_equal'] = a['predictions'] == b['predictions']
    if args.expected_image_rows:
        for name, report in [('baseline', a), ('candidate', b)]:
            vision = report.get('vision', {})
            checks[name + '_image_rows_reached_both_boundaries'] = all(
                vision.get(key) == args.expected_image_rows for key in
                ('image_tokens', 'primary_rows_injected', 'ple_rows_padded'))
    logits = None
    if args.baseline_logits:
        aa, bb = (path.read_bytes() for path in (args.baseline_logits, args.candidate_logits))
        if len(aa) != 4 * 262144 * 2 or len(bb) != len(aa):
            raise ValueError('Expected matching AR4 x 262144 native U16 logits')
        left, right = array.array('H'), array.array('H')
        left.frombytes(aa); right.frombytes(bb)
        if sys.byteorder != 'little':
            left.byteswap(); right.byteswap()
        errors = [abs(int(x) - int(y)) for x, y in zip(left, right)]
        logits = {'dtype': 'uint16-le', 'units': 'native quantized codes; not FP32 logits',
                  'max_absolute_error': max(errors), 'mean_absolute_error': sum(errors) / len(errors),
                  'differing_values': sum(error != 0 for error in errors)}
        checks['native_logits_identical'] = aa == bb
    metrics = []
    for name, report in [('baseline', a), ('candidate', b)]:
        proposed = report.get('proposed_draft_tokens', 0)
        metrics.append({'case': name, 'prefill_seconds': report.get('prefill_seconds'),
                        'decode_seconds': report.get('decode_seconds'),
                        'native_decode_tokens_per_second': report.get('decode_tokens_per_second'),
                        'native_tps_numerator': 'generated tokens after the first prefill-produced token',
                        'acceptance_fraction': report.get('accepted_draft_tokens', 0) / proposed if proposed else None})
    result = {'passed': all(checks.values()), 'checks': checks, 'native_logits': logits,
              'measured_reports': metrics, 'not_a_broad_quality_certificate': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
