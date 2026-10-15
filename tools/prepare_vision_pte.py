"""Copy a qualified PTE and alias only its portable embedding operator name."""
# SPDX-License-Identifier: MIT
import argparse
import hashlib
import json
import mmap
import shutil
import struct
import sys
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema-dir', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--io-contract', type=Path, help='Matching original IO metadata; writes io-contract.vision.json')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or output.exists():
        parser.error('Output must be a new file separate from the original model')
    if sha(source) != args.expected_sha256:
        parser.error('Original PTE SHA256 differs')
    sys.path.insert(0, str(args.schema_dir))
    from executorch_flatbuffer.Program import Program
    original = b'quantized_decomposed::embedding_4bit'
    replacement = b'rainfer::embedding_4bit'
    edits = {}
    with source.open('rb') as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        program = Program.GetRootAsProgram(data, 0)
        for i in range(program.ExecutionPlanLength()):
            plan = program.ExecutionPlan(i)
            if plan.Name() != b'kv_forward':
                continue
            for j in range(plan.OperatorsLength()):
                op = plan.Operators(j)
                if op.Name() != original or op.Overload() != b'dtype_out':
                    continue
                pointer = op._tab.Pos + op._tab.Offset(4)
                start = pointer + struct.unpack_from('<I', data, pointer)[0]
                size = struct.unpack_from('<I', data, start)[0]
                assert data[start + 4:start + 4 + size] == original
                before = data[start:start + 4 + size + 1]
                after = struct.pack('<I', len(replacement)) + replacement + b'\0' * (size - len(replacement) + 1)
                assert len(before) == len(after)
                edits[start] = (before, after)
    if len(edits) != 1:
        raise RuntimeError('Expected exactly one shared portable embedding operator descriptor')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    with output.open('r+b') as stream:
        for offset, (_, after) in edits.items():
            stream.seek(offset); stream.write(after)
    # Verify that every other byte, including all weights and QNN contexts, is unchanged.
    with source.open('rb') as left, output.open('rb') as right:
        position = 0
        while a := left.read(8 * 1024 * 1024):
            b = right.read(len(a))
            expected = bytearray(a)
            for offset, (_, after) in edits.items():
                lo, hi = max(position, offset), min(position + len(a), offset + len(after))
                if lo < hi:
                    expected[lo - position:hi - position] = after[lo - offset:hi - offset]
            if b != expected:
                raise RuntimeError('Unexpected change outside the operator-name alias')
            position += len(a)
        if right.read(1):
            raise RuntimeError('Unexpected output size change')
    report = {'source_sha256': args.expected_sha256, 'vision_pte_sha256': sha(output),
              'bytes': output.stat().st_size, 'weights_and_QNN_contexts_unchanged': True,
              'original_operator': original.decode() + '.dtype_out',
              'vision_operator': replacement.decode() + '.dtype_out',
              'edited_offsets': list(edits), 'device_validated': False,
              'requires_RaInfer_vision_embedding_kernel': True}
    output.with_suffix('.vision.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.io_contract:
        contract=json.loads(args.io_contract.read_text(encoding='utf-8'))
        contract['vision']={'operator':report['vision_operator'],'model_sha256':report['vision_pte_sha256'],
                            'hidden_size':1536,'image_placeholder_id':258880,'embedding_semantics':'projected-unscaled',
                            'device_validated':False}
        (output.parent/'io-contract.vision.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
