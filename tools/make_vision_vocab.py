"""Extract exact model/tokenizer metadata into a weightless vision GGUF."""
# SPDX-License-Identifier: MIT
import argparse
import struct
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf-python', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        parser.error('Output must be a new, separate file')
    sys.path.insert(0, str(args.gguf_python))
    from gguf import GGUFReader
    reader = GGUFReader(str(args.source))
    assert reader.fields['general.architecture'].contents() == 'gemma4'
    assert reader.fields['gemma4.embedding_length'].contents() == 1536
    end = max(field.offset + sum(part.nbytes for part in field.parts) for field in reader.fields.values())
    with args.source.open('rb') as stream:
        metadata = bytearray(stream.read(end))
    assert metadata[:4] == b'GGUF' and struct.unpack_from('<I', metadata, 4)[0] == 3
    struct.pack_into('<Q', metadata, 8, 0)  # zero tensors; preserve every key/value byte
    metadata.extend(b'\0' * (-len(metadata) % 32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(metadata)
    checked = GGUFReader(str(args.output))
    assert len(checked.tensors) == 0
    assert checked.fields['general.architecture'].contents() == 'gemma4'
    assert checked.fields['gemma4.embedding_length'].contents() == 1536
    print(f'Wrote {len(metadata)} bytes of Gemma4 metadata/vocabulary with zero model tensors')


if __name__ == '__main__':
    main()
