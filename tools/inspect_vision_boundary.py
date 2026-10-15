"""Inspect CPU embedding boundaries in an existing PTE without loading weights."""
# SPDX-License-Identifier: MIT
import argparse
import json
import mmap
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema-dir', type=Path, required=True)
    parser.add_argument('--pte', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.schema_dir))
    from executorch_flatbuffer.Program import Program
    from executorch_flatbuffer.Tensor import Tensor
    from executorch_flatbuffer.KernelCall import KernelCall
    from executorch_flatbuffer.InstructionArguments import InstructionArguments
    from executorch_flatbuffer.KernelTypes import KernelTypes
    with args.pte.open('rb') as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        program = Program.GetRootAsProgram(data, 0)
        methods = []
        for method in range(program.ExecutionPlanLength()):
            plan = program.ExecutionPlan(method)
            operators = []
            for index in range(plan.OperatorsLength()):
                op = plan.Operators(index)
                operators.append({'name': op.Name().decode(), 'overload': op.Overload().decode()})
            def tensor_info(index):
                value = plan.Values(index)
                if value.ValType() != KernelTypes.Tensor:
                    return {'index': index, 'type': value.ValType()}
                table = value.Val(); tensor = Tensor(); tensor.Init(table.Bytes, table.Pos)
                return {'index': index, 'dtype': tensor.ScalarType(),
                        'shape': [tensor.Sizes(i) for i in range(tensor.SizesLength())]}
            calls = []
            for chain_index in range(plan.ChainsLength()):
                chain = plan.Chains(chain_index)
                for index in range(chain.InstructionsLength()):
                    instruction = chain.Instructions(index)
                    if instruction.InstrArgsType() == InstructionArguments.KernelCall:
                        table = instruction.InstrArgs(); call = KernelCall(); call.Init(table.Bytes, table.Pos)
                        op = operators[call.OpIndex()]
                        calls.append({'instruction': index, **op,
                                      'args': [tensor_info(call.Args(i)) for i in range(call.ArgsLength())]})
            methods.append({'method': plan.Name().decode(), 'operators': operators, 'cpu_calls': calls})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'methods': methods}, indent=2) + '\n', encoding='utf-8')
        for method in methods:
            print(json.dumps({'method': method['method'], 'operators': method['operators'],
                              'embedding_calls': [c for c in method['cpu_calls'] if 'embedding' in c['name']]}))


if __name__ == '__main__':
    main()
