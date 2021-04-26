import os
import sys

import tflite
# Cannot use latest PIP version because flexbuffers support is only on master branch
from _flatbuffers.python.flatbuffers import flexbuffers

def parse_custom_options(data):
    root = flexbuffers.GetRoot(data)
    return root.Value

if len(sys.argv) < 2:
    raise RuntimeError("Usage: `{}` [FILE].tflite".format(sys.argv[0]))

path = sys.argv[1]

with open(path, 'rb') as f:
    buf = f.read()
    model = tflite.Model.GetRootAsModel(buf, 0)

opcodes_len = model.OperatorCodesLength()
print("Total number of opcodes: ", opcodes_len)
subgraphs_len = model.SubgraphsLength()
for g in range(subgraphs_len):
    print("=== Graph {}/{} ===".format(g+1, subgraphs_len))
    graph = model.Subgraphs(g)
    ops_len = graph.OperatorsLength()
    for o in range(ops_len):
        op = graph.Operators(o)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        if op_code.BuiltinCode() != tflite.BuiltinOperator.CUSTOM:
            print("[{}/{}] Skipping non-custom operator...".format(o+1, ops_len))
            continue

        assert(op.CustomOptionsFormat() == tflite.CustomOptionsFormat.FLEXBUFFERS)
        print("[{}/{}] Found CUSTOM operator '{}'".format(o+1,ops_len,op_code.CustomCode().decode('utf-8')))
        custom_options = bytes([op.CustomOptions(i) for i in range(op.CustomOptionsLength())])
        print("custom_options:", parse_custom_options(custom_options))
    print("=== DONE ===")
