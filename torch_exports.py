import requests
import sys

# This script finds the torch functions, that export to a given onnx op
#
# Usage
# 
# python3 torch_test_ops.py ONNXOpName
#
# Example
#
# python3 torch_test_ops.py ReduceLogSum
#

if len(sys.argv) < 2:
    print("Please provide the name of an onnx op")
    exit()

onnx_op_name = sys.argv[1]
print(f"Searching torch for export to onnx_op_name : {onnx_op_name}")

opset_numbers = [i for i in range(7, 16)]
candidate_file_names = [f"symbolic_opset{i}" for i in opset_numbers]
candidate_file_names.append("symbolic_helper") # Add helper file that also has export logic

source_directory_url = "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/onnx/"
file_extension = ".py"

for file in candidate_file_names:
    print(f"Checking {file}.py")
    file_path = f"{source_directory_url}{file}{file_extension}"
    file_text = requests.get(file_path).text
    lines = file_text.splitlines()
    exports = False
    for (line_num, line) in enumerate(lines):
        if f"\"{onnx_op_name}\"" in line:
            exports = True
            print(f"{line_num} : {line}")
