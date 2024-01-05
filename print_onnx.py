import sys
import onnx

text = onnx.printer.to_text(onnx.load(sys.argv[1]))
print(text)
