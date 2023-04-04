# POC for export custom torch ops defined outside pytorch
# Example of custom op -> ScatterMax : https://github.com/rusty1s/pytorch_scatter/blob/master/csrc/scatter.cpp#L199
#
# code below follows from instructions here https://pytorch.org/docs/stable/onnx.html#c-operators


import torch
import torch.nn as nn
from torch.onnx import symbolic_helper
from torch_scatter import scatter_max

class ScatterMax(nn.Module):

    def forward(self, src: torch.Tensor, index: torch.Tensor):
        # src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        # index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out, argmax = scatter_max(src, index, dim=-1, out=src)
        return out, argmax


@symbolic_helper.parse_args("v", "v", "i", "v", "i", "i")
def symbolic_scatter_max(g, src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    return (index ,g.op("ScatterElements", out, index, src, axis_i=dim, reduction_s="max"))

torch.onnx.register_custom_op_symbolic("torch_scatter::scatter_max", symbolic_scatter_max, 18)

model = ScatterMax()

src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
input_names = ["input_src", "input_idx"]
output_names = ["result_arr1", "result_arr2"]

torch.onnx.export(model, (src, index), "program.onnx", input_names=input_names, output_names=output_names, opset_version=18)
