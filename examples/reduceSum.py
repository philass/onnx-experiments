import torch
import torch.nn as nn

class ReduceSum(nn.Module):
    def __init__(self,):
        super(ReduceSum, self).__init__()

    def forward(self, data: torch.Tensor):
        print(data)
        tr1 = torch.transpose(data, 0, 2)
        sum1 = torch.sum(input=tr1, dim=0, keepdim=True)
        tr2 = torch.transpose(sum1, 0, 2)
        return tr2

def main():
    model = ReduceSum()

    test_shape = (2, 3, 4)

    t_vals = torch.linspace(0, 23, steps=24)
    # Dummy input
    t_input = t_vals.reshape(test_shape)

    input_names = ["input_arr"]
    output_names = ["result_arr"]

    torch.onnx.export(model, t_input, "program.onnx", input_names=input_names, output_names=output_names)

main()
