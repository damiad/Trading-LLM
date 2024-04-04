import torch


def subtract_row_values(tensor, values):
    # Broadcast the array of values to match the shape of the tensor
    values = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    # values = values.unsqueeze(1).expand(-1, tensor.size(1))

    # Subtract the corresponding value from each row of the tensor
    result = tensor - values
    return result


# Example usage:
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
values = [1, 2]
result = subtract_row_values(tensor, values)
print("Result:")
print(result)
