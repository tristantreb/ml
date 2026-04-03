"""
From https://michaelwornow.net/2024/01/18/counting-params-in-transformer
"""

from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")


def count_params(model, is_human: bool = False):
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{params / 1e6:.2f}M" if is_human else params


print(model)
print("Total # of params:", count_params(model, is_human=True))
