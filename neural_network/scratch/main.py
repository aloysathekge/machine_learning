import random
from value import Value
from engine import Module, Neuron, Layer, MLP

x=[Value(2.0),Value(3.0),Value(-1.0)]
# layer=Layer(3, 4)

mlp=MLP(3,[4,4,1])
print(mlp)

out=mlp(x)
print(f"mlp output:{out.data}")
print(f"Total parameters: {len(mlp.parameters())}")

# Backward pass
out.backward()
print(f"First parameter gradient: {mlp.parameters()[0].grad:.4f}")
# Zero grad
mlp.zero_grad()
print(f"After zero_grad: {mlp.parameters()[0].grad}")

# 4. Compute loss (example: want output to be 1.0)
target = Value(1.0)
loss = (out - target) ** 2  # Mean squared error
print(loss.data)

loss.backward()

for p in mlp.parameters():
    p.data-=0.01 *p.grad

mlp.zero_grad()