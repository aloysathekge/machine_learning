import random
from value import Value
from engine import SGD, MSELoss, Module, Neuron, Layer, MLP


xs = [
    [Value(-2.0)], [Value(-1.8)], [Value(-1.6)], [Value(-1.4)], [Value(-1.2)],
    [Value(-1.0)], [Value(-0.8)], [Value(-0.6)], [Value(-0.4)], [Value(-0.2)],
    [Value(0.0)],
    [Value(0.2)], [Value(0.4)], [Value(0.6)], [Value(0.8)], [Value(1.0)],
    [Value(1.2)], [Value(1.4)], [Value(1.6)], [Value(1.8)], [Value(2.0)]
]

ys = [
    Value(4.00), Value(3.24), Value(2.56), Value(1.96), Value(1.44),
    Value(1.00), Value(0.64), Value(0.36), Value(0.16), Value(0.04),
    Value(0.00),
    Value(0.04), Value(0.16), Value(0.36), Value(0.64), Value(1.00),
    Value(1.44), Value(1.96), Value(2.56), Value(3.24), Value(4.00)
]

mlp=MLP(1,[10,1])

optimizer = SGD(mlp.parameters(), lr=0.01)
criterion =MSELoss()

for epoch in range(5000):
    optimizer.zero_grad()
    #forward
    preds=[mlp(x) for x in xs]
    loss=criterion(preds,ys)


    #backward
    loss.backward()

    optimizer.step()

    if epoch %10==0:
        print(f"Epoch {epoch} , loss :{loss.data:.4f}")

print("\n--- Predictions ---")

for x,y in zip(xs,ys):
    pred=mlp(x)
    print(f"x={x[0].data:.1f} -> predicted={pred.data:.2f}, actual={y.data:.2f}")


#Test on new data (model never saw these!)

test_inputs=[5.0,6.0,10.0,0.00,100.0,11.0,3.4,8.1]

for x_val in test_inputs:
    x=[Value(x_val)]
    pred=mlp(x)
    actual=x_val**2
    print(f"x={x_val:.1f} -> predicted={pred.data:.2f}, actual={actual:.2f}")
