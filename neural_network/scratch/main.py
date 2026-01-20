import random
from value import Value
from engine import SGD, MSELoss, Module, Neuron, Layer, MLP


# Train on y = 2x + 1
xs = [[Value(1.0)], [Value(2.0)], [Value(3.0)], [Value(4.0)]]
ys = [Value(3.0), Value(5.0), Value(7.0), Value(9.0)]

mlp=MLP(1,[4,1])

optimizer = SGD(mlp.parameters(), lr=0.01)
criterion =MSELoss()

for epoch in range(1000):
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
print("\nNew data (y=2x+1):")
test_inputs=[5.0,6.0,10.0,0.00,100.0,11.0,3.4,8.1]

for x_val in test_inputs:
    x=[Value(x_val)]
    pred=mlp(x)
    actual=2*x_val+1
    print(f"x={x_val:.1f} -> predicted={pred.data:.2f}, actual={actual:.2f}")
