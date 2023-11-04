# Elixir Micrograd

A little (and slow) neural network library that implements backpropagation using automatic differentiation.

This is based on [Micrograd](https://github.com/karpathy/micrograd) by [Andrej Karpathy](https://github.com/karpathy).

I made this projects as an EXPERIMENT to learn more about Neural Networks and Elixir. The implementation on Elixir was a bit
tricky because of the lack of operator overloading and the immutability of the language. There is maybe a better approach to
solve this problem, but I learned a lot in the process. 

## Automatic gradients example

```elixir
x1 = Value.new(2.0)
w1 = Value.new(-3.0)
x2 = Value.new(0.0)
w2 = Value.new(1.0)

x1w1 = Value.mul(x1, w1)
x2w2 = Value.mul(x2, w2)

x1w1x2w2 = Value.add(x1w1, x2w2)

n = Value.add(x1w1x2w2, 4.0)

o = Value.tanh(n)

# The Value struct keeps track of the operations that were performed

IO.puts "o: #{Value.item(o)}"
# o: -0.9640275800758168

# The backward function will calculate the gradients of the operations
# and store them in the Value struct.

o = Value.backward(o)

# %Value{
#   parents: [
#     %Value{
#       parents: [
#         %Value{
#           parents: [
#             %Value{
#               parents: [
#                 %Value{ data: 2.0, grad: -3.0, },
#                 %Value{ data: -3.0, grad: 2.0, }
#               ],
#               data: -6.0,
#               grad: 1.0,
#             },
#             %Value{
#               parents: [
#                 %Value{ data: 0.0, grad: 1.0, },
#                 %Value{ data: 1.0, grad: 0.0, }
#               ],
#               ...
# }

```

## Create a neural network

```elixir

epochs = 2000
learning_rate = 0.08

# XOR Inputs
xs = [
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0]
]

# XOR Results
ys = [0.0, 1.0, 1.0, 0.0]

# Create a multilayer perceptron
mlp = MLP.new(2, [{4, :tanh}, {4, :sigmoid}, {1, :sigmoid}])

# Train
mlp =
  Enum.reduce(1..epochs, mlp, fn _, mlp ->
  
    # Forward pass
    ypred = Enum.map(xs, &MLP.forward(mlp, &1))

    # Calculate loss
    loss =
      Enum.zip(ys, ypred)
      |> Enum.map(fn {ygt, yout} -> Value.sub(yout, ygt) |> Value.pow(2.0) end)
      |> Value.sum()

    # Backward pass
    loss = Value.backward(loss)

    # Update weights
    MLP.update(mlp, learning_rate, loss)
  end)
  
# Predict
MLP.forward(mlp, [0.0, 1.0]) |> Value.item() # ~= 1.0
```
