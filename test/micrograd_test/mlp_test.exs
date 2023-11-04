defmodule Micrograd.MLPTest do
  use ExUnit.Case

  alias Micrograd.{MLP, Value}

  test "train and predict: XOR" do
    xs = [
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0]
    ]

    ys = [0.0, 1.0, 1.0, 0.0]

    mlp = MLP.new(2, [{4, :tanh}, {4, :sigmoid}, {1, :sigmoid}])

    mlp =
      Enum.reduce(1..2000, mlp, fn _, mlp ->
        ypred = Enum.map(xs, &MLP.forward(mlp, &1))

        loss =
          Enum.zip(ys, ypred)
          |> Enum.map(fn {ygt, yout} -> Value.sub(yout, ygt) |> Value.pow(2.0) end)
          |> Value.sum()

        loss = Value.backward(loss)

        MLP.update(mlp, 0.1, loss)
      end)

    assert_in_delta MLP.forward(mlp, [0.0, 0.0]) |> Value.item(), 0.0, 0.08
    assert_in_delta MLP.forward(mlp, [0.0, 1.0]) |> Value.item(), 1.0, 0.08
    assert_in_delta MLP.forward(mlp, [1.0, 0.0]) |> Value.item(), 1.0, 0.08
    assert_in_delta MLP.forward(mlp, [1.0, 1.0]) |> Value.item(), 0.0, 0.08
  end

  test "train and predict" do
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP.new(3, [{4, :tanh}, {4, :tanh}, {1, :tanh}])

    mlp =
      Enum.reduce(1..50, mlp, fn _, mlp ->
        ypred = Enum.map(xs, &MLP.forward(mlp, &1))

        loss =
          Enum.zip(ys, ypred)
          |> Enum.map(fn {ygt, yout} -> Value.sub(yout, ygt) |> Value.pow(2.0) end)
          |> Value.sum()

        loss = Value.backward(loss)

        MLP.update(mlp, 0.1, loss)
      end)

    assert_in_delta MLP.forward(mlp, [2.0, 3.0, -1.0]) |> Value.item(), 1.0, 0.08
    assert_in_delta MLP.forward(mlp, [3.0, -1.0, 0.5]) |> Value.item(), -1.0, 0.08
    assert_in_delta MLP.forward(mlp, [0.5, 1.0, 1.0]) |> Value.item(), -1.0, 0.08
    assert_in_delta MLP.forward(mlp, [1.0, 1.0, -1.0]) |> Value.item(), 1.0, 0.08
  end
end
