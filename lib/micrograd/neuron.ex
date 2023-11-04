defmodule Micrograd.Neuron do
  @doc """
  A neuron in a multilayer perceptron.
  """

  alias Micrograd.Value

  defstruct weights: nil, bias: nil, activation: nil

  def new(nin, activation) when is_integer(nin) do
    %__MODULE__{
      weights: Enum.map(1..nin, fn _ -> rnd() |> Value.new(ref: make_ref()) end),
      bias: rnd() |> Value.new(ref: make_ref()),
      activation: activation
    }
  end

  def forward(%__MODULE__{weights: weights, bias: bias, activation: activation}, inputs)
      when is_list(inputs) do
    if length(inputs) != length(weights) do
      raise ArgumentError, "inputs length must match weights length"
    end

    inputs
    |> Enum.zip(weights)
    |> Enum.map(fn {input, weight} -> Value.mul(input, weight) end)
    |> Enum.reduce(bias, fn x, acc -> Value.add(x, acc) end)
    |> apply_activation(activation)
  end

  def apply_activation(%Value{} = value, :linear), do: value
  def apply_activation(%Value{} = value, :tanh), do: Value.tanh(value)
  def apply_activation(%Value{} = value, :sigmoid), do: Value.sigmoid(value)
  def apply_activation(%Value{} = value, :relu), do: Value.relu(value)

  def update(%__MODULE__{weights: weights, bias: bias} = neuron, learning_rate, gradients) do
    weights =
      Enum.map(weights, fn %Value{data: data} = weight ->
        weight_grad = Map.fetch!(gradients, weight.ref)

        %{weight | data: data + -learning_rate * weight_grad}
      end)

    bias_grad = Map.fetch!(gradients, bias.ref)
    bias = %{bias | data: bias.data + -learning_rate * bias_grad}

    %__MODULE__{neuron | weights: weights, bias: bias}
  end

  defp rnd(), do: 1 - :rand.uniform_real() * 2
end
