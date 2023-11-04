defmodule Micrograd.Layer do
  @doc """
  A layer in a multilayer perceptron.
  """

  alias Micrograd.{Layer, Neuron}

  defstruct neurons: nil

  def new(nin, nout, activation) when is_integer(nin) and is_integer(nout) do
    %__MODULE__{
      neurons: Enum.map(1..nout, fn _ -> Neuron.new(nin, activation) end)
    }
  end

  def forward(%__MODULE__{neurons: neurons}, inputs) when is_list(inputs) do
    neurons
    |> Enum.map(&Neuron.forward(&1, inputs))
    |> case do
      [neuron] -> neuron
      neurons -> neurons
    end
  end

  def update(%__MODULE__{neurons: neurons}, learning_rate, gradients) do
    neurons =
      Enum.map(neurons, fn neuron ->
        Neuron.update(neuron, learning_rate, gradients)
      end)

    %Layer{neurons: neurons}
  end
end
