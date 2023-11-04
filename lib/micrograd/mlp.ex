defmodule Micrograd.MLP do
  @doc """
  Multilayer perceptron.
  """

  alias Micrograd.Layer
  alias Micrograd.Value

  defstruct layers: nil

  def new(nin, nouts) when is_integer(nin) and is_list(nouts) do
    sz = [nin | nouts]

    layers =
      Enum.map(0..(length(nouts) - 1), fn i ->
        {ninputs, _} = Enum.at(sz, i) |> get_output_conf()
        {noutputs, activation} = Enum.at(sz, i + 1) |> get_output_conf()
        Layer.new(ninputs, noutputs, activation)
      end)

    %__MODULE__{layers: layers}
  end

  def forward(%__MODULE__{layers: layers}, inputs) when is_list(inputs) do
    Enum.reduce(layers, inputs, fn layer, acc -> Layer.forward(layer, acc) end)
  end

  def update(%__MODULE__{layers: layers}, learning_rate, %Value{} = loss) do
    gradients = loss |> collect_gradients()

    layers =
      Enum.map(layers, fn layer ->
        Layer.update(layer, learning_rate, gradients)
      end)

    %__MODULE__{layers: layers}
  end

  defp collect_gradients(value, gradients \\ %{})

  defp collect_gradients(%Value{ref: nil, parents: parents}, gradients) do
    Enum.reduce(parents, gradients, &collect_gradients/2)
  end

  defp collect_gradients(%Value{ref: ref, grad: grad, parents: parents}, gradients) do
    gradients = Map.put(gradients, ref, grad)
    Enum.reduce(parents, gradients, &collect_gradients/2)
  end

  defp collect_gradients(_, gradients), do: gradients

  defp get_output_conf(nout) when is_integer(nout), do: {nout, :relu}
  defp get_output_conf({nout, activation}), do: {nout, activation}
end
