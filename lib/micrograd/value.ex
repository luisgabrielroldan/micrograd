defmodule Micrograd.Value do
  @doc """
  A struct that represents a value in the computational graph.
  """

  defstruct parents: [], op: nil, data: nil, grad: nil, ref: nil, label: nil

  alias __MODULE__, as: V

  import Kernel, except: [div: 2]

  def new(data, opts \\ []) do
    %V{
      parents: opts[:parents] || [],
      op: opts[:op],
      grad: 0.0,
      data: data,
      ref: make_ref(),
      label: opts[:label] || nil
    }
  end

  def sum([%V{} = a | rest]),
    do: Enum.reduce(rest, a, fn x, acc -> add(acc, x) end)

  def add(%V{} = a, %V{} = b), do: new(a.data + b.data, op: :add, parents: [a, b])
  def add(%V{} = a, b) when is_number(b), do: a |> add(new(b))
  def add(a, %V{} = b), do: a |> new() |> add(b)
  def add(a, b) when is_number(a) and is_number(b), do: new(a + b)

  def mul(%V{} = a, %V{} = b), do: new(a.data * b.data, op: :mul, parents: [a, b])
  def mul(%V{} = a, b) when is_number(b), do: a |> mul(new(b))
  def mul(a, %V{} = b), do: a |> new() |> mul(b)
  def mul(a, b) when is_number(a) and is_number(b), do: new(a * b)

  def pow(%V{} = a, b) when is_number(b), do: new(:math.pow(a.data, b), op: :pow, parents: [a, b])
  def pow(a, b) when is_number(a) and is_number(b), do: new(:math.pow(a, b))

  def exp(%V{} = a), do: new(:math.exp(a.data), op: :exp, parents: [a])
  def exp(a) when is_number(a), do: new(:math.exp(a))

  def sub(a, b), do: add(a, neg(b))

  def div(a, b), do: mul(a, pow(b, -1.0))

  def neg(a), do: mul(a, -1.0)

  def tanh(%V{} = a) do
    value = (:math.exp(2 * a.data) - 1.0) / (:math.exp(2 * a.data) + 1.0)
    new(value, op: :tanh, parents: [a])
  end

  def sigmoid(%V{} = a) do
    value = 1.0 / (1.0 + :math.exp(-a.data))
    new(value, op: :sigmoid, parents: [a])
  end

  def relu(%V{} = a) do
    value = if a.data > 0.0, do: a.data, else: 0.0
    new(value, op: :relu, parents: [a])
  end

  def item(%V{data: data}), do: data

  def backward(%V{} = value) do
    value = do_backward(%{value | grad: 1.0})
    gradients = collect_gradients(value)
    update_gradients(value, gradients)
  end

  defp update_gradients(%V{ref: ref} = value, gradients) do
    gradient = Map.fetch!(gradients, ref)
    parents = Enum.map(value.parents, &update_gradients(&1, gradients))
    %{value | parents: parents, grad: gradient}
  end

  defp update_gradients(value, _gradients), do: value

  defp do_backward(%V{op: :add, parents: [a, b], grad: grad} = value) do
    updated_a = aggregate_grad(a, grad) |> do_backward()
    updated_b = aggregate_grad(b, grad) |> do_backward()
    %{value | parents: [updated_a, updated_b]}
  end

  defp do_backward(%V{op: :mul, parents: [a, b], grad: grad} = value) do
    updated_a = aggregate_grad(a, grad * b.data) |> do_backward()
    updated_b = aggregate_grad(b, grad * a.data) |> do_backward()
    %{value | parents: [updated_a, updated_b]}
  end

  defp do_backward(%V{op: :pow, parents: [a, b], grad: grad} = value) when is_number(b) do
    updated_a = aggregate_grad(a, grad * b * :math.pow(a.data, b - 1.0))
    %{value | parents: [do_backward(updated_a), b]}
  end

  defp do_backward(%V{op: :exp, parents: [a], grad: grad} = value) do
    updated_a = aggregate_grad(a, grad * :math.exp(a.data))
    %{value | parents: [do_backward(updated_a)]}
  end

  defp do_backward(%V{op: :tanh, parents: [a], grad: grad} = value) do
    updated_a = aggregate_grad(a, grad * (1.0 - :math.pow(value.data, 2.0)))
    %{value | parents: [do_backward(updated_a)]}
  end

  defp do_backward(%V{op: :sigmoid, parents: [a], grad: grad} = value) do
    updated_a =
      aggregate_grad(a, grad * :math.exp(-a.data) / :math.pow(1.0 + :math.exp(-a.data), 2.0))

    %{value | parents: [do_backward(updated_a)]}
  end

  defp do_backward(%V{op: :relu, parents: [a], grad: grad} = value) do
    updated_a = aggregate_grad(a, grad * if(a.data > 0.0, do: 1.0, else: 0.0))
    %{value | parents: [do_backward(updated_a)]}
  end

  defp do_backward(%V{op: nil} = value), do: value

  defp collect_gradients(value, gradients \\ %{})

  defp collect_gradients(%V{ref: nil} = value, _gradients),
    do: raise("No ref found for #{inspect(value)}")

  defp collect_gradients(%V{ref: ref, grad: grad, parents: parents}, gradients) do
    gradients = Map.update(gradients, ref, grad, fn cg -> cg + grad end)
    Enum.reduce(parents, gradients, &collect_gradients/2)
  end

  defp collect_gradients(_, gradients), do: gradients

  defp aggregate_grad(%{grad: grad} = v, new_grad), do: %{v | grad: grad + new_grad}
  defp aggregate_grad(v, _new_grad), do: v
end
