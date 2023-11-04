defmodule Micrograd.ValueTest do
  use ExUnit.Case

  alias Micrograd.Value

  test "happy path" do
    x1 = Value.new(2.0, label: :x1)
    w1 = Value.new(-3.0, label: :w1)
    x2 = Value.new(0.0, label: :x2)
    w2 = Value.new(1.0, label: :w2)

    x1w1 = %{Value.mul(x1, w1) | label: :x1w1}
    x2w2 = %{Value.mul(x2, w2) | label: :x2w2}
    x1w1x2w2 = %{Value.add(x1w1, x2w2) | label: :x1w1x2w2}

    b = Value.new(6.0, label: :b)

    n = %{Value.add(x1w1x2w2, b) | label: :n}

    o = %{Value.tanh(n) | label: :o}

    v = %{Value.mul(o, x1) | label: :v}

    v = Value.backward(v)

    assert %Value{
             parents: [
               %Value{
                 parents: [
                   %Value{
                     parents: [
                       %Value{
                         parents: [
                           %Value{
                             parents: [
                               %Value{parents: [], data: 2.0, label: :x1},
                               %Value{parents: [], data: -3.0, label: :w1}
                             ],
                             op: :mul,
                             data: -6.0,
                             label: :x1w1
                           },
                           %Value{
                             parents: [
                               %Value{parents: [], data: 0.0, label: :x2},
                               %Value{parents: [], data: 1.0, label: :w2}
                             ],
                             op: :mul,
                             data: 0.0,
                             label: :x2w2
                           }
                         ],
                         op: :add,
                         data: -6.0,
                         label: :x1w1x2w2
                       },
                       %Value{parents: [], data: 6.0, label: :b}
                     ],
                     op: :add,
                     data: 0.0,
                     label: :n
                   }
                 ],
                 op: :tanh,
                 data: 0.0,
                 label: :o
               },
               %Value{parents: [], data: 2.0, label: :x1}
             ],
             op: :mul,
             data: 0.0,
             label: :v
           } = v
  end

  describe "Value.backward/1" do
    test "add" do
      a = Value.new(1.0)
      b = Value.new(2.0)
      c = Value.add(a, b)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 1.0},
                 %Value{data: 2.0, grad: 1.0}
               ],
               op: :add,
               data: 3.0,
               grad: 1.0
             } = Value.backward(c)
    end

    test "sub" do
      a = Value.new(1.0)
      b = Value.new(2.0)
      c = Value.sub(a, b)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 1.0},
                 %Value{data: -2.0, grad: 1.0}
               ],
               op: :add,
               data: -1.0,
               grad: 1.0
             } = Value.backward(c)
    end

    test "mul" do
      a = Value.new(1.0)
      b = Value.new(2.0)
      c = Value.mul(a, b)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 2.0},
                 %Value{data: 2.0, grad: 1.0}
               ],
               op: :mul,
               data: 2.0,
               grad: 1.0
             } = Value.backward(c)
    end

    test "pow" do
      a = Value.new(1.0)
      b = Value.pow(a, 2.0)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 2.0},
                 2.0
               ],
               op: :pow,
               data: 1.0,
               grad: 1.0
             } = Value.backward(b)
    end

    test "neg" do
      a = Value.new(1.0)
      b = Value.neg(a)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: -1.0},
                 %Value{data: -1.0, grad: 1.0}
               ],
               op: :mul,
               data: -1.0,
               grad: 1.0
             } = Value.backward(b)
    end

    test "exp" do
      a = Value.new(1.0)
      b = Value.exp(a)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 2.718281828459045}
               ],
               op: :exp,
               data: 2.718281828459045,
               grad: 1.0
             } = Value.backward(b)
    end

    test "tanh" do
      a = Value.new(1.0)
      b = Value.tanh(a)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 0.41997434161402614}
               ],
               op: :tanh,
               data: 0.7615941559557649,
               grad: 1.0
             } = Value.backward(b)
    end

    test "sigmoid" do
      a = Value.new(1.0)
      b = Value.sigmoid(a)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 0.19661193324148188}
               ],
               op: :sigmoid,
               data: 0.7310585786300049,
               grad: 1.0
             } = Value.backward(b)
    end

    test "relu" do
      a = Value.new(1.0)
      b = Value.relu(a)

      assert %Value{
               parents: [
                 %Value{data: 1.0, grad: 1.0}
               ],
               op: :relu,
               data: 1.0,
               grad: 1.0
             } = Value.backward(b)
    end

    test "gradients are accumulated" do
      a = Value.new(1.0)
      b = Value.new(2.0)
      c = Value.mul(a, b)
      d = Value.mul(a, c)

      assert %Value{
               parents: [
                 # a
                 %Value{data: 1.0, grad: 4.0},
                 %Value{
                   parents: [
                     # a
                     %Value{data: 1.0, grad: 4.0},
                     # b
                     %Value{data: 2.0, grad: 1.0}
                   ]
                 }
               ]
             } = Value.backward(d)
    end
  end
end
