# GRAPP

A simple Autograd engine written in _modern_ cpp that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library with a PyTorch-like API. The DAG only operates over scalar values. However, this is enough to build up entire deep neural nets doing binary classification.

> **DISCLAIMER**: _This is inspired by [pytorch's autograd engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine) as well as [micrograd](https://github.com/karpathy/micrograd) and used for experimenting with modern cpp and the concept of autograd engines and neural nets._

## Example usage

<!-- > __NOTE__: _Check out [demo.ipynb](demo.ipynb) for more detailed examples_ -->

Below is a slightly contrived example showing a number of possible supported operations:

```cpp
using Value

auto a = Value(-4.0);
auto b = Value(2.0);
auto c = a + b
auto d = a * b + math.pow(b, 3);
c += c + 1;
c += 1 + c + (-a);
d += d * 2 + (b + a).relu();
d += 3 * d + (b - a).relu();
auto e = c - d;
auto f = math.pow(e, 2):
auto g = f / 2.0;
g += 10.0 / f;
std::cout << g.data << std::endl; // prints 24.7041, the outcome of this forward pass
g.backward();
std::cout << a.grad << std::endl; // prints 138.8338, i.e. the numerical value of dg/da
std::cout << b.grad << std::endl; // prints 645.5773, i.e. the numerical value of dg/db
```

## Build project

To build the project:

First, cd into the `build` directory:

```bash
cd build
```

Then, link all files:

```bash
cmake ..
```

Finally, compile the project and run the demo:

```bash
cd v1 && make runv1 && ./runv1
```

## Running tests

To run the unit tests you will have to install [DocTest](https://github.com/doctest/doctest/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply run:

```bash
cd build/tests/v1 && make testv1 && ./testv1
```

## LICENSE

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

---

© Carlo Bortolan

> Carlo Bortolan &nbsp;&middot;&nbsp;
> GitHub [carlobortolan](https://github.com/carlobortolan) &nbsp;&middot;&nbsp;
> contact via [carlobortolan@gmail.com](carlobortolan@gmail.com)
