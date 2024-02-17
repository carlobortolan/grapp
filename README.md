# GRAPP

> [!IMPORTANT]
> As of February 18, 2024, this project has been put on hold due to lack of time. While the code partially works, the project will probably not be worked on or finished in the near future.

A simple Autograd engine written in _modern_ cpp that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library with a PyTorch-like API. The DAG only operates over scalar values. However, this is enough to build up entire deep neural nets doing binary classification.

> **DISCLAIMER**: _This is inspired by [pytorch's autograd engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine) as well as [micrograd](https://github.com/karpathy/micrograd) and used for experimenting with modern cpp and the concept of autograd engines and neural nets._

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
