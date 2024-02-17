#include "nn.h"
#include "engine.h" // Assuming Value class is defined in engine.h

#include <iostream>
#include <vector>
#include <random>
#include <cmath> // Include cmath for pow function

using namespace grapp;

int main() {
    // Define the data
    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    // Define the targets
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    // Create the MLP
    MLP n(3, {4, 4, 1});

    // Gradient descent
    for (int k = 0; k < 10000; ++k) {
        // Forward pass
        std::vector<Value> loss_values;
        for (size_t i = 0; i < xs.size(); ++i) {
            auto x = xs[i];
            std::vector<std::shared_ptr<Value>> x_values;
            for (auto val : x) {
                x_values.push_back(std::make_shared<Value>(val));
            }
            auto yout = *n(x_values);
            auto ygt = Value(ys[i]);
            auto loss = (yout - ygt)^2;
            loss_values.push_back(loss);
        }

        // Compute total loss
        auto total_loss = std::accumulate(loss_values.begin(), loss_values.end(), Value(0.0),
                                          [](const Value& a, const Value& b) { return a + b; });

        // Backward pass
        n.zero_grad();
        total_loss.backward();

        // Update parameters using gradient descent with learning rate schedule
        double learning_rate = 0.4; // Learning rate schedule
        for (auto& p : n.parameters()) {
            p->data -= learning_rate * p->grad;
        }

        std::cout << "Iteration: " << k << ", Loss: " << total_loss.data << std::endl;

        // Print predictions after each iteration
        std::cout << "Predictions: ";
        for (const auto& x : xs) {
            std::vector<std::shared_ptr<Value>> x_values;
            for (auto val : x) {
                x_values.push_back(std::make_shared<Value>(val));
            }
            auto yout = *n(x_values);
            std::cout << yout.data << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}