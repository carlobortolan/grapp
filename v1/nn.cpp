#include "nn.h"

#include <algorithm>

namespace grapp {

Neuron::Neuron(size_t nin, bool nonlin) : nonlin(nonlin) {
    // Initialize weights randomly between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < nin; ++i) {
        weights.push_back(std::make_shared<Value>(dis(gen)));
    }

    bias = std::make_shared<Value>(0);
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& inputs) const {
    // Calculate the weighted sum
    auto act = std::accumulate(inputs.begin(), inputs.end(), *bias,
                                [&](const Value& accum, const std::shared_ptr<Value>& input) {
                                    return accum + *input * *weights[&input - &inputs[0]];
                                });

    // Apply non-linear activation function if specified
    return nonlin ? std::make_shared<Value>(act.relu()) : std::make_shared<Value>(act);
}

void Neuron::zero_grad() {
    std::for_each(weights.begin(), weights.end(), [](const std::shared_ptr<Value>& w) { w->grad = 0; });
    bias->grad = 0;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    params.reserve(weights.size() + 1);
    params.insert(params.end(), weights.begin(), weights.end());
    params.push_back(bias);
    return params;
}

Layer::Layer(size_t nin, size_t nout, bool nonlin) {
    neurons.reserve(nout);
    for (size_t i = 0; i < nout; ++i) {
        neurons.emplace_back(nin, nonlin);
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& inputs) const {
    std::vector<std::shared_ptr<Value>> outputs;
    outputs.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        outputs.push_back(neuron(inputs));
    }
    return outputs;
}

void Layer::zero_grad() {
    std::for_each(neurons.begin(), neurons.end(), [](Neuron& n) { n.zero_grad(); });
}

std::vector<std::shared_ptr<Value>> Layer::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    for (const auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

MLP::MLP(size_t nin, const std::vector<size_t>& nouts) {
    layers.reserve(nouts.size());
    size_t prev_nout = nin;
    for (size_t nout : nouts) {
        layers.emplace_back(prev_nout, nout);
        prev_nout = nout;
    }
}

std::shared_ptr<Value> MLP::operator()(const std::vector<std::shared_ptr<Value>>& inputs) const {
    auto x = inputs;
    for (const auto& layer : layers) {
        x = layer(x);
    }
    return x[0];
}

void MLP::zero_grad() {
    std::for_each(layers.begin(), layers.end(), [](Layer& layer) { layer.zero_grad(); });
}

std::vector<std::shared_ptr<Value>> MLP::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    for (const auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

} // namespace grapp
