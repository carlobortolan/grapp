#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <random>

// Assuming Value class is defined in engine.h
#include "engine.h"

namespace grapp {

// Forward declaration of classes
class Module;
class Neuron;
class Layer;
class MLP;

/**
 * @brief Abstract base class for all neural network modules.
 */
class Module {
public:
    /**
     * @brief Reset gradients of all parameters.
     */
    virtual void zero_grad() = 0;
    
    /**
     * @brief Get parameters of the module.
     * @return Vector of shared pointers to Value objects representing parameters.
     */
    virtual std::vector<std::shared_ptr<Value>> parameters() const = 0;
};

/**
 * @brief Single neuron module.
 */
class Neuron : public Module {
public:
    std::vector<std::shared_ptr<Value>> weights; // Weights of the neuron
    std::shared_ptr<Value> bias; // Bias of the neuron
    bool nonlin; // Whether the neuron has a non-linear activation function

    /**
     * @brief Constructor.
     * @param nin Number of input neurons.
     * @param nonlin Whether the neuron has a non-linear activation function (default true).
     */
    explicit Neuron(size_t nin, bool nonlin = true);
    
    /**
     * @brief Forward pass through the neuron.
     * @param inputs Vector of shared pointers to Value objects representing inputs.
     * @return Shared pointer to Value object representing the output of the neuron.
     */
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) const;
    
    /**
     * @brief Reset gradients of all parameters.
     */
    void zero_grad() override;
    
    /**
     * @brief Get parameters of the neuron.
     * @return Vector of shared pointers to Value objects representing parameters.
     */
    std::vector<std::shared_ptr<Value>> parameters() const override;
};

/**
 * @brief Layer of neurons module.
 */
class Layer : public Module {
public:
    std::vector<Neuron> neurons; // Neurons in the layer

    /**
     * @brief Constructor.
     * @param nin Number of input neurons.
     * @param nout Number of output neurons.
     * @param nonlin Whether the neurons have a non-linear activation function (default true).
     */
    explicit Layer(size_t nin, size_t nout, bool nonlin = true);
    
    /**
     * @brief Forward pass through the layer.
     * @param inputs Vector of shared pointers to Value objects representing inputs.
     * @return Vector of shared pointers to Value objects representing outputs.
     */
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) const;
    
    /**
     * @brief Reset gradients of all parameters.
     */
    void zero_grad() override;
    
    /**
     * @brief Get parameters of the layer.
     * @return Vector of shared pointers to Value objects representing parameters.
     */
    std::vector<std::shared_ptr<Value>> parameters() const override;
};

/**
 * @brief Multi-layer perceptron module.
 */
class MLP : public Module {
public:
    std::vector<Layer> layers; // Layers of neurons in the MLP

    /**
     * @brief Constructor.
     * @param nin Number of input neurons.
     * @param nouts Vector containing the number of output neurons for each layer.
     */
    explicit MLP(size_t nin, const std::vector<size_t>& nouts);
    
    /**
     * @brief Forward pass through the MLP.
     * @param inputs Vector of shared pointers to Value objects representing inputs.
     * @return Shared pointer to Value object representing the output of the MLP.
     */
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) const;
    
    /**
     * @brief Reset gradients of all parameters.
     */
    void zero_grad() override;
    
    /**
     * @brief Get parameters of the MLP.
     * @return Vector of shared pointers to Value objects representing parameters.
     */
    std::vector<std::shared_ptr<Value>> parameters() const override;
};

} // namespace grapp
