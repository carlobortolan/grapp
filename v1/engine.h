#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace grapp {

/**
 * @brief Represents a value in the computation graph.
 *
 * The Value class is a wrapper around a scalar value used to represent the input and output of operations.
 */
class Value {
public:
    /**
     * @brief Default constructor.
     */
    Value() = default;

    /**
     * @brief Construct a value with given data.
     *
     * @param data The scalar value.
     */
    explicit Value(double data);

    /**
     * @brief Construct a value with given data and operation.
     *
     * @param data The scalar value.
     * @param op The operation associated with this node.
     */
    Value(double data, std::string_view op);

    /**
     * @brief Construct a value with given data, children, and operation.
     *
     * @param data The scalar value.
     * @param children The children nodes in the computation graph.
     * @param op The operation associated with this node.
     */
    Value(double data, std::vector<Value*> children, std::string_view op);

    /**
     * @brief Construct a value with given data, children, and operation using an initializer list.
     *
     * @param data The scalar value.
     * @param children The children nodes in the computation graph.
     * @param op The operation associated with this node.
     */
    Value(double data, std::initializer_list<Value*> children, std::string_view op);

    /**
     * @brief Assign a new value to this Value object.
     *
     * @param data The new scalar value.
     * @return Reference to the modified Value object.
     */
    auto operator=(double data) -> Value&;

    /**
     * @brief relu activation function
     * This function computes the relu activation function of the value.
     * @return Value object with the relu activation function of the value.
    */
    auto relu() -> Value&;

    /**
     * @brief Compute the gradient of the value.
     *
     * This function computes the gradient of the value using backpropagation.
     */
    auto backward() -> void;

    /* In place operators, modify the given Vector in-place, rather than a copy */

    auto operator+=(double val) -> Value &;
    auto operator-=(double val) -> Value &;
    auto operator*=(double val) -> Value &;
    auto operator/=(double val) -> Value &;
   
    auto operator+=(const Value &y) -> Value &;
    auto operator-=(const Value &y) -> Value &;

    auto visualizeGraph() -> void;

// private:
    double data;
    mutable double grad = 0;
    std::function<void()> _backward = []() {};
    std::vector<Value*> _prev;
    std::string_view _op;
    bool visited = false;
};

// Unary operations (op [Value])

auto operator+(const Value& val) -> Value;
auto operator-(const Value& val) -> Value;

// Binary operations ([Value] op [Value])

auto operator+(const Value& valx, const Value& valy) -> Value;
auto operator-(const Value& valx, const Value& valy) -> Value;
auto operator*(const Value& valx, const Value& valy) -> Value;
auto operator/(const Value& valx, const Value& valy) -> Value;
auto operator^(const Value& valx, const Value& valy) -> Value;

// Binary operations ([Value] op [double])

auto operator+(const Value& valx, double valy) -> Value;
auto operator-(const Value& valx, double valy) -> Value;
auto operator*(const Value& valx, double valy) -> Value;
auto operator/(const Value& valx, double valy) -> Value;
auto operator^(const Value& valx, double valy) -> Value;

// Binary operations ([double] op [Value])

auto operator+(double valx, const Value& valy) -> Value;
auto operator-(double valx, const Value& valy) -> Value;
auto operator*(double valx, const Value& valy) -> Value;
auto operator/(double valx, const Value& valy) -> Value;
auto operator^(double valx, const Value& valy) -> Value;

// Pretty print value.
auto operator<<(std::ostream& ostr, const Value& val) -> std::ostream&;

} // namespace grapp
