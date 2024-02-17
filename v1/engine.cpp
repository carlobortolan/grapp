#include "engine.h"
#include <iostream>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace grapp {

    Value::Value(double data) : data(data) {}

    Value::Value(double data, std::string_view op) : data(data), _op(op) {}   

    Value::Value(double data, std::vector<Value*> children, std::string_view op)
        : data(data), _prev(std::move(children)), _op(op) {}

    Value::Value(double data, std::initializer_list<Value*> children, std::string_view op) 
        : data(data), _prev(children), _op(op) {}

    auto Value::operator=(double data) -> Value & {
        this->data = data;
        return *this;
    }

    auto Value::relu() -> Value& {
        data = (data < 0) ? 0 : data;

        // Define the backward function
        _backward = [this]() {
            grad += (data > 0) ? grad : 0;
        };

        return *this;
    }

    auto Value::backward() -> void {
        std::vector<Value*> topo = {};
        std::set<Value*> visited = {};

         std::function<void(Value*)> build_topo = [&](Value* node) {
            if (visited.count(node)) {
                std::cout << "Already visited" << std::endl;
                return;
            }
            visited.insert(node);
            std::cout << "Insert: " << node << std::endl;

            for (auto* prev : node->_prev) {
                std::cout << "Prev: " << prev << std::endl;
                build_topo(prev);
            }
            std::cout << "Pushback"<< std::endl;
            topo.push_back(node);
        };

        build_topo(this);

        this->grad = 1;

        std::cout << "Starting backward pass..." << std::endl;
        std::reverse(topo.begin(), topo.end());
        for (auto* node : topo) {
            std::cout << "Node: " << *node << ", Grad: " << node->grad << std::endl;
            node->_backward();
        }
        std::cout << "Completed backward pass." << std::endl;

        // Reset visited flags after each backward pass
        for (auto* node : topo) {
            node->visited = false;
        }
    }

    auto Value::operator+=(const Value& val) -> Value& {
        data += val.data;
        _op = "+";
        _prev.push_back(new Value(val));
        _backward = [&val, this]() {
            val.grad += grad;
        };
        return *this;
    }

    auto Value::operator-=(const Value& val) -> Value& {
        data -= val.data;
        _op = "-";
        _prev.push_back(new Value(val));
        _backward = [&val, this]() {
            val.grad -= grad;
        };
        return *this;
    }

    auto operator+(const Value &valx, const Value &valy) -> Value {
        auto result = Value(valx.data + valy.data, "+");
        result._prev.push_back(new Value(valx));
        result._prev.push_back(new Value(valy));

        std::cout << "grad (result): " << result.grad << std::endl;

        std::function<void()> backward_fn = [&valx, &valy, &result]() {
            valx.grad += result.grad;
            valy.grad += result.grad;
            std::cout << "grad (valx): " << valx.grad << std::endl;
            std::cout << "grad (valy): " << valy.grad << std::endl;
        };

        result._backward = backward_fn;

        return result;
    }

    auto operator-(const Value &valx, const Value &valy) -> Value {
        auto result = Value(valx.data - valy.data, "-");
        result._prev.push_back(new Value(valx));
        result._prev.push_back(new Value(valy));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, &valy, &result]() {
            valx.grad += result.grad;
            valy.grad -= result.grad;
            std::cout << "grad (valx): " << valx.grad << std::endl;
            std::cout << "grad (valy): " << valy.grad << std::endl;
        };
        return result;
    }

    auto operator*(const Value &valx, const Value &valy) -> Value {
        auto result = Value(valx.data * valy.data, "*");
        result._prev.push_back(new Value(valx));
        result._prev.push_back(new Value(valy));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, &valy, &result]() {
            valx.grad += valy.data * result.grad;
            valy.grad += valx.data * result.grad;
            std::cout << "grad (valx): " << valx.grad << std::endl;
            std::cout << "grad (valy): " << valy.grad << std::endl;
        };
        return result;
    }

    auto operator/(const Value &valx, const Value &valy) -> Value {
        auto result = Value(valx.data / valy.data, "/");
        result._prev.push_back(new Value(valx));
        result._prev.push_back(new Value(valy));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, &valy, &result]() {
            valx.grad += result.grad / valy.data;
            valy.grad -= result.grad * valx.data / (valy.data * valy.data);
            std::cout << "grad (valx): " << valx.grad << std::endl;
            std::cout << "grad (valy): " << valy.grad << std::endl;
        };
        return result;
    }

    /*auto operator^(const Value &valx, const Value &valy) -> Value {
        auto result = Value(std::pow(valx.data, valy.data), "^");
        result._prev.push_back(new Value(valx));
        result._prev.push_back(new Value(valy));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, &valy, &result]() {
            valx.grad += result.grad * valy.data * std::pow(valx.data, valy.data - 1);
            valy.grad += result.grad * std::pow(valx.data, valy.data) * std::log(valx.data);
            std::cout << "grad (valx): " << valx.grad << std::endl;
            std::cout << "grad (valy): " << valy.grad << std::endl;
        };
        return result;
    }
    */

    auto operator+(const Value &valx, double valy) -> Value {
        auto result = Value(valx.data + valy, "+");
        result._prev.push_back(new Value(valx));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, valy, &result]() {
            valx.grad += result.grad;
        };
        return result;
    }

    auto operator-(const Value &valx, double valy) -> Value {
        auto result = Value(valx.data - valy, "-");
        result._prev.push_back(new Value(valx));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, valy, &result]() {
            valx.grad -= result.grad;
        };
        return result;
    }

    auto operator*(const Value &valx, double valy) -> Value {
        auto result = Value(valx.data * valy, "*");
        result._prev.push_back(new Value(valx));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, valy, &result]() {
            valx.grad += valy * result.grad;
        };
        return result;
    }

    auto operator/(const Value &valx, double valy) -> Value {
        auto result = Value(valx.data / valy, "/");
        result._prev.push_back(new Value(valx));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, valy, &result]() {
            valx.grad += result.grad / valy;
        };
        return result;
    }

    auto operator^(const Value &valx, double valy) -> Value {
        auto result = Value(std::pow(valx.data, valy), "^");
        result._prev.push_back(new Value(valx));

        std::cout << "grad (result): " << result.grad << std::endl;

        result._backward = [&valx, valy, &result]() {
            valx.grad += (valy * std::pow(valx.data, valy - 1)) * result.grad;
        };
        return result;
    }

    auto operator+(double valx, const Value &valy) -> Value {
        auto result = Value(valx + valy.data, "+");
        result._prev.push_back(new Value(valy));
        result._backward = [&valy, &result]() {
            valy.grad += result.grad;
        };
        return result;
    }

    auto operator-(double valx, const Value &valy) -> Value {
        auto result = Value(valx - valy.data, "-");
        result._prev.push_back(new Value(valy));
        result._backward = [&valy, &result]() {
            valy.grad -= result.grad;
        };
        return result;
    }

    auto operator*(double valx, const Value &valy) -> Value {
        auto result = Value(valx * valy.data, "*");
        result._prev.push_back(new Value(valy));
        result._backward = [&valy, valx, &result]() {
            valy.grad += valx * result.grad;
        };
        return result;
    }

    auto operator/(double valx, const Value &valy) -> Value {
        auto result = Value(valx / valy.data, "/");
        result._prev.push_back(new Value(valy));
        result._backward = [&valy, valx, &result]() {
            valy.grad -= result.grad * valx / (valy.data * valy.data);
        };
        return result;
    }

    /*auto operator^(double valx, const Value &valy) -> Value {
        auto result = Value(std::pow(valx, valy.data), "^");
        result._prev.push_back(new Value(valy));
        result._backward = [valx, &valy, &result]() {
            valy.grad += result.grad * std::pow(valx, valy.data) * std::log(valx);
        };
        return result;
    }*/

    auto operator<<(std::ostream &ostr, const Value &node) -> std::ostream & {
        ostr << "Node(value=" << node.data << ", grad_fn=" << node.grad << ", op=" << node._op << ")";
        return ostr;
    }
  
    auto Value::visualizeGraph() -> void {
        std::stringstream ss;
        ss << "digraph ComputationalGraph {\n";
        ss << "    node [shape=ellipse, style=filled, color=lightblue, fontname=Courier];\n";
        ss << "    edge [fontname=Courier];\n\n";

        // Recursive function to traverse the computational graph and generate nodes and edges
        std::function<void(const Value*)> generateGraph = [&](const Value* node) {
            ss << "    \"" << node << "\" [label=\"" << *node << "\"];\n";
            for (const auto* prev : node->_prev) {
                ss << "    \"" << prev << "\" -> \"" << node << "\";\n";
                generateGraph(prev);
            }
        };

        generateGraph(this);

        ss << "}\n";

        std::ofstream file("../../../computational_graph.dot");
        file << ss.str();
        file.close();

        std::cout << "Graphviz representation saved to computational_graph.dot" << std::endl;
    }
}
