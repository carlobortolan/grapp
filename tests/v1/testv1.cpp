#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "v1.h"
/*
TEST_CASE("Sanity check") {
    grapp::Value x(-4.0);
    grapp::Value z = 2 * x + 2 + x;
    grapp::Value q = z.relu() + z * x;
    grapp::Value h = (z * z).relu();
    grapp::Value y = h + q + q * x;
    y.backward();
    auto xmg = x, ymg = y;

    torch::Tensor xpt = torch::tensor({-4.0}, torch::kDouble).requires_grad_(true);
    torch::Tensor zpt = 2 * xpt + 2 + xpt;
    torch::Tensor qpt = zpt.relu() + zpt * xpt;
    torch::Tensor hpt = (zpt * zpt).relu();
    torch::Tensor ypt = hpt + qpt + qpt * xpt;
    ypt.backward();
    auto xmg = x, ymg = y;

    // Forward pass check
    CHECK(xmg.data == ypt.item<double>());
    // Backward pass check
    CHECK(xmg.grad == xpt.grad().item<double>());
}

TEST_CASE("More ops") {
    grapp::Value a(-4.0);
    grapp::Value b(2.0);
    grapp::Value c = a + b;
    grapp::Value d = a * b + (b^3);
    c += c + 1;
    c += 1 + c + (-1 * a);
    d += d * 2 + (b + a).relu();
    d += 3 * d + (b - a).relu();
    grapp::Value e = c - d;
    grapp::Value f = e^2;
    grapp::Value g = f / 2.0;
    g += 10.0 / f;
    g.backward();
    auto amg = a, bmg = b, gmg = g;

    torch::Tensor apt = torch::tensor({-4.0}, torch::kDouble).requires_grad_(true);
    torch::Tensor bpt = torch::tensor({2.0}, torch::kDouble).requires_grad_(true);
    torch::Tensor cpt = apt + bpt;
    torch::Tensor dpt = apt * bpt + torch::pow(bpt, 3);
    cpt += cpt + 1;
    cpt += 1 + cpt + (-apt);
    dpt += dpt * 2 + (bpt + apt).relu();
    dpt += 3 * dpt + (bpt - apt).relu();
    torch::Tensor ept = cpt - dpt;
    torch::Tensor fpt = torch::pow(ept, 2);
    torch::Tensor gpt = fpt / 2.0;
    gpt += 10.0 / fpt;
    gpt.backward();
    auto amg = a, bmg = b, gmg = g;

    // Forward pass check
    double tol = 1e-6;
    CHECK(std::abs(gmg.data - gpt.item<double>()) < tol);
    // Backward pass check
    CHECK(std::abs(amg.grad - apt.grad().item<double>()) < tol);
    CHECK(std::abs(bmg.grad - bpt.grad().item<double>()) < tol);
}
*/

TEST_CASE("Gradient computation") {
    double epsilon = 1e-4; // Small value for numerical stability

    // Step 1: Initialize input values
    grapp::Value a(-4.0);
    grapp::Value b(2.0);

    // Step 2: Perform operations to create a computational graph
    grapp::Value c = a + b;
    grapp::Value d = a * b + (b ^ 3);
    c += c + 1;
    c += 1 + c + (-1 * a);
    d += d * 2 ;//+ (b + a).relu();
    // d += 3 * d + (b - a).relu();
    // grapp::Value e = c - d;
    // grapp::Value f = e ^ 2;
    // grapp::Value g = f / 2.0;
    // g += 10.0 / f;

    // Step 3: Verify data
    /*CHECK(a.data + 4.0 < epsilon);
    CHECK(b.data - 2 < epsilon);
    CHECK(c.data + 1 < epsilon);
    CHECK(d.data - 6 < epsilon);
    CHECK(e.data + 7 < epsilon);
    CHECK(f.data - 49 < epsilon);
    CHECK(g.data - 24.7041 < epsilon);
    */

    // Step 4: Verify ops
    /*CHECK(a._op == "");
    CHECK(b._op == "");
    CHECK(c._op == "+");
    CHECK(d._op == "+");
    CHECK(e._op == "-");
    CHECK(f._op == "^");
    CHECK(g._op == "+");
    */

    // Step 5: Compute gradients
    d.backward();
    // d.visualizeGraph();
    //b.backward();
    //c.backward();
    //d.backward();
    //e.backward();
    //f.backward();
    //g.backward();

    // Step 6: Verify gradients
    CHECK(a.grad == 3);
    CHECK(b.grad == 4);
}