#include <iostream>
#include <limits>
#include <cmath>
#include "attention.hpp"

int main() {
    OnlineSoftmaxState state = {
        -std::numeric_limits<float>::infinity(),
        0.0f
    };

    std::vector<float> block1 = {1.0f, 2.0f};
    std::vector<float> block2 = {3.0f, 4.0f};

    state = online_softmax_update(state, block1);

    std::cout << "After block 1: "
              << "m = " << state.m
              << ", l = " << state.l << '\n';

    state = online_softmax_update(state, block2);

    std::cout << "After block 2: "
              << "m = " << state.m
              << ", l = " << state.l << '\n';

    return 0;
}