#include <iostream>
#include <limits>
#include <cmath>
#include "attention.hpp"
#include <algorithm>
#include <stdexcept>

int main()
{
    OnlineSoftmaxState state = {
        -std::numeric_limits<float>::infinity(),
        0.0f};

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

    OnlineAttentionState old_state = {
        2.0f,
        1.5f,
        {3.0f, 6.0f}};

    const float new_m = 4.0f;
    const float alpha = std::exp(old_state.m - new_m);

    OnlineAttentionState rescaled = old_state;
    rescaled.m = new_m;
    rescaled.l *= alpha;

    for (float &value : rescaled.o)
    {
        value *= alpha;
    }

    std::cout << "alpha = " << alpha << '\n';

    for (size_t k = 0; k < old_state.o.size(); ++k)
    {
        std::cout << "Before: " << old_state.o[k] / old_state.l
                  << ", after rescaling: " << rescaled.o[k] / rescaled.l
                  << '\n';
    }

    const std::vector<std::vector<float>> Q = {
        {1.0f, 2.0f, 3.0f},
        {-1.0f, 1.0f, 2.0f}};

    const std::vector<std::vector<float>> K = {
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {-1.0f, 2.0f, 1.0f}};

    const std::vector<std::vector<float>> V = {
        {10.0f, 0.0f},
        {0.0f, 20.0f},
        {30.0f, 0.0f},
        {0.0f, 40.0f}};

    const auto serial = attention(Q, K, V);

    const std::vector<size_t> block_sizes = {1, 2, 3, 4, 8};

    for (size_t block_size : block_sizes)
    {
        std::cout << "Testing block_size = " << block_size << '\n';

        const auto blocked = blockwise_attention(Q, K, V, block_size);

        for (size_t i = 0; i < serial.size(); ++i)
        {
            for (size_t k = 0; k < serial[i].size(); ++k)
            {
                const float reference = serial[i][k];
                const float actual = blocked[i][k];
                const float error = std::abs(actual - reference);

                std::cout << "Query " << i << ", component " << k
                          << ": serial = " << reference
                          << ", blockwise = " << actual
                          << ", abs error = " << error << '\n';

                if (!std::isfinite(actual) ||
                    !std::isfinite(reference) ||
                    error > 1e-5f)
                {
                    std::cerr << "Attention comparison failed\n";
                    return 1;
                }
            }
        }
    }

    try
    {
        blockwise_attention(Q, K, V, 0);

        // 如果没有抛出异常，说明非法参数检查没有生效。
        std::cerr << "Zero block size was not rejected\n";
        return 1;
    }
    catch (const std::invalid_argument &error)
    {
        std::cout << "Invalid block size rejected: "
                  << error.what() << '\n';
    }
    try
    {
        blockwise_attention({}, K, V, 3);

        std::cerr << "Empty Q was not rejected\n";
        return 1;
    }
    catch (const std::invalid_argument &error)
    {
        std::cout << "Empty Q rejected: "
                  << error.what() << '\n';
    }

    auto short_V = V;
    short_V.pop_back();

    try
    {
        blockwise_attention(Q, K, short_V, 3);

        std::cerr << "Mismatched K/V rows were not rejected\n";
        return 1;
    }
    catch (const std::invalid_argument &error)
    {
        std::cout << "Mismatched K/V rows rejected: "
                  << error.what() << '\n';
    }

    std::cout << "Attention comparison passed\n";
    return 0;
}