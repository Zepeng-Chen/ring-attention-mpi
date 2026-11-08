#pragma once

#include <vector>

std::vector<float> softmax(
    const std::vector<float>& scores
);

std::vector<std::vector<float>> attention_scores(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K
);

std::vector<std::vector<float>> attention(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V
);

struct OnlineSoftmaxState {
    float m;
    float l;
};

OnlineSoftmaxState online_softmax_update(
    const OnlineSoftmaxState& state,
    const std::vector<float>& block
);