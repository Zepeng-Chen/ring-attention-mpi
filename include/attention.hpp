#pragma once

#include <vector>
#include <cstddef>

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

struct OnlineAttentionState {
    float m;
    float l;
    std::vector<float> o;
};

OnlineAttentionState online_attention_update(
    const OnlineAttentionState& state,
    const std::vector<float>& scores,
    const std::vector<std::vector<float>>& V_block
);

std::vector<std::vector<float>> blockwise_attention(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V,
    std::size_t block_size
);