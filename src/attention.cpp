#include "attention.hpp"
#include <limits>

std::vector<float> softmax(
    const std::vector<float>& scores
) {
    if (scores.empty()) {
        return {};
    }

    const float max_score = 
        *std::max_element(scores.begin(), scores.end());

    std::vector<float> probabilities;
    probabilities.reserve(scores.size());

    float sum = 0.0f;

    for (float score : scores) {
        float value = std::exp(score - max_score);
        probabilities.push_back(value);
        sum += value;
    }

    for (float& probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

std::vector<std::vector<float>> attention_scores(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K
) {
    const size_t N = Q.size();
    const size_t M = K.size();
    const size_t d_k = Q[0].size();

    std::vector<std::vector<float>> S(
        N, std::vector<float>(M, 0.0f)
    );

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float dot = 0.0f;

            for (size_t k = 0; k < d_k; ++k) {
                dot += Q[i][k] * K[j][k];
            }

            S[i][j] = dot * scale;
        }
    }
    return S;
}

std::vector<std::vector<float>> attention(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V
) {
    const size_t N = Q.size();
    const size_t M = K.size();
    const size_t d_v = V[0].size();

    // step1: compute scaled attention scores
    auto S = attention_scores(Q, K);

    // step2: apply softmax to each row
    for (auto& row : S) {
        row = softmax(row);
    }

    // step3: multiply attention weights by V
    std::vector<std::vector<float>> O(
        N, std::vector<float>(d_v, 0.0f)
    );

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < d_v; ++k) {
                O[i][k] += S[i][j] * V[j][k];
            }
        }
    }

    return O;
}

OnlineSoftmaxState online_softmax_update(
    const OnlineSoftmaxState& state,
    const std::vector<float>& block
) {
    if (block.empty()) {
        return state;
    }

    // step 1: find maximum score in current block
    const float block_max = *std::max_element(block.begin(), block.end());

    // step 2: compute updated global maximum
    const float m_new = std::max(state.m, block_max);

    // step 3: rescale previous denominator
    float l_new = state.l * std::exp(state.m - m_new);

    // step 4: add contributions from current block
    for (float score : block) {
        l_new += std::exp(score - m_new);
    }

    return {m_new, l_new}; 
}
