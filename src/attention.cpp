#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

    void validate_attention_inputs(
        const std::vector<std::vector<float>> &Q,
        const std::vector<std::vector<float>> &K,
        const std::vector<std::vector<float>> &V)
    {
        if (Q.empty() || K.empty() || V.empty())
        {
            throw std::invalid_argument("Q, K and V must be non-empty");
        }

        if (K.size() != V.size())
        {
            throw std::invalid_argument(
                "K and V must have the same number of rows");
        }
    }

} // namespace

std::vector<float> softmax(
    const std::vector<float> &scores)
{
    if (scores.empty())
    {
        return {};
    }

    const float max_score =
        *std::max_element(scores.begin(), scores.end());

    std::vector<float> probabilities;
    probabilities.reserve(scores.size());

    float sum = 0.0f;

    for (float score : scores)
    {
        float value = std::exp(score - max_score);
        probabilities.push_back(value);
        sum += value;
    }

    for (float &probability : probabilities)
    {
        probability /= sum;
    }

    return probabilities;
}

std::vector<std::vector<float>> attention_scores(
    const std::vector<std::vector<float>> &Q,
    const std::vector<std::vector<float>> &K)
{
    const size_t N = Q.size();
    const size_t M = K.size();
    const size_t d_k = Q[0].size();

    std::vector<std::vector<float>> S(
        N, std::vector<float>(M, 0.0f));

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            float dot = 0.0f;

            for (size_t k = 0; k < d_k; ++k)
            {
                dot += Q[i][k] * K[j][k];
            }

            S[i][j] = dot * scale;
        }
    }
    return S;
}

std::vector<std::vector<float>> attention(
    const std::vector<std::vector<float>> &Q,
    const std::vector<std::vector<float>> &K,
    const std::vector<std::vector<float>> &V)
{
    validate_attention_inputs(Q, K, V);

    const size_t N = Q.size();
    const size_t M = K.size();
    const size_t d_v = V[0].size();

    // step1: compute scaled attention scores
    auto S = attention_scores(Q, K);

    // step2: apply softmax to each row
    for (auto &row : S)
    {
        row = softmax(row);
    }

    // step3: multiply attention weights by V
    std::vector<std::vector<float>> O(
        N, std::vector<float>(d_v, 0.0f));

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            for (size_t k = 0; k < d_v; ++k)
            {
                O[i][k] += S[i][j] * V[j][k];
            }
        }
    }

    return O;
}

OnlineSoftmaxState online_softmax_update(
    const OnlineSoftmaxState &state,
    const std::vector<float> &block)
{
    if (block.empty())
    {
        return state;
    }

    // step 1: find maximum score in current block
    const float block_max = *std::max_element(block.begin(), block.end());

    // step 2: compute updated global maximum
    const float m_new = std::max(state.m, block_max);

    // step 3: rescale previous denominator
    float l_new = state.l * std::exp(state.m - m_new);

    // step 4: add contributions from current block
    for (float score : block)
    {
        l_new += std::exp(score - m_new);
    }

    return {m_new, l_new};
}

OnlineAttentionState online_attention_update(
    const OnlineAttentionState &state,
    const std::vector<float> &scores,
    const std::vector<std::vector<float>> &V_block)
{
    if (scores.empty())
    {
        return state;
    }

    const auto softmax_state = online_softmax_update(
        {state.m, state.l},
        scores);

    OnlineAttentionState next = {
        softmax_state.m,
        softmax_state.l,
        state.o};

    const float alpha = std::exp(state.m - next.m);

    for (float &value : next.o)
    {
        value *= alpha;
    }

    for (size_t j = 0; j < scores.size(); ++j)
    {
        const float weight = std::exp(scores[j] - next.m);

        for (size_t k = 0; k < next.o.size(); ++k)
        {
            next.o[k] += weight * V_block[j][k];
        }
    }

    return next;
}

std::vector<std::vector<float>> blockwise_attention(
    const std::vector<std::vector<float>> &Q,
    const std::vector<std::vector<float>> &K,
    const std::vector<std::vector<float>> &V,
    std::size_t block_size)
{
    if (block_size == 0)
    {
        throw std::invalid_argument("block_size must be positive");
    }

    validate_attention_inputs(Q, K, V);

    const std::size_t N = Q.size();
    const std::size_t M = K.size();
    const std::size_t d_k = Q[0].size();
    const std::size_t d_v = V[0].size();

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    std::vector<std::vector<float>> O(
        N, std::vector<float>(d_v, 0.0f));

    for (std::size_t i = 0; i < N; ++i)
    {
        OnlineAttentionState state = {
            -std::numeric_limits<float>::infinity(),
            0.0f,
            std::vector<float>(d_v, 0.0f)};

        for (std::size_t start = 0; start < M; start += block_size)
        {
            const std::size_t end = std::min(start + block_size, M);

            std::vector<float> scores;
            std::vector<std::vector<float>> V_block;

            for (std::size_t j = start; j < end; ++j)
            {
                float dot = 0.0f;

                for (std::size_t k = 0; k < d_k; ++k)
                {
                    dot += Q[i][k] * K[j][k];
                }

                scores.push_back(dot * scale);
                V_block.push_back(V[j]);
            }

            state = online_attention_update(state, scores, V_block);
        }

        for (std::size_t k = 0; k < d_v; ++k)
        {
            O[i][k] = state.o[k] / state.l;
        }
    }

    return O;
}
