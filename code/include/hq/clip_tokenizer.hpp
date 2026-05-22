#pragma once
/// @file clip_tokenizer.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Simplified CLIP BPE tokenizer for text-to-image pipelines.
/// Produces int64 token IDs compatible with CLIP text encoder.
///
/// This is a minimal but functional implementation. It handles:
/// - BPE subword splitting using a provided vocabulary
/// - <|startoftext|> (49406) and <|endoftext|> (49407) wrapping
/// - Padding/truncating to sequence length (default 77)
/// - Lowercase and basic punctuation normalization
///
/// For production, consider integrating huggingface/tokenizers or
/// the OpenAI tiktoken library. This implementation is self-contained
/// and has no external dependencies.

#include "hq/cxx26_features.hpp"

#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "clip_tokenizer.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <string>
#include <unordered_map>
#include <vector>

namespace hq {

struct TokenizerError {
    std::string message;
};

class CLIPTokenizer {
public:
    /// Default constructor -- loads the built-in minimal vocabulary.
    CLIPTokenizer();

    /// Constructor with explicit vocabulary file path.
    /// @param bpe_merges_file Path to BPE merges file (one merge per line: "token1 token2")
    /// @param vocab_file Optional path to vocabulary JSON (token -> id mapping)
    explicit CLIPTokenizer(const std::string& bpe_merges_file,
                          const std::string& vocab_file = "");

    ~CLIPTokenizer() = default;
    CLIPTokenizer(const CLIPTokenizer&) = default;
    CLIPTokenizer& operator=(const CLIPTokenizer&) = default;
    CLIPTokenizer(CLIPTokenizer&&) noexcept = default;
    CLIPTokenizer& operator=(CLIPTokenizer&&) noexcept = default;

    /// Tokenize a text prompt into CLIP token IDs.
    /// @param text The text prompt to tokenize.
    /// @param max_length Maximum sequence length (default 77 for CLIP).
    /// @return Vector of token IDs, padded to max_length with <|endoftext|>.
    [[nodiscard]] std::vector<std::int64_t> encode(
        const std::string& text,
        std::size_t max_length = 77) const;

    /// Tokenize without padding (returns actual token count).
    [[nodiscard]] std::vector<std::int64_t> encode_raw(
        const std::string& text) const;

    /// Decode token IDs back to text (for debugging).
    [[nodiscard]] std::string decode(
        const std::vector<std::int64_t>& token_ids) const;

    /// Check if vocabulary is loaded.
    [[nodiscard]] bool is_loaded() const noexcept { return !vocab_.empty(); }

    /// Get vocabulary size.
    [[nodiscard]] std::size_t vocab_size() const noexcept { return vocab_.size(); }

    // CLIP special token IDs
    static constexpr std::int64_t BOS_TOKEN = 49406;   // <|startoftext|>
    static constexpr std::int64_t EOS_TOKEN = 49407;   // <|endoftext|>
    static constexpr std::int64_t PAD_TOKEN = 49407;   // <|endoftext|> used as pad
    static constexpr std::int64_t UNK_TOKEN = 49407;   // unknown -> endoftext

private:
    std::unordered_map<std::string, std::int64_t> vocab_;
    std::unordered_map<std::string, std::int64_t> bpe_ranks_;

    // Load the built-in minimal vocabulary (covers common English words)
    void load_builtin_vocab_();

    // Load vocabulary from files
    [[nodiscard]] std::expected<void, TokenizerError> load_from_files_(
        const std::string& bpe_merges_file,
        const std::string& vocab_file);

    // BPE encoding of a single word
    [[nodiscard]] std::vector<std::string> bpe_encode_word_(
        const std::string& word) const;

    // Normalize text (lowercase, handle punctuation)
    [[nodiscard]] std::string normalize_text_(const std::string& text) const;

    // Split text into words
    [[nodiscard]] std::vector<std::string> split_words_(
        const std::string& text) const;
};

} // namespace hq
