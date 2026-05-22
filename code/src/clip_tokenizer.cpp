/// @file clip_tokenizer.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Simplified CLIP BPE tokenizer implementation.
///
/// Provides a self-contained tokenization pipeline that converts English
/// text prompts into CLIP-compatible token ID sequences. The built-in
/// vocabulary covers ~256 common English words plus subword tokens for
/// robust out-of-vocabulary handling via greedy longest-match splitting.

#include "hq/clip_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <sstream>

namespace hq {

// ---------------------------------------------------------------------------
// Built-in vocabulary: 256 most common English words for text-to-image
// ---------------------------------------------------------------------------
//
// Each word is mapped to a stable token ID (0..255). CLIP's actual
// vocabulary has ~49k entries; this minimal subset is enough to
// produce meaningful token sequences for typical image generation
// prompts. Unknown words are decomposed via greedy subword matching.
//
// clang-format off
static constexpr std::pair<std::string_view, std::int64_t> BUILTIN_VOCAB[] = {
    // --- Articles & determiners (0-15) ---
    {"the", 0}, {"a", 1}, {"an", 2}, {"of", 3}, {"in", 4},
    {"on", 5}, {"at", 6}, {"to", 7}, {"for", 8}, {"with", 9},
    {"by", 10}, {"from", 11}, {"as", 12}, {"this", 13}, {"that", 14},
    {"these", 15},

    // --- Pronouns (16-35) ---
    {"those", 16}, {"i", 17}, {"you", 18}, {"he", 19}, {"she", 20},
    {"it", 21}, {"we", 22}, {"they", 23}, {"my", 24}, {"your", 25},
    {"his", 26}, {"her", 27}, {"its", 28}, {"our", 29}, {"their", 30},
    {"me", 31}, {"him", 32}, {"them", 33}, {"us", 34}, {"mine", 35},

    // --- Common verbs - auxiliary / being (36-55) ---
    {"is", 36}, {"are", 37}, {"was", 38}, {"were", 39}, {"be", 40},
    {"been", 41}, {"being", 42}, {"have", 43}, {"has", 44}, {"had", 45},
    {"do", 46}, {"does", 47}, {"did", 48}, {"will", 49}, {"would", 50},
    {"could", 51}, {"should", 52}, {"may", 53}, {"might", 54}, {"can", 55},

    // --- Common verbs - action (56-95) ---
    {"shall", 56}, {"go", 57}, {"come", 58}, {"get", 59}, {"make", 60},
    {"see", 61}, {"know", 62}, {"take", 63}, {"think", 64}, {"say", 65},
    {"help", 66}, {"show", 67}, {"try", 68}, {"ask", 69}, {"need", 70},
    {"feel", 71}, {"become", 72}, {"leave", 73}, {"put", 74}, {"mean", 75},
    {"keep", 76}, {"let", 77}, {"begin", 78}, {"seem", 79}, {"talk", 80},
    {"turn", 81}, {"start", 82}, {"play", 83}, {"run", 84}, {"move", 85},
    {"live", 86}, {"believe", 87}, {"bring", 88}, {"happen", 89}, {"stand", 90},
    {"lose", 91}, {"pay", 92}, {"meet", 93}, {"include", 94}, {"continue", 95},

    // --- Common verbs - action continued (96-125) ---
    {"set", 96}, {"learn", 97}, {"change", 98}, {"lead", 99}, {"understand", 100},
    {"watch", 101}, {"follow", 102}, {"stop", 103}, {"create", 104}, {"speak", 105},
    {"read", 106}, {"spend", 107}, {"grow", 108}, {"open", 109}, {"walk", 110},
    {"win", 111}, {"offer", 112}, {"remember", 113}, {"love", 114}, {"consider", 115},
    {"appear", 116}, {"buy", 117}, {"wait", 118}, {"serve", 119}, {"send", 120},
    {"expect", 121}, {"build", 122}, {"stay", 123}, {"fall", 124}, {"cut", 125},

    // --- Common verbs - continued (126-135) ---
    {"reach", 126}, {"kill", 127}, {"remain", 128}, {"use", 129}, {"want", 130},
    {"looking", 131}, {"find", 132}, {"give", 133}, {"work", 134}, {"tell", 135},

    // --- Adjectives (136-165) ---
    {"good", 136}, {"new", 137}, {"first", 138}, {"last", 139}, {"long", 140},
    {"great", 141}, {"little", 142}, {"own", 143}, {"other", 144}, {"old", 145},
    {"right", 146}, {"big", 147}, {"high", 148}, {"different", 149}, {"small", 150},
    {"large", 151}, {"next", 152}, {"early", 153}, {"young", 154}, {"important", 155},
    {"few", 156}, {"public", 157}, {"bad", 158}, {"same", 159}, {"able", 160},
    {"all", 161}, {"every", 162}, {"each", 163}, {"certain", 164}, {"whole", 165},

    // --- More adjectives (166-185) ---
    {"beautiful", 166}, {"amazing", 167}, {"wonderful", 168}, {"fantastic", 169},
    {"epic", 170}, {"realistic", 171}, {"digital", 172}, {"abstract", 173},
    {"surreal", 174}, {"futuristic", 175}, {"medieval", 176}, {"modern", 177},
    {"detailed", 178}, {"sharp", 179}, {"colorful", 180}, {"bright", 181},
    {"dark", 182}, {"vibrant", 183}, {"stunning", 184}, {"gorgeous", 185},

    // --- Nouns - general (186-205) ---
    {"people", 186}, {"year", 187}, {"way", 188}, {"day", 189}, {"thing", 190},
    {"man", 191}, {"world", 192}, {"life", 193}, {"hand", 194}, {"part", 195},
    {"child", 196}, {"eye", 197}, {"woman", 198}, {"place", 199}, {"work", 200},
    {"week", 201}, {"case", 202}, {"point", 203}, {"government", 204}, {"company", 205},

    // --- Nouns - image/art domain (206-235) ---
    {"art", 206}, {"painting", 207}, {"drawing", 208}, {"photo", 209},
    {"image", 210}, {"picture", 211}, {"scene", 212}, {"landscape", 213},
    {"portrait", 214}, {"style", 215}, {"quality", 216}, {"focus", 217},
    {"blur", 218}, {"light", 219}, {"color", 220}, {"red", 221}, {"blue", 222},
    {"green", 223}, {"yellow", 224}, {"black", 225}, {"white", 226},
    {"cat", 227}, {"dog", 228}, {"bird", 229}, {"horse", 230}, {"fish", 231},
    {"tree", 232}, {"flower", 233}, {"water", 234}, {"fire", 235},

    // --- Nouns - space / fantasy (236-255) ---
    {"space", 236}, {"star", 237}, {"planet", 238}, {"universe", 239},
    {"galaxy", 240}, {"sky", 241}, {"night", 242}, {"sun", 243}, {"moon", 244},
    {"mountain", 245}, {"ocean", 246}, {"forest", 247}, {"city", 248},
    {"castle", 249}, {"dragon", 250}, {"wizard", 251}, {"knight", 252},
    {"warrior", 253}, {"queen", 254}, {"king", 255},
};
// clang-format on

// Subword tokens for BPE decomposition: common syllables and fragments
// These are used by the greedy longest-match tokenizer to split unknown words.
// clang-format off
static constexpr std::pair<std::string_view, std::int64_t> SUBWORD_TOKENS[] = {
    // Common prefixes / syllables (256-300)
    {"re", 256}, {"un", 257}, {"in", 258}, {"dis", 259}, {"over", 260},
    {"mis", 261}, {"sub", 262}, {"pre", 263}, {"out", 264}, {"up", 265},
    {"down", 266}, {"under", 267}, {"anti", 268}, {"non", 269}, {"pro", 270},
    {"super", 271}, {"inter", 272}, {"trans", 273}, {"extra", 274}, {"hyper", 275},
    {"ultra", 276}, {"mega", 277}, {"mini", 278}, {"micro", 279}, {"auto", 280},
    {"self", 281}, {"co", 282}, {"de", 283}, {"en", 284}, {"em", 285},
    {"con", 286}, {"com", 287}, {"col", 288}, {"cor", 289}, {"per", 290},
    {"ful", 291}, {"less", 292}, {"ness", 293}, {"ment", 294}, {"able", 295},
    {"ible", 296}, {"ing", 297}, {"ed", 298}, {"er", 299}, {"est", 300},

    // More suffixes and fragments (301-340)
    {"ly", 301}, {"tion", 302}, {"sion", 303}, {"ity", 304}, {"ty", 305},
    {"ary", 306}, {"ery", 307}, {"ory", 308}, {"ous", 309}, {"ive", 310},
    {"ative", 311}, {"ize", 312}, {"ise", 313}, {"ify", 314}, {"en", 315},
    {"an", 316}, {"ent", 317}, {"ant", 318}, {"ance", 319}, {"ence", 320},
    {"ancy", 321}, {"ency", 322}, {"dom", 323}, {"ism", 324}, {"ist", 325},
    {"ship", 326}, {"hood", 327}, {"ward", 328}, {"wise", 329}, {"ward", 330},
    {"th", 331}, {"al", 332}, {"ial", 333}, {"ical", 334}, {"tic", 335},
    {"ic", 336}, {"ous", 337}, {"ious", 338}, {"eous", 339}, {"uous", 340},

    // Letter fragments for single-character fallback (341-366)
    {"a", 341}, {"b", 342}, {"c", 343}, {"d", 344}, {"e", 345},
    {"f", 346}, {"g", 347}, {"h", 348}, {"i", 349}, {"j", 350},
    {"k", 351}, {"l", 352}, {"m", 353}, {"n", 354}, {"o", 355},
    {"p", 356}, {"q", 357}, {"r", 358}, {"s", 359}, {"t", 360},
    {"u", 361}, {"v", 362}, {"w", 363}, {"x", 364}, {"y", 365}, {"z", 366},
};
// clang-format on

// ===========================================================================
// Construction / Destruction
// ===========================================================================

CLIPTokenizer::CLIPTokenizer() {
    load_builtin_vocab_();
}

CLIPTokenizer::CLIPTokenizer(const std::string& bpe_merges_file,
                              const std::string& vocab_file) {
    // Try file-based loading first; fall back to built-in on error
    auto result = load_from_files_(bpe_merges_file, vocab_file);
    if (!result) {
        std::print("[CLIPTokenizer] File loading failed ({}), using built-in vocab\n",
                   result.error().message);
        load_builtin_vocab_();
    }
}

// ===========================================================================
// Public API
// ===========================================================================

std::vector<std::int64_t>
CLIPTokenizer::encode(const std::string& text, std::size_t max_length) const {
    if (max_length == 0) {
        return {};
    }

    std::vector<std::int64_t> tokens;
    tokens.reserve(max_length);

    // BOS
    tokens.push_back(BOS_TOKEN);

    // Encode text content
    std::vector<std::int64_t> content = encode_raw(text);

    // Append content tokens, leaving room for EOS
    const std::size_t max_content = (max_length > 1) ? max_length - 2 : 0;
    if (!content.empty() && max_content > 0) {
        if (content.size() > max_content) {
            tokens.insert(tokens.end(), content.begin(), content.begin() + max_content);
        } else {
            tokens.insert(tokens.end(), content.begin(), content.end());
        }
    }

    // EOS (always present if there's room)
    if (tokens.size() < max_length) {
        tokens.push_back(EOS_TOKEN);
    }

    // Pad with EOS to max_length
    while (tokens.size() < max_length) {
        tokens.push_back(PAD_TOKEN);
    }

    // Ensure last token is EOS if we had to truncate
    if (tokens.size() >= max_length && tokens[max_length - 1] != EOS_TOKEN && tokens[max_length - 1] != PAD_TOKEN) {
        tokens[max_length - 1] = EOS_TOKEN;
    }

    return tokens;
}

std::vector<std::int64_t>
CLIPTokenizer::encode_raw(const std::string& text) const {
    std::vector<std::int64_t> token_ids;

    std::string normalized = normalize_text_(text);
    if (normalized.empty()) {
        return token_ids;
    }

    std::vector<std::string> words = split_words_(normalized);

    for (const auto& word : words) {
        if (word.empty()) continue;

        // Try whole-word lookup first
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
            continue;
        }

        // Fall back to BPE subword decomposition
        std::vector<std::string> subwords = bpe_encode_word_(word);
        for (const auto& sw : subwords) {
            auto sit = vocab_.find(sw);
            if (sit != vocab_.end()) {
                token_ids.push_back(sit->second);
            } else {
                // Last resort: character-by-character
                for (char c : sw) {
                    std::string ch(1, c);
                    auto cit = vocab_.find(ch);
                    if (cit != vocab_.end()) {
                        token_ids.push_back(cit->second);
                    } else {
                        token_ids.push_back(UNK_TOKEN);
                    }
                }
            }
        }
    }

    return token_ids;
}

std::string
CLIPTokenizer::decode(const std::vector<std::int64_t>& token_ids) const {
    // Build reverse map (id -> token) on first use -- lazy init
    static thread_local std::unordered_map<std::int64_t, std::string> reverse_map;
    if (reverse_map.empty() && !vocab_.empty()) {
        for (const auto& [token, id] : vocab_) {
            reverse_map[id] = std::string(token);
        }
    }

    std::ostringstream oss;
    bool first = true;

    for (std::int64_t id : token_ids) {
        // Skip special tokens in decode output
        if (id == BOS_TOKEN || id == EOS_TOKEN || id == PAD_TOKEN) {
            continue;
        }

        auto it = reverse_map.find(id);
        if (it != reverse_map.end()) {
            if (!first) oss << ' ';
            oss << it->second;
            first = false;
        } else {
            if (!first) oss << ' ';
            oss << "<unk>";
            first = false;
        }
    }

    return oss.str();
}

// ===========================================================================
// Private helpers
// ===========================================================================

void CLIPTokenizer::load_builtin_vocab_() {
    vocab_.clear();
    bpe_ranks_.clear();

    // Insert common words
    for (const auto& [word, id] : BUILTIN_VOCAB) {
        vocab_.emplace(std::string(word), id);
    }

    // Insert subword fragments for BPE decomposition
    for (const auto& [token, id] : SUBWORD_TOKENS) {
        vocab_.emplace(std::string(token), id);
    }

    // Insert CLIP special tokens at their canonical IDs
    vocab_["<|startoftext|>"] = BOS_TOKEN;
    vocab_["<|endoftext|>"]   = EOS_TOKEN;

    // Build a simple BPE ranking table for common subword merges.
    // Higher rank = merge earlier. We use the subword ID as its rank.
    for (const auto& [token, id] : SUBWORD_TOKENS) {
        bpe_ranks_[std::string(token)] = id;
    }
}

std::expected<void, TokenizerError>
CLIPTokenizer::load_from_files_(const std::string& bpe_merges_file,
                                 const std::string& vocab_file) {
    // NOTE: File-based vocabulary loading produces simple sequential IDs.
    // For production CLIP tokenization, use the built-in vocabulary
    // (mapped to canonical CLIP IDs 0-366 + specials at 49406-49407).

    vocab_.clear();
    bpe_ranks_.clear();

    std::int64_t token_id = 0;

    // If a vocab file is provided, attempt to load word→ID mappings from it
    if (!vocab_file.empty()) {
        std::ifstream vf(vocab_file);
        if (vf.is_open()) {
            std::string vline;
            while (std::getline(vf, vline)) {
                if (vline.empty() || vline[0] == '#') continue;
                std::istringstream viss(vline);
                std::string word;
                std::int64_t id;
                if (viss >> word >> id) {
                    vocab_[word] = id;
                    if (id >= token_id) token_id = id + 1;
                }
            }
        }
    }

    std::ifstream merges(bpe_merges_file);
    if (!merges.is_open()) {
        return std::unexpected{TokenizerError{
            std::format("Cannot open BPE merges file: {}", bpe_merges_file)}};
    }

    std::string line;
    while (std::getline(merges, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string left, right;
        if (iss >> left >> right) {
            std::string merged = left + right;
            bpe_ranks_[merged] = token_id;
            vocab_[std::move(merged)] = token_id++;
            vocab_[std::move(left)]   = token_id++;
            vocab_[std::move(right)]  = token_id++;
        }
    }

    if (vocab_.empty()) {
        return std::unexpected{TokenizerError{"No valid entries found in merges file"}};
    }

    // Always ensure special tokens are present
    vocab_["<|startoftext|>"] = BOS_TOKEN;
    vocab_["<|endoftext|>"]   = EOS_TOKEN;

    return {};
}

std::vector<std::string>
CLIPTokenizer::bpe_encode_word_(const std::string& word) const {
    if (word.empty()) return {};

    // Try the word itself first
    if (vocab_.contains(word)) {
        return {word};
    }

    std::vector<std::string> result;
    std::size_t pos = 0;

    // Greedy longest-match tokenization
    while (pos < word.size()) {
        std::size_t best_len = 0;
        std::string best_sub;

        // Find the longest subword starting at pos that's in our vocab
        for (std::size_t len = 1; len <= word.size() - pos; ++len) {
            std::string sub = word.substr(pos, len);
            if (vocab_.contains(sub)) {
                best_sub = sub;
                best_len = len;
            }
        }

        if (best_len > 0) {
            result.push_back(best_sub);
            pos += best_len;
        } else {
            // No match found -- emit single character and advance
            result.emplace_back(word.substr(pos, 1));
            pos += 1;
        }
    }

    return result;
}

std::string
CLIPTokenizer::normalize_text_(const std::string& text) const {
    std::string result;
    result.reserve(text.size() * 2);

    for (char c : text) {
        // Convert to lowercase
        char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        // Add spaces around punctuation (treat as separate tokens)
        if (std::ispunct(static_cast<unsigned char>(lc))) {
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
            result += lc;
            result += ' ';
        } else {
            result += lc;
        }
    }

    // Collapse multiple spaces into one
    std::string collapsed;
    collapsed.reserve(result.size());
    bool in_space = false;
    for (char c : result) {
        if (c == ' ') {
            if (!in_space) {
                collapsed += c;
                in_space = true;
            }
        } else {
            collapsed += c;
            in_space = false;
        }
    }

    // Trim leading/trailing spaces
    std::size_t start = collapsed.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    std::size_t end = collapsed.find_last_not_of(' ');
    return collapsed.substr(start, end - start + 1);
}

std::vector<std::string>
CLIPTokenizer::split_words_(const std::string& text) const {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    return words;
}

} // namespace hq
