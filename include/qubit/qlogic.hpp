#pragma once

#include "qubit/qmath.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

enum class QLogicTruth : std::uint8_t {
    False = 0,
    True = 1,
    Unknown = 2,
};

struct QHornLogicConfig {
    std::size_t max_atoms{1U << 16U};
    std::size_t max_input_facts{1U << 20U};
    std::size_t max_rules{1U << 20U};
    std::size_t max_premise_terms{1U << 22U};
};

struct QHornLogicReceipt {
    std::size_t atoms{0U};
    std::size_t input_facts{0U};
    std::size_t rules{0U};
    std::size_t premise_terms{0U};
    std::size_t known_literals{0U};
    std::size_t derived_literals{0U};
    std::size_t known_true{0U};
    std::size_t known_false{0U};
    std::string canonical{};
    bool exact{true};
    bool consistent{true};
};

class QHornLogic {
public:
    using AtomId = std::size_t;

    struct Literal {
        AtomId atom{0U};
        bool value{true};

        friend bool operator==(const Literal&, const Literal&) = default;
    };

    struct Rule {
        std::vector<Literal> premises{};
        Literal conclusion{};

        friend bool operator==(const Rule&, const Rule&) = default;
    };

    explicit QHornLogic(QHornLogicConfig config = {}) : config_(config) {
        if (config_.max_atoms == 0U || config_.max_input_facts == 0U ||
            config_.max_rules == 0U || config_.max_premise_terms == 0U) {
            throw QMathError("Horn-logic limits must be positive");
        }
    }

    [[nodiscard]] AtomId add_atom(std::string identity) {
        if (identity.empty()) throw QMathError("Horn-logic atom identity is empty");
        const auto found = index_.find(identity);
        if (found != index_.end()) return found->second;
        if (atoms_.size() >= config_.max_atoms) throw QMathError("Horn-logic atom cap exceeded");
        const AtomId id = atoms_.size();
        atoms_.push_back(std::move(identity));
        index_.emplace(atoms_.back(), id);
        compiled_.reset();
        return id;
    }

    [[nodiscard]] std::optional<AtomId> find_atom(std::string_view identity) const {
        const auto found = index_.find(std::string(identity));
        if (found == index_.end()) return std::nullopt;
        return found->second;
    }

    [[nodiscard]] bool add_fact(Literal fact) {
        require_literal(fact);
        if (std::find(facts_.begin(), facts_.end(), fact) != facts_.end()) return false;
        if (facts_.size() >= config_.max_input_facts) throw QMathError("Horn-logic fact cap exceeded");
        std::vector<Literal> candidate = facts_;
        candidate.push_back(fact);
        Compiled next = compile(candidate, rules_);
        facts_ = std::move(candidate);
        compiled_ = std::move(next);
        return true;
    }

    [[nodiscard]] bool add_fact(AtomId atom, bool value = true) {
        return add_fact(Literal{atom, value});
    }

    [[nodiscard]] bool add_rule(std::span<const Literal> premises, Literal conclusion) {
        require_literal(conclusion);
        Rule candidate_rule = normalize_rule(premises, conclusion);
        if (std::find(rules_.begin(), rules_.end(), candidate_rule) != rules_.end()) return false;
        if (rules_.size() >= config_.max_rules) throw QMathError("Horn-logic rule cap exceeded");
        const std::size_t existing_terms = premise_terms(rules_);
        if (candidate_rule.premises.size() > config_.max_premise_terms - existing_terms) {
            throw QMathError("Horn-logic premise-term cap exceeded");
        }
        std::vector<Rule> candidate = rules_;
        candidate.push_back(std::move(candidate_rule));
        Compiled next = compile(facts_, candidate);
        rules_ = std::move(candidate);
        compiled_ = std::move(next);
        return true;
    }

    [[nodiscard]] QLogicTruth truth(AtomId atom) const {
        require_atom(atom);
        ensure_compiled();
        const bool negative = known(*compiled_, Literal{atom, false});
        const bool positive = known(*compiled_, Literal{atom, true});
        if (negative && positive) throw QMathError("Horn-logic accepted state is contradictory");
        if (positive) return QLogicTruth::True;
        if (negative) return QLogicTruth::False;
        return QLogicTruth::Unknown;
    }

    [[nodiscard]] QLogicTruth truth(std::string_view identity) const {
        const std::optional<AtomId> atom = find_atom(identity);
        if (!atom.has_value()) throw QMathError("Horn-logic atom identity is unknown");
        return truth(*atom);
    }

    [[nodiscard]] bool entails(Literal literal) const {
        require_literal(literal);
        ensure_compiled();
        return known(*compiled_, literal);
    }

    [[nodiscard]] std::span<const std::string> atoms() const noexcept {
        return atoms_;
    }

    [[nodiscard]] std::size_t atom_count() const noexcept {
        return atoms_.size();
    }

    [[nodiscard]] std::size_t input_fact_count() const noexcept {
        return facts_.size();
    }

    [[nodiscard]] std::size_t rule_count() const noexcept {
        return rules_.size();
    }

    [[nodiscard]] std::string canonical() const {
        ensure_compiled();
        return compiled_->canonical;
    }

    [[nodiscard]] QHornLogicReceipt receipt() const {
        ensure_compiled();
        return QHornLogicReceipt{
            atoms_.size(),
            facts_.size(),
            rules_.size(),
            premise_terms(rules_),
            compiled_->known_literals,
            compiled_->known_literals - facts_.size(),
            compiled_->known_true,
            compiled_->known_false,
            compiled_->canonical,
            true,
            true,
        };
    }

private:
    struct Compiled {
        std::vector<std::uint8_t> known{};
        std::size_t known_literals{0U};
        std::size_t known_true{0U};
        std::size_t known_false{0U};
        std::string canonical{};
    };

    QHornLogicConfig config_{};
    std::vector<std::string> atoms_{};
    std::unordered_map<std::string, AtomId> index_{};
    std::vector<Literal> facts_{};
    std::vector<Rule> rules_{};
    mutable std::optional<Compiled> compiled_{};

    [[nodiscard]] static std::size_t literal_index(Literal literal) noexcept {
        return literal.atom * 2U + static_cast<std::size_t>(literal.value);
    }

    void require_atom(AtomId atom) const {
        if (atom >= atoms_.size()) throw QMathError("Horn-logic atom id is out of range");
    }

    void require_literal(Literal literal) const {
        require_atom(literal.atom);
    }

    [[nodiscard]] Rule normalize_rule(
        std::span<const Literal> premises,
        Literal conclusion) const {
        if (premises.empty()) throw QMathError("Horn-logic rule requires at least one premise");
        Rule result;
        result.premises.assign(premises.begin(), premises.end());
        for (const Literal literal : result.premises) require_literal(literal);
        std::sort(result.premises.begin(), result.premises.end(), [](Literal lhs, Literal rhs) {
            if (lhs.atom != rhs.atom) return lhs.atom < rhs.atom;
            return lhs.value < rhs.value;
        });
        result.premises.erase(
            std::unique(result.premises.begin(), result.premises.end()),
            result.premises.end());
        for (std::size_t i = 1U; i < result.premises.size(); ++i) {
            if (result.premises[i - 1U].atom == result.premises[i].atom &&
                result.premises[i - 1U].value != result.premises[i].value) {
                throw QMathError("Horn-logic rule contains contradictory premises");
            }
        }
        result.conclusion = conclusion;
        return result;
    }

    [[nodiscard]] static std::size_t premise_terms(std::span<const Rule> rules) noexcept {
        std::size_t count = 0U;
        for (const Rule& rule : rules) count += rule.premises.size();
        return count;
    }

    [[nodiscard]] static bool known(const Compiled& compiled, Literal literal) noexcept {
        return compiled.known[literal_index(literal)] != 0U;
    }

    [[nodiscard]] Compiled compile(
        std::span<const Literal> facts,
        std::span<const Rule> rules) const {
        if (facts.size() > config_.max_input_facts || rules.size() > config_.max_rules ||
            premise_terms(rules) > config_.max_premise_terms) {
            throw QMathError("Horn-logic compile limits exceeded");
        }
        Compiled result;
        result.known.assign(atoms_.size() * 2U, 0U);
        std::deque<std::size_t> queue;
        const auto admit = [&](Literal literal) {
            const std::size_t index = literal_index(literal);
            const std::size_t opposite = literal_index(Literal{literal.atom, !literal.value});
            if (result.known[opposite] != 0U) {
                throw QMathError("Horn-logic closure is contradictory");
            }
            if (result.known[index] == 0U) {
                result.known[index] = 1U;
                queue.push_back(index);
            }
        };
        for (const Literal fact : facts) admit(fact);

        std::vector<std::size_t> remaining(rules.size(), 0U);
        std::vector<std::vector<std::size_t>> watchers(atoms_.size() * 2U);
        for (std::size_t rule_index = 0U; rule_index < rules.size(); ++rule_index) {
            const Rule& rule = rules[rule_index];
            remaining[rule_index] = rule.premises.size();
            for (const Literal premise : rule.premises) {
                watchers[literal_index(premise)].push_back(rule_index);
            }
        }
        while (!queue.empty()) {
            const std::size_t literal = queue.front();
            queue.pop_front();
            for (const std::size_t rule_index : watchers[literal]) {
                if (remaining[rule_index] == 0U) continue;
                --remaining[rule_index];
                if (remaining[rule_index] == 0U) admit(rules[rule_index].conclusion);
            }
        }

        for (AtomId atom = 0U; atom < atoms_.size(); ++atom) {
            const bool negative = result.known[literal_index(Literal{atom, false})] != 0U;
            const bool positive = result.known[literal_index(Literal{atom, true})] != 0U;
            if (negative && positive) throw QMathError("Horn-logic closure is contradictory");
            result.known_false += static_cast<std::size_t>(negative);
            result.known_true += static_cast<std::size_t>(positive);
        }
        result.known_literals = result.known_true + result.known_false;
        result.canonical = canonical_state(facts, rules, result);
        return result;
    }

    [[nodiscard]] std::string literal_canonical(Literal literal) const {
        const std::string& atom = atoms_[literal.atom];
        return std::to_string(atom.size()) + ':' + atom + '=' + (literal.value ? "1" : "0");
    }

    [[nodiscard]] std::string rule_canonical(const Rule& rule) const {
        std::vector<std::string> premises;
        premises.reserve(rule.premises.size());
        for (const Literal literal : rule.premises) premises.push_back(literal_canonical(literal));
        std::sort(premises.begin(), premises.end());
        std::string out;
        for (const std::string& premise : premises) {
            if (!out.empty()) out += '&';
            out += premise;
        }
        out += "->";
        out += literal_canonical(rule.conclusion);
        return out;
    }

    [[nodiscard]] std::string canonical_state(
        std::span<const Literal> facts,
        std::span<const Rule> rules,
        const Compiled& compiled) const {
        std::vector<std::string> atoms = atoms_;
        std::sort(atoms.begin(), atoms.end());
        std::vector<std::string> fact_rows;
        fact_rows.reserve(facts.size());
        for (const Literal fact : facts) fact_rows.push_back(literal_canonical(fact));
        std::sort(fact_rows.begin(), fact_rows.end());
        std::vector<std::string> rule_rows;
        rule_rows.reserve(rules.size());
        for (const Rule& rule : rules) rule_rows.push_back(rule_canonical(rule));
        std::sort(rule_rows.begin(), rule_rows.end());
        std::vector<std::string> closure_rows;
        closure_rows.reserve(compiled.known_literals);
        for (AtomId atom = 0U; atom < atoms_.size(); ++atom) {
            if (known(compiled, Literal{atom, false})) closure_rows.push_back(literal_canonical(Literal{atom, false}));
            if (known(compiled, Literal{atom, true})) closure_rows.push_back(literal_canonical(Literal{atom, true}));
        }
        std::sort(closure_rows.begin(), closure_rows.end());

        std::string out = "horn-logic:v1:A{";
        for (const std::string& atom : atoms) out += std::to_string(atom.size()) + ':' + atom + ';';
        out += "}F{";
        for (const std::string& row : fact_rows) out += row + ';';
        out += "}R{";
        for (const std::string& row : rule_rows) out += row + ';';
        out += "}C{";
        for (const std::string& row : closure_rows) out += row + ';';
        out += '}';
        return out;
    }

    void ensure_compiled() const {
        if (!compiled_.has_value()) compiled_ = compile(facts_, rules_);
    }
};

[[nodiscard]] inline const char* qlogic_truth_name(QLogicTruth truth) noexcept {
    switch (truth) {
        case QLogicTruth::False:
            return "False";
        case QLogicTruth::True:
            return "True";
        case QLogicTruth::Unknown:
            return "Unknown";
    }
    return "unknown";
}

}  // namespace qubit
