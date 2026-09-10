#pragma once

#include "qubit/qplan.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qubit {

enum class PauliAxis : std::uint8_t {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
};

struct PauliFactor {
    QubitId qubit{0};
    PauliAxis axis{PauliAxis::I};
};

struct PauliTerm {
    QComplex coefficient{1.0, 0.0};
    std::vector<PauliFactor> factors{};
};

struct PauliPropagationConfig {
    std::size_t max_terms{4'096};
};

struct PauliPropagationStats {
    std::size_t source_operations{0};
    std::size_t visited_operations{0};
    std::size_t peak_terms{0};
    std::size_t peak_support{0};
};

class PauliPropagationPlan;

class PauliObservable {
public:
    explicit PauliObservable(
        std::size_t qubit_count,
        PauliPropagationConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] std::size_t support_size() const;
    [[nodiscard]] const PauliPropagationConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<PauliTerm>& terms() const noexcept { return terms_; }

    void add_term(QComplex coefficient, std::span<const PauliFactor> factors = {});
    void clear() noexcept { terms_.clear(); }

    void propagate_backward(std::span<const Operation> operations);
    [[nodiscard]] PauliObservable propagated_backward(
        std::span<const Operation> operations) const;

    [[nodiscard]] QComplex expectation(const QRegister& state) const;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;

private:
    friend class PauliPropagationPlan;

    std::size_t qubit_count_{0};
    PauliPropagationConfig config_{};
    std::vector<PauliTerm> terms_{};

    void normalize_terms();
    void apply_backward(const Operation& operation);
};

class PauliPropagationPlan {
public:
    PauliPropagationPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations);

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t source_operation_count() const noexcept {
        return operations_.size();
    }
    [[nodiscard]] std::size_t indexed_reference_count() const noexcept {
        return indexed_reference_count_;
    }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    [[nodiscard]] PauliObservable propagate_backward(
        const PauliObservable& observable,
        PauliPropagationStats* stats = nullptr) const;
    [[nodiscard]] std::optional<PauliObservable> try_propagate_backward(
        const PauliObservable& observable,
        const char** reason = nullptr,
        PauliPropagationStats* stats = nullptr) const;

private:
    std::size_t qubit_count_{0};
    std::vector<Operation> operations_{};
    std::vector<std::vector<std::size_t>> operation_indices_{};
    std::size_t indexed_reference_count_{0};
};

[[nodiscard]] const char* pauli_axis_name(PauliAxis axis) noexcept;

}  // namespace qubit