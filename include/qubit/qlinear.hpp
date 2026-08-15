#pragma once

#include "qubit/qmath.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

enum class QLinearStatus : std::uint8_t {
    Unique = 0,
    Underdetermined = 1,
    Inconsistent = 2,
};

struct QExactLinearConfig {
    std::size_t max_variables{256U};
    std::size_t max_equations{4'096U};
    std::size_t max_cells{1U << 20U};
};

struct QExactLinearResult {
    QLinearStatus status{QLinearStatus::Underdetermined};
    QMathType variable_type{};
    std::vector<std::string> variables{};
    std::size_t equations{0U};
    std::size_t rank{0U};
    std::size_t augmented_rank{0U};
    std::vector<std::size_t> pivot_columns{};
    std::vector<QRational> rref{};
    std::vector<QRational> solution{};
    std::string canonical{};
    bool exact{true};

    [[nodiscard]] bool unique() const noexcept {
        return status == QLinearStatus::Unique;
    }

    [[nodiscard]] bool consistent() const noexcept {
        return status != QLinearStatus::Inconsistent;
    }
};

class QExactLinearSystem {
public:
    explicit QExactLinearSystem(
        std::vector<std::string> variables,
        QMathType variable_type = QMathType::scalar_type(QMathScalar::Rational),
        QExactLinearConfig config = {})
        : variables_(std::move(variables)),
          variable_type_(std::move(variable_type)),
          config_(config) {
        if (variables_.empty() || variables_.size() > config_.max_variables ||
            config_.max_variables == 0U || config_.max_equations == 0U || config_.max_cells == 0U) {
            throw QMathError("exact linear-system limits or variable count are invalid");
        }
        if (variable_type_.space != QMathSpace::Scalar ||
            variable_type_.scalar == QMathScalar::Boolean) {
            throw QMathError("exact linear-system variables require a non-Boolean scalar type");
        }
        if (!std::is_sorted(variables_.begin(), variables_.end()) ||
            std::adjacent_find(variables_.begin(), variables_.end()) != variables_.end()) {
            throw QMathError("exact linear-system variable identities must be sorted and unique");
        }
        if (std::any_of(variables_.begin(), variables_.end(), [](const std::string& value) {
                return value.empty();
            })) {
            throw QMathError("exact linear-system variable identity is empty");
        }
    }

    void add_equation(std::span<const QRational> coefficients, const QRational& rhs) {
        if (coefficients.size() != variables_.size()) {
            throw QMathError("exact linear-system coefficient width mismatch");
        }
        if (equation_count_ >= config_.max_equations) {
            throw QMathError("exact linear-system equation cap exceeded");
        }
        const std::size_t width = variables_.size() + 1U;
        if (width > config_.max_cells || equation_count_ + 1U > config_.max_cells / width) {
            throw QMathError("exact linear-system cell cap exceeded");
        }
        matrix_.insert(matrix_.end(), coefficients.begin(), coefficients.end());
        matrix_.push_back(rhs);
        ++equation_count_;
    }

    [[nodiscard]] std::size_t variable_count() const noexcept {
        return variables_.size();
    }

    [[nodiscard]] std::size_t equation_count() const noexcept {
        return equation_count_;
    }

    [[nodiscard]] const QMathType& variable_type() const noexcept {
        return variable_type_;
    }

    [[nodiscard]] std::span<const std::string> variables() const noexcept {
        return variables_;
    }

    [[nodiscard]] QExactLinearResult solve() const {
        QExactLinearResult result;
        result.variable_type = variable_type_;
        result.variables = variables_;
        result.equations = equation_count_;

        const std::size_t columns = variables_.size();
        const std::size_t width = columns + 1U;
        result.rref = matrix_;
        std::size_t pivot_row = 0U;
        for (std::size_t column = 0U; column < columns && pivot_row < equation_count_; ++column) {
            std::size_t selected = pivot_row;
            while (selected < equation_count_ &&
                   result.rref[selected * width + column].is_zero()) {
                ++selected;
            }
            if (selected == equation_count_) continue;
            if (selected != pivot_row) {
                for (std::size_t offset = 0U; offset < width; ++offset) {
                    std::swap(
                        result.rref[pivot_row * width + offset],
                        result.rref[selected * width + offset]);
                }
            }

            const QRational pivot = result.rref[pivot_row * width + column];
            for (std::size_t offset = column; offset < width; ++offset) {
                result.rref[pivot_row * width + offset] =
                    result.rref[pivot_row * width + offset] / pivot;
            }

            for (std::size_t row = 0U; row < equation_count_; ++row) {
                if (row == pivot_row) continue;
                const QRational factor = result.rref[row * width + column];
                if (factor.is_zero()) continue;
                for (std::size_t offset = column; offset < width; ++offset) {
                    result.rref[row * width + offset] =
                        result.rref[row * width + offset] -
                        factor * result.rref[pivot_row * width + offset];
                }
            }
            result.pivot_columns.push_back(column);
            ++pivot_row;
        }

        result.rank = result.pivot_columns.size();
        bool inconsistent = false;
        for (std::size_t row = 0U; row < equation_count_; ++row) {
            bool zero_coefficients = true;
            for (std::size_t column = 0U; column < columns; ++column) {
                if (!result.rref[row * width + column].is_zero()) {
                    zero_coefficients = false;
                    break;
                }
            }
            if (zero_coefficients && !result.rref[row * width + columns].is_zero()) {
                inconsistent = true;
                break;
            }
        }
        result.augmented_rank = result.rank + static_cast<std::size_t>(inconsistent);
        if (inconsistent) {
            result.status = QLinearStatus::Inconsistent;
        } else if (result.rank == columns) {
            result.status = QLinearStatus::Unique;
            result.solution.assign(columns, QRational(0));
            for (std::size_t row = 0U; row < result.pivot_columns.size(); ++row) {
                result.solution[result.pivot_columns[row]] = result.rref[row * width + columns];
            }
        } else {
            result.status = QLinearStatus::Underdetermined;
        }
        result.canonical = canonical(result, width);
        return result;
    }

private:
    std::vector<std::string> variables_{};
    QMathType variable_type_{};
    QExactLinearConfig config_{};
    std::vector<QRational> matrix_{};
    std::size_t equation_count_{0U};

    [[nodiscard]] static std::string canonical(
        const QExactLinearResult& result,
        std::size_t width) {
        std::string out = "exact-linear:v1:";
        out += result.variable_type.canonical();
        out += ":V{";
        for (const std::string& variable : result.variables) {
            out += std::to_string(variable.size());
            out += ':';
            out += variable;
            out += ';';
        }
        out += "}:S=";
        out += std::to_string(static_cast<unsigned>(result.status));
        out += ":R=" + std::to_string(result.rank);
        out += ":AR=" + std::to_string(result.augmented_rank);
        out += ":M{";
        for (std::size_t row = 0U; row < result.equations; ++row) {
            bool zero_row = true;
            for (std::size_t column = 0U; column < width; ++column) {
                if (!result.rref[row * width + column].is_zero()) {
                    zero_row = false;
                    break;
                }
            }
            if (zero_row) continue;
            for (std::size_t column = 0U; column < width; ++column) {
                if (column != 0U) out += ',';
                out += result.rref[row * width + column].canonical();
            }
            out += ';';
        }
        out += '}';
        return out;
    }
};

[[nodiscard]] inline const char* qlinear_status_name(QLinearStatus status) noexcept {
    switch (status) {
        case QLinearStatus::Unique:
            return "Unique";
        case QLinearStatus::Underdetermined:
            return "Underdetermined";
        case QLinearStatus::Inconsistent:
            return "Inconsistent";
    }
    return "unknown";
}

}  // namespace qubit
