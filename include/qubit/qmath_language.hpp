#pragma once

#include "qubit/qclifford3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

enum class QMathLanguageRoute : std::uint8_t {
    Symbolic = 0,
    WeylStructural = 1,
    QubitWeylNative = 2,
    QutritCyclotomicWeyl = 3,
    QutritClifford = 4,
};

enum class QMathFidelity : std::uint8_t {
    Unverified = 0,
    ExactAlgebraic = 1,
    ExactStructural = 2,
    BoundedNumerical = 3,
};

struct QMathErrorBounds {
    double absolute{0.0};
    double relative{0.0};

    friend bool operator==(const QMathErrorBounds&, const QMathErrorBounds&) = default;
};

struct QMathEvidence {
    QMathFidelity fidelity{QMathFidelity::Unverified};
    std::optional<QMathErrorBounds> error_bounds{};
    bool source_identity_preserved{false};
    bool type_preserved{false};
    bool physical_dimensions_preserved{false};
    bool coefficient_arithmetic_exact{false};
    bool phase_arithmetic_exact{false};

    [[nodiscard]] static constexpr QMathEvidence exact_algebraic(
        bool phase_exact = false) noexcept {
        return QMathEvidence{
            QMathFidelity::ExactAlgebraic,
            std::nullopt,
            true,
            true,
            true,
            true,
            phase_exact,
        };
    }

    [[nodiscard]] static constexpr QMathEvidence exact_structural() noexcept {
        return QMathEvidence{
            QMathFidelity::ExactStructural,
            std::nullopt,
            true,
            true,
            true,
            false,
            false,
        };
    }

    [[nodiscard]] static QMathEvidence bounded_numerical(
        double absolute_error,
        double relative_error) {
        if (!std::isfinite(absolute_error) || !std::isfinite(relative_error) ||
            absolute_error < 0.0 || relative_error < 0.0) {
            throw QMathError("bounded numerical evidence requires finite nonnegative error bounds");
        }
        return QMathEvidence{
            QMathFidelity::BoundedNumerical,
            QMathErrorBounds{absolute_error, relative_error},
            true,
            true,
            true,
            false,
            false,
        };
    }

    [[nodiscard]] constexpr bool exact_math() const noexcept {
        return fidelity == QMathFidelity::ExactAlgebraic &&
               source_identity_preserved && type_preserved && physical_dimensions_preserved &&
               coefficient_arithmetic_exact;
    }

    [[nodiscard]] constexpr bool exact_structure() const noexcept {
        return (fidelity == QMathFidelity::ExactAlgebraic || fidelity == QMathFidelity::ExactStructural) &&
               source_identity_preserved && type_preserved && physical_dimensions_preserved;
    }

    [[nodiscard]] bool bounded_numeric() const noexcept {
        return fidelity == QMathFidelity::BoundedNumerical && error_bounds.has_value();
    }
};

static_assert(QMathEvidence::exact_algebraic().exact_math());
static_assert(QMathEvidence::exact_structural().exact_structure());
static_assert(!QMathEvidence::exact_structural().exact_math());

struct QMathLanguageReceipt {
    QMathLanguageRoute route{QMathLanguageRoute::Symbolic};
    QMathEvidence evidence{};
    std::optional<QMathType> type{};
    std::vector<std::uint32_t> local_dimensions{};
    std::vector<std::string> dependencies{};
    std::size_t sites{0U};
    std::size_t source_terms{0U};
    std::size_t support_terms{0U};
    std::size_t program_steps{0U};
    std::size_t generator_images{0U};
    std::size_t lowered_operations{0U};
    QRational global_phase_turns{0};
    std::string canonical{};
    bool exact{false};
    bool symbolic_ready{false};
    bool native_lowering_ready{false};
    bool transform_ready{false};
};

struct QMathLanguageCompilation {
    QMathLanguageReceipt receipt{};
    QMathExpr symbolic{};
    std::optional<QWeylQubitLowering> qubit_lowering{};
    std::optional<QWeyl3Algebra> qutrit_algebra{};
    std::optional<QClifford3Map> qutrit_clifford{};
};

class QMathLanguageCompiler {
public:
    [[nodiscard]] static QMathLanguageCompilation compile(
        const QMathExpr& expression,
        const QMathArena& arena) {
        if (!expression) throw QMathError("mathematical language cannot compile a null expression");
        QMathLanguageCompilation result;
        result.symbolic = expression;
        result.receipt.route = QMathLanguageRoute::Symbolic;
        result.receipt.evidence = exact_algebraic_expression(expression)
            ? QMathEvidence::exact_algebraic()
            : QMathEvidence::exact_structural();
        result.receipt.type = expression->type;
        result.receipt.dependencies = arena.dependencies(expression);
        result.receipt.source_terms = 1U;
        result.receipt.support_terms = result.receipt.dependencies.size();
        result.receipt.canonical = expression->canonical;
        result.receipt.exact = result.receipt.evidence.exact_structure();
        result.receipt.symbolic_ready = true;
        return result;
    }

    [[nodiscard]] static QMathLanguageCompilation compile(
        const QWeylOperator& value,
        QMathArena& arena) {
        QMathLanguageCompilation result;
        const QWeylMathProjection projection = project_weyl_qmath(value, arena);
        result.symbolic = projection.expression;
        fill_weyl_receipt(result.receipt, value);
        result.receipt.type = projection.expression->type;
        result.receipt.global_phase_turns = projection.global_phase_turns;
        result.receipt.evidence = QMathEvidence::exact_algebraic(true);
        result.receipt.exact = result.receipt.evidence.exact_math();
        result.receipt.symbolic_ready = true;

        if (uniform_dimension(value.space(), 2U)) {
            result.receipt.route = QMathLanguageRoute::QubitWeylNative;
            result.qubit_lowering = lower_weyl_qubits(value);
            result.receipt.lowered_operations = result.qubit_lowering->operations.size();
            result.receipt.global_phase_turns = result.qubit_lowering->global_phase_turns;
            result.receipt.native_lowering_ready = true;
            return result;
        }

        if (uniform_dimension(value.space(), 3U)) {
            try {
                QWeyl3Algebra algebra(value.space());
                algebra.add(value);
                result.receipt.route = QMathLanguageRoute::QutritCyclotomicWeyl;
                result.receipt.source_terms = algebra.term_count();
                result.receipt.transform_ready = true;
                result.qutrit_algebra = std::move(algebra);
                return result;
            } catch (const QMathError&) {
            }
        }

        result.receipt.route = QMathLanguageRoute::WeylStructural;
        return result;
    }

    [[nodiscard]] static QMathLanguageCompilation compile(
        const QWeyl3Algebra& algebra,
        QMathArena& arena) {
        QMathLanguageCompilation result;
        result.receipt.route = QMathLanguageRoute::QutritCyclotomicWeyl;
        result.receipt.evidence = QMathEvidence::exact_algebraic(true);
        result.receipt.local_dimensions = algebra.space().dimensions();
        result.receipt.sites = algebra.space().site_count();
        result.receipt.source_terms = algebra.term_count();
        for (const QWeyl3Term& term : algebra.terms()) {
            result.receipt.support_terms += term.basis.support_size();
        }
        const QWeylOperator identity = QWeylOperator::identity(algebra.space());
        result.receipt.type = project_weyl_qmath(identity, arena).expression->type;
        result.receipt.canonical = algebra.canonical();
        result.receipt.exact = result.receipt.evidence.exact_math();
        result.receipt.transform_ready = true;
        result.qutrit_algebra = algebra;
        return result;
    }

    [[nodiscard]] static QMathLanguageCompilation compile(
        const QClifford3Map& map,
        QMathArena& arena) {
        QMathLanguageCompilation result;
        result.receipt.route = QMathLanguageRoute::QutritClifford;
        result.receipt.evidence = QMathEvidence::exact_algebraic(true);
        result.receipt.local_dimensions = map.space().dimensions();
        result.receipt.sites = map.space().site_count();
        result.receipt.generator_images = map.symplectic().dimension();
        const QWeylOperator identity = QWeylOperator::identity(map.space());
        result.receipt.type = project_weyl_qmath(identity, arena).expression->type;
        result.receipt.canonical = map.canonical();
        result.receipt.exact = result.receipt.evidence.exact_math();
        result.receipt.transform_ready = true;
        result.qutrit_clifford = map;
        return result;
    }

    [[nodiscard]] static QMathLanguageCompilation compile(
        const QClifford3Program& program,
        QMathArena& arena) {
        QClifford3CompileReceipt source;
        const QClifford3Map map = program.compile(&source);
        QMathLanguageCompilation result = compile(map, arena);
        result.receipt.program_steps = source.steps;
        result.receipt.generator_images = source.generator_images;
        return result;
    }

private:
    [[nodiscard]] static bool exact_algebraic_expression(const QMathExpr& expression) noexcept {
        switch (expression->kind) {
            case QMathKind::Zero:
            case QMathKind::One:
            case QMathKind::Rational:
                return true;
            case QMathKind::Sum:
            case QMathKind::Product:
                return std::all_of(
                    expression->args.begin(),
                    expression->args.end(),
                    [](const QMathExpr& value) { return exact_algebraic_expression(value); });
            default:
                return false;
        }
    }

    [[nodiscard]] static bool uniform_dimension(
        const QWeylSpace& space,
        std::uint32_t dimension) noexcept {
        return std::all_of(
            space.dimensions().begin(),
            space.dimensions().end(),
            [dimension](std::uint32_t value) { return value == dimension; });
    }

    static void fill_weyl_receipt(
        QMathLanguageReceipt& receipt,
        const QWeylOperator& value) {
        receipt.route = QMathLanguageRoute::WeylStructural;
        receipt.local_dimensions = value.space().dimensions();
        receipt.sites = value.space().site_count();
        receipt.source_terms = 1U;
        receipt.support_terms = value.support_size();
        receipt.global_phase_turns = value.phase_turns();
        receipt.canonical = value.canonical();
    }
};

[[nodiscard]] inline const char* qmath_language_route_name(QMathLanguageRoute route) noexcept {
    switch (route) {
        case QMathLanguageRoute::Symbolic:
            return "QMathSymbolic";
        case QMathLanguageRoute::WeylStructural:
            return "WeylStructural";
        case QMathLanguageRoute::QubitWeylNative:
            return "QubitWeylNative";
        case QMathLanguageRoute::QutritCyclotomicWeyl:
            return "QutritCyclotomicWeyl";
        case QMathLanguageRoute::QutritClifford:
            return "QutritClifford";
    }
    return "unknown";
}

[[nodiscard]] inline const char* qmath_fidelity_name(QMathFidelity fidelity) noexcept {
    switch (fidelity) {
        case QMathFidelity::Unverified:
            return "Unverified";
        case QMathFidelity::ExactAlgebraic:
            return "ExactAlgebraic";
        case QMathFidelity::ExactStructural:
            return "ExactStructural";
        case QMathFidelity::BoundedNumerical:
            return "BoundedNumerical";
    }
    return "unknown";
}

}  // namespace qubit
