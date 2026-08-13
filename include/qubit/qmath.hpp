#pragma once

#include "qubit/qexact.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

struct QPhysicalDimension {
    enum Axis : std::size_t {
        Length = 0,
        Mass = 1,
        Time = 2,
        Current = 3,
        Temperature = 4,
        Amount = 5,
        LuminousIntensity = 6,
    };

    std::array<QRational, 7> exponent{};

    [[nodiscard]] static QPhysicalDimension dimensionless() { return {}; }
    [[nodiscard]] static QPhysicalDimension base(Axis axis) {
        QPhysicalDimension value;
        value.exponent[axis] = QRational(1);
        return value;
    }

    [[nodiscard]] QPhysicalDimension multiplied(const QPhysicalDimension& rhs) const {
        QPhysicalDimension result;
        for (std::size_t i = 0; i < exponent.size(); ++i) result.exponent[i] = exponent[i] + rhs.exponent[i];
        return result;
    }

    [[nodiscard]] QPhysicalDimension divided(const QPhysicalDimension& rhs) const {
        QPhysicalDimension result;
        for (std::size_t i = 0; i < exponent.size(); ++i) result.exponent[i] = exponent[i] - rhs.exponent[i];
        return result;
    }

    [[nodiscard]] QPhysicalDimension powered(const QRational& power) const {
        QPhysicalDimension result;
        for (std::size_t i = 0; i < exponent.size(); ++i) result.exponent[i] = exponent[i] * power;
        return result;
    }

    [[nodiscard]] bool is_dimensionless() const {
        return std::all_of(exponent.begin(), exponent.end(), [](const QRational& value) { return value.is_zero(); });
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "[";
        for (std::size_t i = 0; i < exponent.size(); ++i) {
            if (i != 0) out += ',';
            out += exponent[i].canonical();
        }
        out += ']';
        return out;
    }

    friend bool operator==(const QPhysicalDimension&, const QPhysicalDimension&) = default;
};

enum class QMathScalar : std::uint8_t {
    Boolean = 0,
    Integer = 1,
    Rational = 2,
    Real = 3,
    Complex = 4,
};

enum class QMathSpace : std::uint8_t {
    Scalar = 0,
    Vector = 1,
    Matrix = 2,
    Tensor = 3,
    Operator = 4,
    State = 5,
    Field = 6,
};

struct QMathType {
    QMathScalar scalar{QMathScalar::Real};
    QMathSpace space{QMathSpace::Scalar};
    std::vector<std::size_t> shape{};
    QPhysicalDimension dimension{};

    [[nodiscard]] static QMathType scalar_type(
        QMathScalar scalar_kind = QMathScalar::Real,
        QPhysicalDimension physical_dimension = {}) {
        return QMathType{scalar_kind, QMathSpace::Scalar, {}, std::move(physical_dimension)};
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = std::to_string(static_cast<unsigned>(scalar)) + ":" +
                          std::to_string(static_cast<unsigned>(space)) + ":";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) out += 'x';
            out += std::to_string(shape[i]);
        }
        out += ':' + dimension.canonical();
        return out;
    }

    friend bool operator==(const QMathType&, const QMathType&) = default;
};

enum class QMathKind : std::uint8_t {
    Zero = 0,
    One = 1,
    Rational = 2,
    Symbol = 3,
    Sum = 4,
    Product = 5,
    Power = 6,
    Function = 7,
    Equation = 8,
    Derivative = 9,
    Integral = 10,
    Commutator = 11,
    TensorProduct = 12,
    Apply = 13,
};

struct QMathNode;
using QMathExpr = std::shared_ptr<const QMathNode>;

struct QMathNode {
    QMathKind kind{QMathKind::Zero};
    QMathType type{};
    QRational rational{};
    std::string label{};
    std::vector<QMathExpr> args{};
    bool commutative{true};
    std::size_t depth{1};
    std::uint64_t structural_hash{0};
    std::string canonical{};
};

struct QMathConfig {
    std::size_t max_nodes{1U << 20};
    std::size_t max_arity{1U << 18};
    std::size_t max_depth{4096};
};

class QMathArena {
public:
    explicit QMathArena(QMathConfig config = {}) : config_(config) {
        if (config_.max_nodes == 0 || config_.max_arity == 0 || config_.max_depth == 0) {
            throw QMathError("QMath limits must be positive");
        }
    }

    [[nodiscard]] std::size_t node_count() const noexcept { return node_count_; }

    [[nodiscard]] QMathExpr zero(QMathType type = QMathType::scalar_type(QMathScalar::Rational)) {
        return leaf(QMathKind::Zero, std::move(type), {}, {}, true);
    }

    [[nodiscard]] QMathExpr one(QMathType type = QMathType::scalar_type(QMathScalar::Rational)) {
        return leaf(QMathKind::One, std::move(type), QRational(1), {}, true);
    }

    [[nodiscard]] QMathExpr rational(
        std::int64_t numerator,
        std::int64_t denominator = 1,
        QPhysicalDimension dimension = {}) {
        QMathType type = QMathType::scalar_type(QMathScalar::Rational, std::move(dimension));
        return exact_constant(QRational(numerator, denominator), std::move(type));
    }

    [[nodiscard]] QMathExpr rational(QRational value, QPhysicalDimension dimension = {}) {
        QMathType type = QMathType::scalar_type(QMathScalar::Rational, std::move(dimension));
        return exact_constant(value, std::move(type));
    }

    [[nodiscard]] QMathExpr symbol(std::string name, QMathType type, bool commutative = true) {
        if (name.empty()) throw QMathError("empty mathematical symbol");
        if (type.space != QMathSpace::Scalar) commutative = false;
        return leaf(QMathKind::Symbol, std::move(type), {}, std::move(name), commutative);
    }

    [[nodiscard]] QMathExpr add(std::initializer_list<QMathExpr> values) {
        return add(std::span<const QMathExpr>(values.begin(), values.size()));
    }

    [[nodiscard]] QMathExpr add(std::span<const QMathExpr> values) {
        if (values.empty()) return zero();
        std::vector<QMathExpr> terms;
        terms.reserve(values.size());
        for (const QMathExpr& value : values) {
            require_expr(value);
            if (value->kind == QMathKind::Sum) terms.insert(terms.end(), value->args.begin(), value->args.end());
            else terms.push_back(value);
        }
        check_arity(terms.size());
        QMathType result_type = terms.front()->type;
        for (std::size_t i = 1; i < terms.size(); ++i) result_type = add_type(result_type, terms[i]->type);

        QRational scalar_constant(0);
        bool have_scalar_constant = false;
        std::vector<QMathExpr> normalized;
        normalized.reserve(terms.size());
        for (const QMathExpr& term : terms) {
            if (is_zero(term)) continue;
            if ((term->kind == QMathKind::Rational || term->kind == QMathKind::One) &&
                term->type.space == QMathSpace::Scalar && term->type.dimension == result_type.dimension) {
                scalar_constant = scalar_constant + (term->kind == QMathKind::One ? QRational(1) : term->rational);
                have_scalar_constant = true;
            } else {
                normalized.push_back(term);
            }
        }
        if (have_scalar_constant && !scalar_constant.is_zero()) normalized.push_back(exact_constant(scalar_constant, result_type));
        if (normalized.empty()) return zero(result_type);
        if (normalized.size() == 1) return normalized.front();
        std::sort(normalized.begin(), normalized.end(), canonical_less);
        return branch(QMathKind::Sum, std::move(result_type), std::move(normalized), {}, true);
    }

    [[nodiscard]] QMathExpr negate(const QMathExpr& value) {
        require_expr(value);
        return multiply({rational(-1), value});
    }

    [[nodiscard]] QMathExpr multiply(std::initializer_list<QMathExpr> values) {
        return multiply(std::span<const QMathExpr>(values.begin(), values.size()));
    }

    [[nodiscard]] QMathExpr multiply(std::span<const QMathExpr> values) {
        if (values.empty()) return one();
        std::vector<QMathExpr> factors;
        factors.reserve(values.size());
        for (const QMathExpr& value : values) {
            require_expr(value);
            if (value->kind == QMathKind::Product) factors.insert(factors.end(), value->args.begin(), value->args.end());
            else factors.push_back(value);
        }
        check_arity(factors.size());
        QMathType result_type = product_type(factors);
        for (const QMathExpr& factor : factors) if (is_zero(factor)) return zero(result_type);

        QRational coefficient(1);
        QPhysicalDimension coefficient_dimension;
        bool have_coefficient = false;
        std::vector<QMathExpr> normalized;
        normalized.reserve(factors.size());
        for (const QMathExpr& factor : factors) {
            if (factor->kind == QMathKind::Rational || factor->kind == QMathKind::One) {
                coefficient = coefficient * (factor->kind == QMathKind::One ? QRational(1) : factor->rational);
                coefficient_dimension = coefficient_dimension.multiplied(factor->type.dimension);
                have_coefficient = true;
                continue;
            }
            normalized.push_back(factor);
        }
        if (normalized.empty()) return exact_constant(coefficient, result_type);
        if (have_coefficient && (!coefficient.is_one() || !coefficient_dimension.is_dimensionless())) normalized.push_back(rational(coefficient, coefficient_dimension));
        if (normalized.size() == 1 && normalized.front()->type == result_type) return normalized.front();

        const bool all_commutative = std::all_of(normalized.begin(), normalized.end(), [](const QMathExpr& value) { return value->commutative; });
        if (all_commutative) {
            std::sort(normalized.begin(), normalized.end(), canonical_less);
        } else {
            std::vector<QMathExpr> commuting;
            std::vector<QMathExpr> ordered;
            commuting.reserve(normalized.size());
            ordered.reserve(normalized.size());
            for (const QMathExpr& value : normalized) (value->commutative ? commuting : ordered).push_back(value);
            std::sort(commuting.begin(), commuting.end(), canonical_less);
            commuting.insert(commuting.end(), ordered.begin(), ordered.end());
            normalized = std::move(commuting);
        }
        return branch(QMathKind::Product, std::move(result_type), std::move(normalized), {}, all_commutative);
    }

    [[nodiscard]] QMathExpr power(const QMathExpr& base, const QRational& exponent) {
        require_expr(base);
        if (exponent.is_zero()) {
            QMathType identity_type = base->type;
            identity_type.dimension = {};
            return base->type.space == QMathSpace::Scalar ? one(QMathType::scalar_type(base->type.scalar)) : one(identity_type);
        }
        if (exponent.is_one()) return base;

        QMathType result_type = base->type;
        result_type.dimension = base->type.dimension.powered(exponent);
        std::int64_t count = 0;
        if ((base->kind == QMathKind::Rational || base->kind == QMathKind::One) && exponent.integer_int64(count)) {
            QRational value = base->kind == QMathKind::One ? QRational(1) : base->rational;
            const bool invert = count < 0;
            if (count == std::numeric_limits<std::int64_t>::min()) return branch(QMathKind::Power, std::move(result_type), {base}, exponent.canonical(), base->commutative);
            if (invert) count = -count;
            QRational result(1);
            QRational factor = value;
            while (count != 0) {
                if ((count & 1) != 0) result = result * factor;
                count >>= 1;
                if (count != 0) factor = factor * factor;
            }
            if (invert) result = QRational(1) / result;
            return rational(std::move(result), result_type.dimension);
        }
        return branch(QMathKind::Power, std::move(result_type), {base}, exponent.canonical(), base->commutative);
    }

    [[nodiscard]] QMathExpr function(std::string name, std::initializer_list<QMathExpr> args, QMathType result_type, bool commutative = true) {
        if (name.empty()) throw QMathError("empty mathematical function name");
        std::vector<QMathExpr> values(args.begin(), args.end());
        for (const QMathExpr& arg : values) require_expr(arg);
        check_arity(values.size());
        if (result_type.space != QMathSpace::Scalar) commutative = false;
        return branch(QMathKind::Function, std::move(result_type), std::move(values), std::move(name), commutative);
    }

    [[nodiscard]] QMathExpr equation(const QMathExpr& lhs, const QMathExpr& rhs) {
        require_expr(lhs);
        require_expr(rhs);
        (void)add_type(lhs->type, rhs->type);
        return branch(QMathKind::Equation, QMathType::scalar_type(QMathScalar::Boolean), {lhs, rhs}, {}, true);
    }

    [[nodiscard]] QMathExpr commutator(const QMathExpr& lhs, const QMathExpr& rhs) {
        require_expr(lhs);
        require_expr(rhs);
        if (lhs->type.space != QMathSpace::Operator || rhs->type.space != QMathSpace::Operator) throw QMathError("commutator requires operator expressions");
        QMathType result = product_type(std::vector<QMathExpr>{lhs, rhs});
        if (lhs->canonical == rhs->canonical) return zero(std::move(result));
        return branch(QMathKind::Commutator, std::move(result), {lhs, rhs}, {}, false);
    }

    [[nodiscard]] QMathExpr tensor_product(const QMathExpr& lhs, const QMathExpr& rhs) {
        require_expr(lhs);
        require_expr(rhs);
        QMathType result;
        result.scalar = promote(lhs->type.scalar, rhs->type.scalar);
        if (lhs->type.space == rhs->type.space && (lhs->type.space == QMathSpace::State || lhs->type.space == QMathSpace::Operator)) result.space = lhs->type.space;
        else result.space = QMathSpace::Tensor;
        result.shape = lhs->type.shape;
        result.shape.insert(result.shape.end(), rhs->type.shape.begin(), rhs->type.shape.end());
        result.dimension = lhs->type.dimension.multiplied(rhs->type.dimension);
        return branch(QMathKind::TensorProduct, std::move(result), {lhs, rhs}, {}, false);
    }

    [[nodiscard]] QMathExpr apply(const QMathExpr& op, const QMathExpr& value) {
        require_expr(op);
        require_expr(value);
        if (op->type.space != QMathSpace::Operator) throw QMathError("operator application requires operator lhs");
        QMathType result = value->type;
        result.scalar = promote(op->type.scalar, value->type.scalar);
        result.dimension = op->type.dimension.multiplied(value->type.dimension);
        return branch(QMathKind::Apply, std::move(result), {op, value}, {}, false);
    }

    [[nodiscard]] QMathExpr derivative(const QMathExpr& expression, const QMathExpr& variable) {
        require_expr(expression);
        require_symbol(variable);
        QMathType result_type = expression->type;
        result_type.dimension = expression->type.dimension.divided(variable->type.dimension);
        if (is_zero(expression) || is_one(expression) || expression->kind == QMathKind::Rational) return zero(result_type);
        if (expression->kind == QMathKind::Symbol) return expression->canonical == variable->canonical ? one(result_type) : zero(result_type);
        if (expression->kind == QMathKind::Sum) {
            std::vector<QMathExpr> values;
            values.reserve(expression->args.size());
            for (const QMathExpr& term : expression->args) values.push_back(derivative(term, variable));
            return add(values);
        }
        if (expression->kind == QMathKind::Product) {
            std::vector<QMathExpr> terms;
            for (std::size_t i = 0; i < expression->args.size(); ++i) {
                const QMathExpr diff = derivative(expression->args[i], variable);
                if (is_zero(diff)) continue;
                std::vector<QMathExpr> factors = expression->args;
                factors[i] = diff;
                terms.push_back(multiply(factors));
            }
            if (terms.empty()) return zero(result_type);
            return add(terms);
        }
        if (expression->kind == QMathKind::Power && expression->args.size() == 1) {
            const QRational exponent = parse_rational(expression->label);
            const QMathExpr diff = derivative(expression->args.front(), variable);
            if (is_zero(diff)) return zero(result_type);
            return multiply({rational(exponent), power(expression->args.front(), exponent - QRational(1)), diff});
        }
        return branch(QMathKind::Derivative, std::move(result_type), {expression, variable}, {}, expression->commutative);
    }

    [[nodiscard]] QMathExpr integral(const QMathExpr& expression, const QMathExpr& variable) {
        require_expr(expression);
        require_symbol(variable);
        QMathType result_type = expression->type;
        result_type.dimension = expression->type.dimension.multiplied(variable->type.dimension);
        if (is_zero(expression)) return zero(result_type);
        if (expression->kind == QMathKind::Sum) {
            std::vector<QMathExpr> values;
            values.reserve(expression->args.size());
            for (const QMathExpr& term : expression->args) values.push_back(integral(term, variable));
            return add(values);
        }
        if (expression->kind == QMathKind::Rational || expression->kind == QMathKind::One) return multiply({expression, variable});
        if (expression->kind == QMathKind::Symbol && expression->canonical == variable->canonical) return multiply({rational(1, 2), power(variable, QRational(2))});
        if (expression->kind == QMathKind::Power && expression->args.size() == 1 && expression->args.front()->canonical == variable->canonical) {
            const QRational exponent = parse_rational(expression->label);
            const QRational next = exponent + QRational(1);
            if (!next.is_zero()) return multiply({rational(QRational(1) / next), power(variable, next)});
        }
        return branch(QMathKind::Integral, std::move(result_type), {expression, variable}, {}, expression->commutative);
    }

    [[nodiscard]] std::vector<std::string> dependencies(const QMathExpr& expression) const {
        require_expr(expression);
        std::vector<std::string> values;
        collect_dependencies(expression, values);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    [[nodiscard]] bool equivalent(const QMathExpr& lhs, const QMathExpr& rhs) const {
        require_expr(lhs);
        require_expr(rhs);
        return lhs.get() == rhs.get() || lhs->canonical == rhs->canonical;
    }

private:
    [[nodiscard]] QMathExpr exact_constant(const QRational& value, QMathType type) {
        if (type.space != QMathSpace::Scalar) throw QMathError("exact constant requires scalar type");
        if (value.is_zero()) return zero(std::move(type));
        if (value.is_one() && type.dimension.is_dimensionless()) return one(std::move(type));
        return leaf(QMathKind::Rational, std::move(type), value, {}, true);
    }

    static void require_expr(const QMathExpr& value) { if (!value) throw QMathError("null mathematical expression"); }
    static void require_symbol(const QMathExpr& value) {
        require_expr(value);
        if (value->kind != QMathKind::Symbol) throw QMathError("operation requires symbol variable");
    }
    static bool is_zero(const QMathExpr& value) { return value->kind == QMathKind::Zero; }
    static bool is_one(const QMathExpr& value) { return value->kind == QMathKind::One; }
    static bool canonical_less(const QMathExpr& lhs, const QMathExpr& rhs) { return lhs->canonical < rhs->canonical; }
    static QMathScalar promote(QMathScalar lhs, QMathScalar rhs) {
        return static_cast<QMathScalar>(std::max(static_cast<unsigned>(lhs), static_cast<unsigned>(rhs)));
    }
    static QMathType add_type(const QMathType& lhs, const QMathType& rhs) {
        if (lhs.space != rhs.space || lhs.shape != rhs.shape || lhs.dimension != rhs.dimension) throw QMathError("incompatible mathematical addition");
        QMathType result = lhs;
        result.scalar = promote(lhs.scalar, rhs.scalar);
        return result;
    }
    static QMathType product_type(const std::vector<QMathExpr>& factors) {
        if (factors.empty()) return QMathType::scalar_type(QMathScalar::Rational);
        QMathScalar scalar = QMathScalar::Integer;
        QPhysicalDimension dimension;
        const QMathExpr* non_scalar = nullptr;
        bool all_operators = true;
        std::size_t non_scalar_count = 0;
        for (const QMathExpr& factor : factors) {
            scalar = promote(scalar, factor->type.scalar);
            dimension = dimension.multiplied(factor->type.dimension);
            if (factor->type.space != QMathSpace::Scalar) {
                ++non_scalar_count;
                if (non_scalar == nullptr) non_scalar = &factor;
                if (factor->type.space != QMathSpace::Operator) all_operators = false;
            }
        }
        if (non_scalar_count == 0) return QMathType::scalar_type(scalar, dimension);
        if (non_scalar_count == 1) {
            QMathType result = (*non_scalar)->type;
            result.scalar = scalar;
            result.dimension = dimension;
            return result;
        }
        if (!all_operators) throw QMathError("ambiguous non-scalar multiplication; use tensor_product or apply");
        QMathType result = (*non_scalar)->type;
        result.scalar = scalar;
        result.dimension = dimension;
        return result;
    }
    static QRational parse_rational(std::string_view text) { return QRational::parse(text); }
    static std::uint64_t hash64(std::string_view text) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char ch : text) {
            hash ^= static_cast<std::uint64_t>(ch);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
    void check_arity(std::size_t arity) const { if (arity > config_.max_arity) throw QMathError("QMath arity limit exceeded"); }
    QMathExpr leaf(QMathKind kind, QMathType type, QRational rational_value, std::string label, bool commutative) {
        std::string canonical = "qmath1:" + std::to_string(static_cast<unsigned>(kind)) + ':' + type.canonical() + ':' + rational_value.canonical() + ':' + label;
        return intern(kind, std::move(type), rational_value, std::move(label), {}, commutative, 1, std::move(canonical));
    }
    QMathExpr branch(QMathKind kind, QMathType type, std::vector<QMathExpr> args, std::string label, bool commutative) {
        check_arity(args.size());
        std::size_t depth = 1;
        for (const QMathExpr& arg : args) {
            require_expr(arg);
            depth = std::max(depth, arg->depth + 1);
        }
        if (depth > config_.max_depth) throw QMathError("QMath depth limit exceeded");
        std::string canonical = "qmath1:" + std::to_string(static_cast<unsigned>(kind)) + ':' + type.canonical() + ':' + label + '(';
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i != 0) canonical += ';';
            canonical += args[i]->canonical;
        }
        canonical += ')';
        return intern(kind, std::move(type), {}, std::move(label), std::move(args), commutative, depth, std::move(canonical));
    }
    QMathExpr intern(QMathKind kind, QMathType type, QRational rational_value, std::string label, std::vector<QMathExpr> args, bool commutative, std::size_t depth, std::string canonical) {
        const auto found = interned_.find(canonical);
        if (found != interned_.end()) {
            if (QMathExpr existing = found->second.lock()) return existing;
            interned_.erase(found);
        }
        if (node_count_ >= config_.max_nodes) throw QMathError("QMath node limit exceeded");
        auto node = std::make_shared<QMathNode>();
        node->kind = kind;
        node->type = std::move(type);
        node->rational = rational_value;
        node->label = std::move(label);
        node->args = std::move(args);
        node->commutative = commutative;
        node->depth = depth;
        node->structural_hash = hash64(canonical);
        node->canonical = std::move(canonical);
        QMathExpr result = node;
        interned_.emplace(result->canonical, result);
        ++node_count_;
        return result;
    }
    static void collect_dependencies(const QMathExpr& expression, std::vector<std::string>& out) {
        if (expression->kind == QMathKind::Symbol) out.push_back(expression->label);
        for (const QMathExpr& arg : expression->args) collect_dependencies(arg, out);
    }

    QMathConfig config_{};
    std::unordered_map<std::string, std::weak_ptr<const QMathNode>> interned_{};
    std::size_t node_count_{0};
};

}  // namespace qubit
