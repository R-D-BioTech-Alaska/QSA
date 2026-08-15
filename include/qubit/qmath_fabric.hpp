#pragma once

#include "qubit/qmath_language.hpp"
#include "qubit/qrepresentation_compiler.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct QMathFabricChannelBinding {
    std::string dependency{};
    QubitId channel{0U};

    friend bool operator==(const QMathFabricChannelBinding&, const QMathFabricChannelBinding&) = default;
};

struct QMathFabricBindingReceipt {
    QMathLanguageRoute route{QMathLanguageRoute::Symbolic};
    QMathFidelity fidelity{QMathFidelity::Unverified};
    ExactRepresentationDependencyType dependency_type{0U};
    std::vector<QMathFabricChannelBinding> bindings{};
    std::size_t component{0U};
    std::size_t generation{0U};
    std::size_t numerical_generation{0U};
    std::size_t declared_dependencies_before{0U};
    std::size_t declared_dependencies_after{0U};
    std::size_t component_merges{0U};
    bool prior_component_known{false};
    bool numerical_generation_preserved{false};
    bool exact{false};
    bool structural_dependency{true};
    std::string source_canonical{};
};

class QMathRepresentationFabricBridge {
public:
    [[nodiscard]] static QMathFabricBindingReceipt bind(
        ExactRepresentationFabric& fabric,
        const QMathLanguageCompilation& source,
        std::span<const QMathFabricChannelBinding> bindings) {
        require_supported(source);
        if (bindings.size() != source.receipt.dependencies.size() || bindings.size() < 2U) {
            throw QMathError("QMath fabric binding must cover every source dependency exactly once");
        }

        std::vector<QubitId> support;
        support.reserve(bindings.size());
        for (std::size_t index = 0U; index < bindings.size(); ++index) {
            if (bindings[index].dependency != source.receipt.dependencies[index]) {
                throw QMathError("QMath fabric binding dependency order or identity changed");
            }
            support.push_back(bindings[index].channel);
        }
        std::vector<QubitId> normalized = support;
        std::sort(normalized.begin(), normalized.end());
        if (std::adjacent_find(normalized.begin(), normalized.end()) != normalized.end()) {
            throw QMathError("QMath fabric binding maps distinct dependencies onto one channel");
        }

        const ExactRepresentationDependencyType type = dependency_type(source.receipt.route);
        const std::vector<ExactRepresentationDependencyReceipt> before_dependencies = fabric.dependency_receipts();
        for (const ExactRepresentationDependencyReceipt& receipt : before_dependencies) {
            if (receipt.type == type && receipt.support == normalized) {
                throw QMathError("QMath fabric dependency is already bound");
            }
        }
        const std::optional<ExactRepresentationComponentReceipt> prior =
            component_for_support(fabric, before_dependencies, normalized);
        const ExactRepresentationFabricStats before = fabric.stats();

        fabric.declare_dependency(type, support);

        const ExactRepresentationFabricStats after = fabric.stats();
        const std::vector<ExactRepresentationDependencyReceipt> dependency_receipts = fabric.dependency_receipts();
        const ExactRepresentationDependencyReceipt* added = nullptr;
        for (const ExactRepresentationDependencyReceipt& receipt : dependency_receipts) {
            if (receipt.type == type && receipt.support == normalized) {
                if (added != nullptr) {
                    throw QMathError("QMath fabric dependency identity became ambiguous");
                }
                added = &receipt;
            }
        }
        if (added == nullptr || after.declared_dependencies != before.declared_dependencies + 1U) {
            throw QMathError("QMath fabric did not retain the declared typed dependency");
        }

        QMathFabricBindingReceipt result;
        result.route = source.receipt.route;
        result.fidelity = source.receipt.evidence.fidelity;
        result.dependency_type = type;
        result.bindings.assign(bindings.begin(), bindings.end());
        result.component = added->component;
        result.generation = added->generation;
        result.numerical_generation = added->numerical_generation;
        result.declared_dependencies_before = before.declared_dependencies;
        result.declared_dependencies_after = after.declared_dependencies;
        result.component_merges = after.component_merges - before.component_merges;
        result.prior_component_known = prior.has_value();
        result.numerical_generation_preserved = prior.has_value() &&
            prior->component == added->component &&
            prior->numerical_generation == added->numerical_generation;
        result.exact = source.receipt.exact;
        result.source_canonical = source.receipt.canonical;
        return result;
    }

    [[nodiscard]] static constexpr ExactRepresentationDependencyType dependency_type(
        QMathLanguageRoute route) {
        switch (route) {
            case QMathLanguageRoute::StrictOrder:
                return 0x514D415448000001ULL;
            case QMathLanguageRoute::AffineRelation:
                return 0x514D415448000002ULL;
            case QMathLanguageRoute::HornLogic:
                return 0x514D415448000003ULL;
            default:
                throw QMathError("QMath language route has no persistent fabric dependency contract");
        }
    }

private:
    static void require_supported(const QMathLanguageCompilation& source) {
        if (!source.receipt.exact || !source.receipt.transform_ready ||
            !source.receipt.evidence.exact_structure()) {
            throw QMathError("QMath fabric requires exact structured mathematical-language evidence");
        }
        (void)dependency_type(source.receipt.route);
    }

    [[nodiscard]] static std::optional<ExactRepresentationComponentReceipt> component_for_support(
        const ExactRepresentationFabric& fabric,
        std::span<const ExactRepresentationDependencyReceipt> dependencies,
        std::span<const QubitId> support) {
        std::optional<std::size_t> component;
        for (const QubitId channel : support) {
            std::optional<std::size_t> channel_component;
            for (const ExactRepresentationDependencyReceipt& dependency : dependencies) {
                if (std::binary_search(dependency.support.begin(), dependency.support.end(), channel)) {
                    if (channel_component.has_value() && *channel_component != dependency.component) {
                        throw QMathError("QMath fabric dependency receipts disagree on channel component");
                    }
                    channel_component = dependency.component;
                }
            }
            if (!channel_component.has_value()) return std::nullopt;
            if (component.has_value() && *component != *channel_component) return std::nullopt;
            component = channel_component;
        }
        if (!component.has_value()) return std::nullopt;
        for (const ExactRepresentationComponentReceipt& receipt : fabric.component_receipts()) {
            if (receipt.component == *component) return receipt;
        }
        throw QMathError("QMath fabric component receipt is missing");
    }
};

}  // namespace qubit
