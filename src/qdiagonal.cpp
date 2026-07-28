#include "qubit/qdiagonal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace qubit {
namespace {

[[nodiscard]] bool finite(QComplex value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

template <class PhaseRange>
[[nodiscard]] QComplex coefficient_for(
    BasisIndex basis,
    const PhaseRange& phases) {
    QComplex coefficient{1.0, 0.0};
    for (const auto& phase : phases) {
        coefficient *= ((basis >> phase.position) & 1U) == 0U
            ? phase.zero
            : phase.one;
    }
    return coefficient;
}

}  // namespace

void CompiledDiagonalPlan::validate_config(
    const CompiledDiagonalConfig& config) {
    if (config.max_coefficient_bytes == 0U) {
        throw QStateError("Compiled diagonal coefficient limit must be positive");
    }
    if (!std::isfinite(config.unitary_tolerance) ||
        config.unitary_tolerance <= 0.0) {
        throw QStateError("Compiled diagonal unitary tolerance must be finite and positive");
    }
}

void CompiledDiagonalPlan::validate_phase(
    const QDiagonalPhase& phase,
    const CompiledDiagonalConfig& config) {
    if (!finite(phase.zero) || !finite(phase.one)) {
        throw QStateError("Compiled diagonal phase contains a non-finite value");
    }
    if (std::abs(phase.zero.norm2() - 1.0) > config.unitary_tolerance ||
        std::abs(phase.one.norm2() - 1.0) > config.unitary_tolerance) {
        throw QStateError("Compiled diagonal phase must be unit magnitude");
    }
}

CompiledDiagonalPlan::CompiledDiagonalPlan(
    const QRegister& prototype,
    std::span<const QDiagonalPhase> phases,
    CompiledDiagonalConfig config)
    : phase_count_(phases.size()), config_(config) {
    validate_config(config_);
    if (phases.empty()) {
        return;
    }

    std::unordered_map<QubitId, QDiagonalPhase> combined;
    combined.reserve(phases.size());
    for (const QDiagonalPhase& phase : phases) {
        prototype.validate_qubit(phase.qubit);
        validate_phase(phase, config_);
        const auto [iterator, inserted] = combined.emplace(phase.qubit, phase);
        if (!inserted) {
            iterator->second.zero = phase.zero * iterator->second.zero;
            iterator->second.one = phase.one * iterator->second.one;
            validate_phase(iterator->second, config_);
        }
    }

    std::unordered_map<std::size_t, std::size_t> plan_by_component;
    plan_by_component.reserve(combined.size());
    for (const auto& [qubit, phase] : combined) {
        const std::size_t component_index = prototype.component_index(qubit);
        auto [iterator, inserted] = plan_by_component.emplace(
            component_index, components_.size());
        if (inserted) {
            const StateComponent& component = prototype.components_[component_index];
            ComponentPlan plan;
            plan.qubits = component.qubits;
            plan.kind = component.is_cell()
                ? ComponentKind::Cell
                : (std::get<AmplitudeStore>(component.state).mode_ == StorageMode::Dense
                       ? ComponentKind::Dense
                       : ComponentKind::Sparse);
            components_.push_back(std::move(plan));
        }
        ComponentPlan& plan = components_[iterator->second];
        const auto position = std::find(plan.qubits.begin(), plan.qubits.end(), qubit);
        if (position == plan.qubits.end()) {
            throw QStateError("Compiled diagonal prototype mapping is inconsistent");
        }
        plan.phases.push_back(LocalPhase{
            static_cast<std::size_t>(std::distance(plan.qubits.begin(), position)),
            phase.zero,
            phase.one,
        });
    }

    std::size_t coefficient_bytes_value = 0U;
    for (ComponentPlan& plan : components_) {
        std::sort(plan.phases.begin(), plan.phases.end(), [](const LocalPhase& left,
                                                            const LocalPhase& right) {
            return left.position < right.position;
        });
        if (plan.kind != ComponentKind::Dense) {
            continue;
        }
        if (plan.qubits.size() >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("Compiled diagonal dense component is too wide");
        }
        const std::size_t dimension = std::size_t{1} << plan.qubits.size();
        if (dimension > std::numeric_limits<std::size_t>::max() / sizeof(QComplex)) {
            throw QStateError("Compiled diagonal coefficient size overflows");
        }
        const std::size_t bytes = dimension * sizeof(QComplex);
        if (bytes > config_.max_coefficient_bytes ||
            coefficient_bytes_value > config_.max_coefficient_bytes - bytes) {
            throw QStateError("Compiled diagonal coefficients exceed configured memory limit");
        }
        coefficient_bytes_value += bytes;
        plan.coefficients.resize(dimension);
        for (std::size_t basis = 0; basis < dimension; ++basis) {
            plan.coefficients[basis] = coefficient_for(
                static_cast<BasisIndex>(basis), plan.phases);
        }
    }
}

std::size_t CompiledDiagonalPlan::coefficient_bytes() const noexcept {
    std::size_t result = 0U;
    for (const ComponentPlan& component : components_) {
        result += component.coefficients.capacity() * sizeof(QComplex);
    }
    return result;
}

void CompiledDiagonalPlan::execute(QRegister& state) const {
    for (const ComponentPlan& plan : components_) {
        if (plan.qubits.empty()) {
            continue;
        }
        const std::size_t index = state.component_index(plan.qubits.front());
        const StateComponent& component = state.components_[index];
        if (component.qubits != plan.qubits) {
            throw QStateError("Compiled diagonal component structure changed");
        }
        const ComponentKind kind = component.is_cell()
            ? ComponentKind::Cell
            : (std::get<AmplitudeStore>(component.state).mode_ == StorageMode::Dense
                   ? ComponentKind::Dense
                   : ComponentKind::Sparse);
        if (kind != plan.kind) {
            throw QStateError("Compiled diagonal component storage changed");
        }
        if (kind == ComponentKind::Dense &&
            std::get<AmplitudeStore>(component.state).dense_.size() !=
                plan.coefficients.size()) {
            throw QStateError("Compiled diagonal dense dimension changed");
        }
    }

    for (const ComponentPlan& plan : components_) {
        if (plan.qubits.empty()) {
            continue;
        }
        const std::size_t index = state.component_index(plan.qubits.front());
        StateComponent& component = state.components_[index];
        const ComponentKind kind = plan.kind;

        if (kind == ComponentKind::Cell) {
            if (plan.phases.size() != 1U || plan.phases.front().position != 0U) {
                throw QStateError("Compiled diagonal cell plan is inconsistent");
            }
            QMatrix2 matrix{};
            matrix.values[0] = plan.phases.front().zero;
            matrix.values[3] = plan.phases.front().one;
            state.apply_cell_matrix(std::get<BlochCell>(component.state), matrix);
            continue;
        }

        AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
        if (kind == ComponentKind::Dense) {
            for (std::size_t basis = 0; basis < store.dense_.size(); ++basis) {
                store.dense_[basis] *= plan.coefficients[basis];
            }
        } else {
            for (auto& [basis, amplitude] : store.sparse_) {
                amplitude *= coefficient_for(basis, plan.phases);
            }
        }
    }
}

void CompiledDiagonalPlan::execute_many(
    std::span<QRegister* const> states,
    std::size_t worker_count) const {
    if (states.empty()) {
        return;
    }
    for (QRegister* state : states) {
        if (state == nullptr) {
            throw QStateError("Compiled diagonal state list contains a null register");
        }
    }
    if (worker_count == 0U) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (worker_count == 0U) {
            worker_count = 1U;
        }
    }
    worker_count = std::min(worker_count, states.size());
    if (worker_count <= 1U) {
        for (QRegister* state : states) {
            execute(*state);
        }
        return;
    }

    std::atomic<std::size_t> next{0U};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    const auto worker = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
            if (index >= states.size()) {
                return;
            }
            try {
                execute(*states[index]);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                }
                stop.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) {
        thread.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace qubit
