#include "test_qdot_support.hpp"

#include <limits>
#include <string>
#include <utility>

using namespace qubit;
using namespace qubit::qdot;

template <typename Function>
bool throws_qstate(Function&& function) {
    try {
        function();
    } catch (const QStateError&) {
        return true;
    }
    return false;
}

int main() {
    require(throws_qstate([] { PocketConfig c; c.dot_count = 0; QuantumDotPocket p(c); }),
            "zero dots must be rejected");
    require(throws_qstate([] { PocketConfig c; c.dt = 0.0; QuantumDotPocket p(c); }),
            "zero dt must be rejected");
    require(throws_qstate([] { PocketConfig c; c.dt = std::numeric_limits<double>::infinity(); QuantumDotPocket p(c); }),
            "infinite dt must be rejected");
    require(throws_qstate([] { PocketConfig c; c.dt = std::numeric_limits<double>::quiet_NaN(); QuantumDotPocket p(c); }),
            "NaN dt must be rejected");
    require(throws_qstate([] { PocketConfig c; c.trotter_steps = 0; QuantumDotPocket p(c); }),
            "zero trotter steps must be rejected");
    require(throws_qstate([] { PocketConfig c; c.trotter_steps = 1'000'001; QuantumDotPocket p(c); }),
            "excessive trotter steps must be rejected");
    require(throws_qstate([] { PocketConfig c; c.dot.charge_drive = std::numeric_limits<double>::quiet_NaN(); QuantumDotPocket p(c); }),
            "NaN dot configuration must be rejected");
    require(throws_qstate([] { PocketConfig c; c.coupling.scale = std::numeric_limits<double>::infinity(); QuantumDotPocket p(c); }),
            "infinite coupling configuration must be rejected");
    require(throws_qstate([] { PocketConfig c; c.dot_count = 4; c.topology = Topology::PairPlusContext; QuantumDotPocket p(c); }),
            "PairPlusContext must reject even dot counts");
    require(throws_qstate([] { PocketConfig c; c.dot_count = 3; c.topology = Topology::PairBlocks; QuantumDotPocket p(c); }),
            "PairBlocks must reject odd dot counts");
    require(throws_qstate([] { PocketConfig c; c.dot_count = 2; c.topology = Topology::Chain; QuantumDotPocket p(c); }),
            "specialized pocket must reject chain topology");
    {
        PocketConfig c;
        c.dot_count = 3;
        c.topology = Topology::Chain;
        QRegister reference(6);
        std::vector<DotInput> chain_inputs(3);
        apply_reference_step(reference, c, chain_inputs);
        require(reference.validate(), "generic QSA chain reference must remain available");
    }

    PocketConfig context_config;
    context_config.dot_count = 5;
    context_config.topology = Topology::PairPlusContext;
    QuantumDotPocket context(context_config);
    require(context.component_count() == 3,
            "five-dot pair-plus-context must have two pairs and one context block");
    require(context.max_component_qubits() == 4,
            "pair block must remain four logical qubits");
    require(context.validate(), "fresh pair-plus-context pocket must validate");

    PocketConfig block_config;
    block_config.dot_count = 6;
    block_config.topology = Topology::PairBlocks;
    QuantumDotPocket blocks(block_config);
    require(blocks.component_count() == 3, "six-dot pair blocks must contain three blocks");
    require(blocks.validate(), "fresh pair-block pocket must validate");

    std::vector<DotInput> inputs(context_config.dot_count);
    inputs[0] = {0.7, -0.2, 0.8};
    context.step(inputs);
    const auto evolved = context.materialize(12);
    context.reset();
    const auto reset = context.materialize(12);
    require(evolved != reset, "reset test requires an evolved state");
    require_near(reset.front().re, 1.0, 0.0, "reset must restore |0...0>");
    for (std::size_t i = 1; i < reset.size(); ++i) {
        require(reset[i].norm2() == 0.0, "reset must clear all nonzero basis states");
    }

    QuantumDotPocket exception_safe(context_config);
    const auto before = exception_safe.materialize(12);
    auto bad = inputs;
    bad[2].theta = std::numeric_limits<double>::quiet_NaN();
    require(throws_qstate([&] { exception_safe.step(bad); }), "NaN input must be rejected");
    require(exception_safe.materialize(12) == before,
            "rejected input must not partially mutate state");
    require(exception_safe.validate(), "state must remain valid after rejected input");
    bad = inputs;
    bad[1].strength = -0.01;
    require(throws_qstate([&] { exception_safe.step(bad); }),
            "negative strength must be rejected");
    bad[1].strength = 1.01;
    require(throws_qstate([&] { exception_safe.step(bad); }),
            "strength above one must be rejected");
    require(throws_qstate([&] {
                exception_safe.step(std::span<const DotInput>(inputs.data(), inputs.size() - 1));
            }),
            "wrong input count must be rejected");

    {
        PocketConfig overflow = context_config;
        overflow.dt = std::numeric_limits<double>::max();
        overflow.dot.charge_drive = std::numeric_limits<double>::max();
        QuantumDotPocket guarded(overflow);
        const auto guarded_before = guarded.materialize(12);
        require(throws_qstate([&] { guarded.step(inputs); }),
                "finite values whose derived angles overflow must be rejected");
        require(guarded.materialize(12) == guarded_before,
                "derived-angle rejection must happen before state mutation");
        require(guarded.validate(), "overflow rejection must preserve a valid state");
    }
    {
        PocketConfig overflow = context_config;
        overflow.dt = std::numeric_limits<double>::max();
        overflow.coupling.scale = std::numeric_limits<double>::max();
        QuantumDotPocket guarded(overflow);
        const auto guarded_before = guarded.materialize(12);
        require(throws_qstate([&] { guarded.step(inputs); }),
                "finite coupling values whose product overflows must be rejected");
        require(guarded.materialize(12) == guarded_before,
                "coupling-overflow rejection must happen before state mutation");
    }

    QuantumDotPocket original(context_config);
    original.step(inputs);
    QuantumDotPocket copy = original;
    require(copy.materialize(12) == original.materialize(12),
            "copy must preserve exact state");
    auto changed = inputs;
    changed[0].theta += 0.3;
    copy.step(changed);
    require(copy.materialize(12) != original.materialize(12),
            "copied pockets must be independent");
    QuantumDotPocket moved = std::move(copy);
    require(moved.validate(), "moved pocket must remain valid");

    require(throws_qstate([&] { (void)original.materialize(2); }),
            "materialization limit must be enforced");

    std::cout << "qdot safety tests passed\n";
    return 0;
}
