#include "qubit/qmath_language.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qubit;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    QHornLogic logic;
    const auto powered = logic.add_atom("powered");
    const auto sensor_ok = logic.add_atom("sensor_ok");
    const auto ready = logic.add_atom("ready");
    const auto safe = logic.add_atom("safe");
    const auto launch = logic.add_atom("launch");
    const auto override_enabled = logic.add_atom("override_enabled");
    const auto unrelated = logic.add_atom("unrelated");

    require(logic.add_fact(powered, true) && logic.add_fact(sensor_ok, true) &&
            logic.add_fact(override_enabled, false),
            "Horn facts were not admitted");

    const std::vector<QHornLogic::Literal> ready_premises{
        {powered, true},
        {sensor_ok, true},
    };
    require(logic.add_rule(ready_premises, {ready, true}),
            "Horn conjunction rule was not admitted");

    const std::vector<QHornLogic::Literal> safe_premises{{ready, true}};
    require(logic.add_rule(safe_premises, {safe, true}),
            "Horn chained rule was not admitted");

    const std::vector<QHornLogic::Literal> launch_premises{
        {safe, true},
        {override_enabled, false},
    };
    require(logic.add_rule(launch_premises, {launch, true}),
            "Horn negative-premise rule was not admitted");

    require(logic.truth(powered) == QLogicTruth::True &&
            logic.truth(override_enabled) == QLogicTruth::False &&
            logic.truth(ready) == QLogicTruth::True &&
            logic.truth(safe) == QLogicTruth::True &&
            logic.truth(launch) == QLogicTruth::True &&
            logic.truth(unrelated) == QLogicTruth::Unknown,
            "Horn closure produced the wrong true/false/unknown state");
    require(logic.entails({launch, true}) && !logic.entails({launch, false}),
            "Horn literal entailment is wrong");

    const QHornLogicReceipt receipt = logic.receipt();
    require(receipt.atoms == 7U && receipt.input_facts == 3U && receipt.rules == 3U &&
            receipt.premise_terms == 5U && receipt.known_literals == 6U &&
            receipt.derived_literals == 3U && receipt.known_true == 5U &&
            receipt.known_false == 1U && receipt.exact && receipt.consistent,
            "Horn receipt lost exact closure accounting");

    const std::string before_conflict = logic.canonical();
    const std::vector<QHornLogic::Literal> contradictory_premise{{launch, true}};
    bool rejected = false;
    try {
        (void)logic.add_rule(contradictory_premise, {powered, false});
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected && logic.canonical() == before_conflict && logic.rule_count() == 3U,
            "Horn contradiction rejection mutated accepted state");

    require(!logic.add_rule(ready_premises, {ready, true}) && logic.rule_count() == 3U,
            "Horn duplicate rule changed accepted state");
    require(!logic.add_fact(powered, true) && logic.input_fact_count() == 3U,
            "Horn duplicate fact changed accepted state");

    rejected = false;
    try {
        const std::vector<QHornLogic::Literal> impossible{{powered, true}, {powered, false}};
        (void)logic.add_rule(impossible, {unrelated, true});
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected && logic.truth(unrelated) == QLogicTruth::Unknown,
            "Horn contradictory premises were accepted or mutated state");

    QHornLogic reordered;
    const auto reordered_unrelated = reordered.add_atom("unrelated");
    const auto reordered_override = reordered.add_atom("override_enabled");
    const auto reordered_launch = reordered.add_atom("launch");
    const auto reordered_safe = reordered.add_atom("safe");
    const auto reordered_ready = reordered.add_atom("ready");
    const auto reordered_sensor = reordered.add_atom("sensor_ok");
    const auto reordered_powered = reordered.add_atom("powered");
    require(reordered.add_fact(reordered_override, false) &&
            reordered.add_fact(reordered_sensor, true) &&
            reordered.add_fact(reordered_powered, true),
            "reordered Horn facts were not admitted");
    const std::vector<QHornLogic::Literal> reordered_launch_premises{
        {reordered_override, false},
        {reordered_safe, true},
    };
    const std::vector<QHornLogic::Literal> reordered_safe_premises{{reordered_ready, true}};
    const std::vector<QHornLogic::Literal> reordered_ready_premises{
        {reordered_sensor, true},
        {reordered_powered, true},
    };
    require(reordered.add_rule(reordered_launch_premises, {reordered_launch, true}) &&
            reordered.add_rule(reordered_safe_premises, {reordered_safe, true}) &&
            reordered.add_rule(reordered_ready_premises, {reordered_ready, true}) &&
            reordered.canonical() == logic.canonical() &&
            reordered.truth(reordered_unrelated) == QLogicTruth::Unknown,
            "Horn canonical identity depends on atom, fact, premise, or rule insertion order");

    const QMathLanguageCompilation language = QMathLanguageCompiler::compile(logic);
    require(language.receipt.route == QMathLanguageRoute::HornLogic &&
            language.receipt.evidence.fidelity == QMathFidelity::ExactStructural &&
            language.receipt.evidence.exact_structure() &&
            !language.receipt.evidence.exact_math() &&
            language.receipt.type.has_value() &&
            language.receipt.type->scalar == QMathScalar::Boolean &&
            language.receipt.sites == 7U && language.receipt.source_terms == 6U &&
            language.receipt.support_terms == 6U && language.receipt.transform_ready &&
            language.horn_logic.has_value() &&
            language.horn_logic->canonical == logic.canonical() &&
            language.receipt.dependencies == std::vector<std::string>({
                "launch", "override_enabled", "powered", "ready", "safe", "sensor_ok", "unrelated"}) &&
            std::string(qmath_language_route_name(language.receipt.route)) == "HornLogic",
            "QMath language lost Horn identity, Boolean type, dependencies, or exact structural evidence");

    QHornLogic capped(QHornLogicConfig{2U, 2U, 1U, 1U});
    const auto cap_a = capped.add_atom("cap:A");
    const auto cap_b = capped.add_atom("cap:B");
    rejected = false;
    try { (void)capped.add_atom("cap:C"); } catch (const QMathError&) { rejected = true; }
    require(rejected, "Horn atom cap did not fail closed");
    require(capped.add_fact(cap_a, true), "Horn capped fact admission failed");
    const std::vector<QHornLogic::Literal> capped_premises{{cap_a, true}};
    require(capped.add_rule(capped_premises, {cap_b, true}), "Horn capped rule admission failed");
    rejected = false;
    try {
        const std::vector<QHornLogic::Literal> second_rule{{cap_b, true}};
        (void)capped.add_rule(second_rule, {cap_a, true});
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "Horn rule cap did not fail closed");

    std::cout << "QHorn logic tests passed\n";
}
