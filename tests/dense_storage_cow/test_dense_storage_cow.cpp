#include "dense_fixture.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    constexpr std::size_t group_count = 4U;
    constexpr std::size_t group_width = 10U;
    qubit::QRegister root =
        qsa_dense_cow_test::make_dense_components(group_count, group_width);
    const std::vector<std::uint8_t> root_qsc = root.encode_qsc();

    for (std::size_t group = 0; group < group_count; ++group) {
        const auto qubit = static_cast<qubit::QubitId>(group * group_width);
        require(
            root.component_storage_owner_count(qubit) == 1L,
            "fresh dense component has shared storage");
    }

    qubit::QRegister branch = root;
    for (std::size_t group = 0; group < group_count; ++group) {
        const auto qubit = static_cast<qubit::QubitId>(group * group_width);
        require(
            root.component_storage_owner_count(qubit) == 2L,
            "root dense component was not shared after register copy");
        require(
            branch.component_storage_owner_count(qubit) == 2L,
            "branch dense component was not shared after register copy");
    }

    const auto first = static_cast<qubit::QubitId>(0U);
    branch.apply_rz(first, 0.271);
    require(
        root.component_storage_owner_count(first) == 1L,
        "root retained the detached branch payload");
    require(
        branch.component_storage_owner_count(first) == 1L,
        "mutated dense component did not detach");

    for (std::size_t group = 1; group < group_count; ++group) {
        const auto qubit = static_cast<qubit::QubitId>(group * group_width);
        require(
            root.component_storage_owner_count(qubit) == 2L,
            "untouched root component detached early");
        require(
            branch.component_storage_owner_count(qubit) == 2L,
            "untouched branch component detached early");
    }

    require(root.encode_qsc() == root_qsc, "branch mutation changed the root QSC");

    qubit::QRegister control = qubit::QRegister::decode_qsc(root_qsc);
    control.apply_rz(first, 0.271);
    require(
        branch.encode_qsc() == control.encode_qsc(),
        "dense copy-on-write changed the exact branch result");

    const auto second = static_cast<qubit::QubitId>(group_width);
    branch.apply_ry(second, -0.193);
    require(
        root.component_storage_owner_count(second) == 1L,
        "root retained the second detached payload");
    require(
        branch.component_storage_owner_count(second) == 1L,
        "second mutated dense component did not detach");

    for (std::size_t group = 2; group < group_count; ++group) {
        const auto qubit = static_cast<qubit::QubitId>(group * group_width);
        require(
            root.component_storage_owner_count(qubit) == 2L &&
                branch.component_storage_owner_count(qubit) == 2L,
            "unmodified dense payload was copied");
    }

    std::string reason;
    require(root.validate(&reason), "root register failed validation");
    require(branch.validate(&reason), "branch register failed validation");
    require(control.validate(&reason), "control register failed validation");

    std::cout << "QSA dense storage copy-on-write tests passed.\n";
    return 0;
}
