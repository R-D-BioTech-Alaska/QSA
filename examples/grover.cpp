#include "qubit/qgrover.hpp"

#include <array>
#include <iostream>

int main() {
    const std::array<qubit::BasisIndex, 1> marked{731};
    qubit::GroverSearch search(20, marked);

    std::cout << "Logical states: " << search.space_size() << '\n';
    std::cout << "Optimal iterations: " << search.optimal_iterations() << '\n';
    search.run_optimal();
    std::cout << "Success probability: " << search.success_probability() << '\n';
    std::cout << "Sampled basis state: " << search.sample_basis(0x515341ULL) << '\n';
    std::cout << "Engine bytes: " << search.estimated_bytes() << '\n';
    return 0;
}
