#include "test_qdot_support.hpp"

#include <thread>
#include <vector>

using namespace qubit::qdot;

std::vector<double> run_worker(std::size_t worker) {
    PocketConfig config;
    config.dot_count = 9;
    config.topology = Topology::PairPlusContext;
    config.dt += 0.001 * static_cast<double>(worker);
    QuantumDotPocket pocket(config);
    std::vector<DotInput> inputs(config.dot_count);
    for (std::size_t step = 0; step < 700; ++step) {
        for (std::size_t dot = 0; dot < inputs.size(); ++dot) {
            inputs[dot] = DotInput{
                0.12 + 0.003 * static_cast<double>(step) + 0.02 * static_cast<double>(dot),
                -0.21 + 0.004 * static_cast<double>(worker) - 0.01 * static_cast<double>(dot),
                0.4 + 0.5 * std::abs(
                                  std::sin(0.01 * static_cast<double>(step + dot + worker))),
            };
        }
        pocket.step(inputs);
    }
    require(pocket.validate(), "thread worker produced invalid state");
    return pocket.probabilities_one();
}

int main() {
    constexpr std::size_t workers = 12;
    std::vector<std::vector<double>> parallel(workers);
    std::vector<std::thread> threads;
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] { parallel[worker] = run_worker(worker); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const auto serial = run_worker(worker);
        require(parallel[worker] == serial, "parallel and serial results differ");
    }
    std::cout << "qdot concurrency and determinism tests passed\n";
    return 0;
}
