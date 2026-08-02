cmake_minimum_required(VERSION 3.20)
project(QSAPauliSupport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(QSTATE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(QSTATE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(QSTATE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(QSTATE_BUILD_PYTHON_TESTS OFF CACHE BOOL "" FORCE)
set(QSTATE_INSTALL_DEVELOPMENT OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." qsa)

enable_testing()

add_executable(qstate_pauli_support_test test_pauli_support.cpp)
target_link_libraries(qstate_pauli_support_test PRIVATE QSA::qstate_core)
add_test(NAME qstate_pauli_support_test COMMAND qstate_pauli_support_test)

add_executable(qstate_pauli_support_benchmark benchmark_pauli_support.cpp)
target_link_libraries(qstate_pauli_support_benchmark PRIVATE QSA::qstate_core)
