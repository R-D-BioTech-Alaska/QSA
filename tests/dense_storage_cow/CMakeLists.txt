cmake_minimum_required(VERSION 3.20)
project(QSADenseStorageCow LANGUAGES CXX)

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

add_executable(qstate_dense_storage_cow_test test_dense_storage_cow.cpp)
target_link_libraries(qstate_dense_storage_cow_test PRIVATE QSA::qstate_core)
add_test(NAME qstate_dense_storage_cow_test COMMAND qstate_dense_storage_cow_test)

add_executable(qstate_dense_storage_cow_benchmark benchmark_dense_storage_cow.cpp)
target_link_libraries(qstate_dense_storage_cow_benchmark PRIVATE QSA::qstate_core)
