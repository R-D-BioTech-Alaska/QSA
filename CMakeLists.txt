cmake_minimum_required(VERSION 3.20)
project(QubitNativeStateEngine VERSION 0.1.6 LANGUAGES CXX)

find_package(Threads REQUIRED)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QSTATE_BUILD_TESTS "Build Qubit state-engine tests" ON)
option(QSTATE_BUILD_EXAMPLES "Build Qubit state-engine examples" ON)
option(QSTATE_BUILD_BENCHMARKS "Build Qubit state-engine benchmarks" ${QSTATE_BUILD_EXAMPLES})
option(QSTATE_BUILD_PYTHON_TESTS "Build Python bridge compatibility tests" ON)
option(QSTATE_ENABLE_IPO "Enable interprocedural optimization in release builds" ON)
option(QSTATE_NATIVE_ARCH "Optimize for the current CPU instead of portable wheels" OFF)
option(QSTATE_INSTALL_DEVELOPMENT "Install C++ headers, static library, and CMake package files" ON)

add_library(qstate_core STATIC
    src/qstate.cpp
    src/qstructure.cpp
    src/qadaptive.cpp
    src/qdot.cpp
    src/qdot_reference.cpp
    src/qsc.cpp
    src/qplan.cpp
    src/qgrover.cpp
    src/qsymmetry.cpp
    src/qsymmetry_components.cpp
    src/qrouter.cpp
    src/qstabilizer.cpp
    src/qphase_graph.cpp
    src/qexecution.cpp
    src/qread.cpp
)
set_target_properties(qstate_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(qstate_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(qstate_core PUBLIC Threads::Threads)
target_compile_options(qstate_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
)

if(QSTATE_NATIVE_ARCH)
    target_compile_options(qstate_core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native -mtune=native>
    )
endif()

add_library(qstate SHARED src/c_api.cpp)
target_link_libraries(qstate PRIVATE qstate_core)
target_include_directories(qstate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(qstate PRIVATE QSTATE_BUILD_SHARED)
set_target_properties(qstate PROPERTIES OUTPUT_NAME qstate)

add_library(QSA::qstate ALIAS qstate)
add_library(QSA::qstate_core ALIAS qstate_core)

if(QSTATE_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT QSTATE_IPO_SUPPORTED OUTPUT QSTATE_IPO_ERROR)
    if(QSTATE_IPO_SUPPORTED)
        set_property(TARGET qstate_core qstate PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(STATUS "QSA IPO unavailable: ${QSTATE_IPO_ERROR}")
    endif()
endif()

if(QSTATE_BUILD_TESTS)
    enable_testing()

    add_executable(qstate_tests tests/test_qstate.cpp)
    target_link_libraries(qstate_tests PRIVATE qstate_core)
    target_compile_definitions(qstate_tests PRIVATE
        QSA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME qstate_tests COMMAND qstate_tests)

    add_executable(qstate_c_api_tests tests/test_c_api.cpp)
    target_link_libraries(qstate_c_api_tests PRIVATE qstate)
    add_test(NAME qstate_c_api_tests COMMAND qstate_c_api_tests)

    add_executable(qstate_plan_tests tests/test_qplan.cpp)
    target_link_libraries(qstate_plan_tests PRIVATE qstate_core)
    add_test(NAME qstate_plan_tests COMMAND qstate_plan_tests)

    add_executable(qstate_grover_tests tests/test_grover.cpp)
    target_link_libraries(qstate_grover_tests PRIVATE qstate_core)
    add_test(NAME qstate_grover_tests COMMAND qstate_grover_tests)

    add_executable(qstate_symmetry_tests tests/test_symmetry.cpp)
    target_link_libraries(qstate_symmetry_tests PRIVATE qstate_core)
    add_test(NAME qstate_symmetry_tests COMMAND qstate_symmetry_tests)

    add_executable(qstate_qsc_adversarial_tests tests/test_qsc_adversarial.cpp)
    target_link_libraries(qstate_qsc_adversarial_tests PRIVATE qstate_core)
    add_test(NAME qstate_qsc_adversarial_tests COMMAND qstate_qsc_adversarial_tests)

    add_executable(qstate_qdot_safety_tests tests/test_qdot_safety.cpp)
    target_link_libraries(qstate_qdot_safety_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_safety_tests COMMAND qstate_qdot_safety_tests)

    add_executable(qstate_qdot_differential_tests tests/test_qdot_differential.cpp)
    target_link_libraries(qstate_qdot_differential_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_differential_tests COMMAND qstate_qdot_differential_tests)

    add_executable(qstate_qdot_stress_tests tests/test_qdot_stress.cpp)
    target_link_libraries(qstate_qdot_stress_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_stress_tests COMMAND qstate_qdot_stress_tests)

    add_executable(qstate_qdot_concurrency_tests tests/test_qdot_concurrency.cpp)
    target_link_libraries(qstate_qdot_concurrency_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_concurrency_tests COMMAND qstate_qdot_concurrency_tests)

    add_executable(qstate_qdot_robustness_tests tests/test_qdot_robustness.cpp)
    target_link_libraries(qstate_qdot_robustness_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_robustness_tests COMMAND qstate_qdot_robustness_tests)

    add_executable(qstate_qdot_no_regression_tests tests/test_qdot_no_regression.cpp)
    target_link_libraries(qstate_qdot_no_regression_tests PRIVATE qstate_core)
    add_test(NAME qstate_qdot_no_regression_tests COMMAND qstate_qdot_no_regression_tests)

    add_executable(qstate_structural_acceleration_tests tests/test_structural_acceleration.cpp)
    target_link_libraries(qstate_structural_acceleration_tests PRIVATE qstate_core)
    add_test(NAME qstate_structural_acceleration_tests COMMAND qstate_structural_acceleration_tests)

    add_executable(qstate_joint_measure_tests tests/test_joint_measure.cpp)
    target_link_libraries(qstate_joint_measure_tests PRIVATE qstate_core)
    add_test(NAME qstate_joint_measure_tests COMMAND qstate_joint_measure_tests)

    add_executable(qstate_adaptive_compaction_tests tests/test_adaptive_compaction.cpp)
    target_link_libraries(qstate_adaptive_compaction_tests PRIVATE qstate_core)
    add_test(NAME qstate_adaptive_compaction_tests COMMAND qstate_adaptive_compaction_tests)

    add_executable(qstate_representation_advisor_tests tests/test_representation_advisor.cpp)
    target_link_libraries(qstate_representation_advisor_tests PRIVATE qstate_core)
    add_test(NAME qstate_representation_advisor_tests COMMAND qstate_representation_advisor_tests)

    add_executable(qstate_stabilizer_tests tests/test_stabilizer.cpp)
    target_link_libraries(qstate_stabilizer_tests PRIVATE qstate_core)
    add_test(NAME qstate_stabilizer_tests COMMAND qstate_stabilizer_tests)

    add_executable(qstate_phase_graph_tests tests/test_phase_graph.cpp)
    target_link_libraries(qstate_phase_graph_tests PRIVATE qstate_core)
    add_test(NAME qstate_phase_graph_tests COMMAND qstate_phase_graph_tests)

    add_executable(qstate_execution_acceleration_tests tests/test_execution_acceleration.cpp)
    target_link_libraries(qstate_execution_acceleration_tests PRIVATE qstate_core)
    add_test(NAME qstate_execution_acceleration_tests COMMAND qstate_execution_acceleration_tests)

    add_executable(qstate_read_views_tests tests/test_read_views.cpp)
    target_link_libraries(qstate_read_views_tests PRIVATE qstate_core)
    add_test(NAME qstate_read_views_tests COMMAND qstate_read_views_tests)

    if(QSTATE_BUILD_PYTHON_TESTS)
        find_package(Python3 COMPONENTS Interpreter)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME qstate_python_qsa_env
                COMMAND ${CMAKE_COMMAND} -E env
                    --unset=QUBIT_NATIVE_LIB
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    "QSA_NATIVE_LIB=$<TARGET_FILE:qstate>"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_compat.py
            )
            add_test(
                NAME qstate_python_legacy_env
                COMMAND ${CMAKE_COMMAND} -E env
                    --unset=QSA_NATIVE_LIB
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    "QUBIT_NATIVE_LIB=$<TARGET_FILE:qstate>"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_compat.py
            )
            add_test(
                NAME qstate_contract_manifest
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_contract_manifest.py
            )
            add_test(
                NAME qstate_version_sync
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_version_sync.py
            )
            add_test(
                NAME qstate_python_grover_differential
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    "QSA_NATIVE_LIB=$<TARGET_FILE:qstate>"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/differential_grover.py
            )
            add_test(
                NAME qstate_python_symmetry_differential
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
                    "QSA_NATIVE_LIB=$<TARGET_FILE:qstate>"
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/differential_symmetry.py
            )
        endif()
    endif()
endif()

if(QSTATE_BUILD_EXAMPLES)
    add_executable(qstate_bell examples/bell.cpp)
    target_link_libraries(qstate_bell PRIVATE qstate_core)

    add_executable(qstate_grover examples/grover.cpp)
    target_link_libraries(qstate_grover PRIVATE qstate_core)

    add_executable(qstate_symmetry examples/symmetry.cpp)
    target_link_libraries(qstate_symmetry PRIVATE qstate_core)
endif()

if(QSTATE_BUILD_BENCHMARKS)
    add_executable(qstate_benchmark benchmarks/structured_states.cpp)
    target_link_libraries(qstate_benchmark PRIVATE qstate_core)

    add_executable(qstate_hot_paths benchmarks/hot_paths.cpp)
    target_link_libraries(qstate_hot_paths PRIVATE qstate_core)

    add_executable(qstate_plan_benchmark benchmarks/plans_and_ensembles.cpp)
    target_link_libraries(qstate_plan_benchmark PRIVATE qstate_core)

    add_executable(qstate_grover_benchmark benchmarks/grover_search.cpp)
    target_link_libraries(qstate_grover_benchmark PRIVATE qstate_core)

    add_executable(qstate_symmetry_benchmark benchmarks/symmetry_algebra.cpp)
    target_link_libraries(qstate_symmetry_benchmark PRIVATE qstate_core)

    add_executable(qstate_qdot_benchmark benchmarks/quantum_dot.cpp)
    target_link_libraries(qstate_qdot_benchmark PRIVATE qstate_core)

    add_executable(qstate_structural_acceleration_benchmark benchmarks/structural_acceleration.cpp)
    target_link_libraries(qstate_structural_acceleration_benchmark PRIVATE qstate_core)

    add_executable(qstate_adaptive_compaction_benchmark benchmarks/adaptive_compaction.cpp)
    target_link_libraries(qstate_adaptive_compaction_benchmark PRIVATE qstate_core)

    add_executable(qstate_representation_advisor_benchmark benchmarks/representation_advisor.cpp)
    target_link_libraries(qstate_representation_advisor_benchmark PRIVATE qstate_core)

    add_executable(qstate_stabilizer_benchmark benchmarks/stabilizer.cpp)
    target_link_libraries(qstate_stabilizer_benchmark PRIVATE qstate_core)

    add_executable(qstate_phase_graph_benchmark benchmarks/phase_graph.cpp)
    target_link_libraries(qstate_phase_graph_benchmark PRIVATE qstate_core)

    add_executable(qstate_execution_acceleration_benchmark benchmarks/execution_acceleration.cpp)
    target_link_libraries(qstate_execution_acceleration_benchmark PRIVATE qstate_core)

    add_executable(qstate_read_views_benchmark benchmarks/read_views.cpp)
    target_link_libraries(qstate_read_views_benchmark PRIVATE qstate_core)
endif()

include(CMakePackageConfigHelpers)

if(QSTATE_INSTALL_DEVELOPMENT)
    install(TARGETS qstate qstate_core
        EXPORT QSATargets
        LIBRARY DESTINATION qsa
        RUNTIME DESTINATION qsa
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    configure_package_config_file(
        cmake/QSAConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/QSAConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QSA
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/QSAConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(EXPORT QSATargets
        FILE QSATargets.cmake
        NAMESPACE QSA::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QSA
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/QSAConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/QSAConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QSA
    )
else()
    install(TARGETS qstate
        LIBRARY DESTINATION qsa
        RUNTIME DESTINATION qsa
    )
endif()
