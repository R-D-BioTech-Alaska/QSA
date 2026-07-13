cmake_minimum_required(VERSION 3.20)
project(QubitNativeStateEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QSTATE_BUILD_TESTS "Build Qubit state-engine tests" ON)
option(QSTATE_BUILD_EXAMPLES "Build Qubit state-engine examples" ON)

add_library(qstate_core STATIC
    src/qstate.cpp
    src/qsc.cpp
)
set_target_properties(qstate_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(qstate_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(qstate_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
)

add_library(qstate SHARED src/c_api.cpp)
target_link_libraries(qstate PRIVATE qstate_core)
target_include_directories(qstate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(qstate PRIVATE QSTATE_BUILD_SHARED)
set_target_properties(qstate PROPERTIES OUTPUT_NAME qstate)

if(QSTATE_BUILD_TESTS)
    enable_testing()
    add_executable(qstate_tests tests/test_qstate.cpp)
    target_link_libraries(qstate_tests PRIVATE qstate_core)
    add_test(NAME qstate_tests COMMAND qstate_tests)
endif()

if(QSTATE_BUILD_EXAMPLES)
    add_executable(qstate_bell examples/bell.cpp)
    target_link_libraries(qstate_bell PRIVATE qstate_core)

    add_executable(qstate_benchmark benchmarks/structured_states.cpp)
    target_link_libraries(qstate_benchmark PRIVATE qstate_core)
endif()


include(GNUInstallDirs)
install(TARGETS qstate
    LIBRARY DESTINATION qsa
    RUNTIME DESTINATION qsa
)
install(TARGETS qstate_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
