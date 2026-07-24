cmake_minimum_required(VERSION 3.20)
project(QSAConsumer LANGUAGES C CXX)

find_package(QSA CONFIG REQUIRED)

add_executable(qsa_consumer_cpp main.cpp)
target_link_libraries(qsa_consumer_cpp PRIVATE QSA::qstate_core)
target_compile_features(qsa_consumer_cpp PRIVATE cxx_std_20)

add_executable(qsa_consumer_c main.c)
target_link_libraries(qsa_consumer_c PRIVATE QSA::qstate)
