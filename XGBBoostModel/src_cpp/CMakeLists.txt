cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(xgboost-example LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")


find_package(dmlc CONFIG REQUIRED)
find_package(xgboost CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)


set(PREDICT "predict")

# Adicionando o executável
add_executable(${PREDICT} ${CMAKE_CURRENT_SOURCE_DIR}/XGB_Boost_Model.cpp)


set_target_properties(
    ${PREDICT}
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)


target_link_libraries(
    ${PREDICT}
    PRIVATE
        xgboost::xgboost
        dmlc::dmlc
        cxxopts::cxxopts
        OpenMP::OpenMP_CXX
        Threads::Threads
)


if(MSVC)
    target_compile_options(${PREDICT} PRIVATE /W4)  # Warning level 4 for MSVC
else()
    target_compile_options(${PREDICT} PRIVATE -Wall -Wextra)  # Wall e Wextra para outros compiladores
endif()


message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "DMLC found: ${dmlc_FOUND}")
message(STATUS "XGBoost found: ${xgboost_FOUND}")
message(STATUS "CXXOpts found: ${cxxopts_FOUND}")
