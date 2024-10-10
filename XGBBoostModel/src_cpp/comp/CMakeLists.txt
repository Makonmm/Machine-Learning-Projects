cmake_minimum_required(VERSION 3.5 FATAL_ERROR)  # Atualizado para 3.5
project(xgboost-example-superbuild LANGUAGES CXX C VERSION 0.1.0)

include(ExternalProject)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Git REQUIRED)

# Definindo os prefixos para as dependências
set(XGBOOST_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/xgboost)
set(XGBOOST_INSTALL_PREFIX ${XGBOOST_PREFIX}/install)


ExternalProject_Add(
    xgboost
    PREFIX ${XGBOOST_PREFIX}
    GIT_REPOSITORY "https://github.com/dmlc/xgboost.git"
    GIT_SHALLOW ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DBUILD_STATIC_LIB=ON
        -DCMAKE_INSTALL_PREFIX=${XGBOOST_INSTALL_PREFIX}
    LOG_INSTALL ON
)

set(CXXOPTS_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/cxxopts)
set(CXXOPTS_INSTALL_PREFIX ${CXXOPTS_PREFIX}/install)

ExternalProject_Add(
    cxxopts
    PREFIX ${CXXOPTS_PREFIX}
    URL "https://github.com/jarro2783/cxxopts/archive/v2.1.2.tar.gz"
    URL_HASH MD5=51af98ad0b7ec6fb7aa1cf96ea65fff8
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CXXOPTS_INSTALL_PREFIX}
    LOG_DOWNLOAD ON
)


set(TARGET_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/predict")  # Ajustado
ExternalProject_Add(
    predict
    PREFIX ${TARGET_PREFIX}
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.."
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/predict/build"  # Ajustado
    INSTALL_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
        -Dxgboost_DIR:PATH=${XGBOOST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/xgboost/
        -Ddmlc_DIR:PATH=${XGBOOST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/dmlc/
        -Dcxxopts_DIR:PATH=${CXXOPTS_INSTALL_PREFIX}/lib/cmake/cxxopts/
    DEPENDS
        xgboost
        cxxopts
)


message(STATUS "XGBoost will be installed in: ${XGBOOST_INSTALL_PREFIX}")
message(STATUS "CXXOpts will be installed in: ${CXXOPTS_INSTALL_PREFIX}")
