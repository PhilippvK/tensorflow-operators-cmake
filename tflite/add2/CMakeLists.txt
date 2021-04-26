CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)

#SET(CMAKE_TOOLCHAIN_FILE "/Users/Philipp/src/ml_on_mcu/deps/install/etiss/etiss/examples/SW/riscv/cmake/pulpino_tumeda/toolchain.cmake")

PROJECT(tflite_micro_custom)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

IF(NOT ${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
    MESSAGE(FATAL_ERROR "The libraries can only be built on linux operating systems!")
ENDIF()

IF(CMAKE_CROSSCOMPILING)
    MESSAGE(STATUS "Cross-compiling static library for ${CMAKE_SYSTEM_PROCESSOR}")
    MESSAGE(STATUS "For building the shared libray reconfigure CMake without `-DCMAKE_TOOLCHAIN_FILE`")
    SET(ADDITIONAL_SOURCES
        )
    SET(LIBRARY_TYPE "STATIC")
ELSE()
    MESSAGE(STATUS "Builing shared library for ${CMAKE_SYSTEM_NAME}")
    MESSAGE(STATUS "For cross-compiling  the static libray reconfigure CMake using `-DCMAKE_TOOLCHAIN_FILE=...`")
    SET(ADDITIONAL_SOURCES
        ${PROJECT_SOURCE_DIR}/src/registration.cpp
        )
    SET(LIBRARY_TYPE "SHARED")
ENDIF()

IF(NOT TF_DIR)
    MESSAGE(FATAL_ERROR "Please define TF_DIR before configuring CMake!")
ENDIF()

SET(TF_INCS
    ${TF_DIR}
    ${TF_DIR}/tensorflow/lite/micro/tools/make/downloads
    ${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
    ${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
    ${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy
    ${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
    )

SET(PROJECT_HEADERS
    )

SET(PROJECT_SRCS
    ${PROJECT_SOURCE_DIR}/src/custom_implementations.cpp
    ${ADDITIONAL_SOURCES}
    )

ADD_LIBRARY(${PROJECT_NAME} ${LIBRARY_TYPE}
    ${PROJECT_SRCS}
    )
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
    ${TF_INCS}
    )
TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PUBLIC
    TF_LITE_STATIC_MEMORY
    NDEBUG
    TF_LITE_DISABLE_X86_NEON
    TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
    )
TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_11)
