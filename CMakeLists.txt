# ==============================================================================
# DyFence CMake build script
# ==============================================================================
#
# Basic build:
#   mkdir build && cd build
#   cmake ..
#   make -j
#   The default build type is Release.
#
# Debug build:
#   cmake .. -DCMAKE_BUILD_TYPE=Debug
#
# Selective test builds:
#   cmake .. -DTEST_SELECT=<mode>
#
# Supported TEST_SELECT values:
#   - "ALL"        (default): all handwritten tests and generated model targets.
#   - "DEV"        : development tests under test/dev_test/*.cpp.
#   - "FUNC"       : functional tests under test/functional_test/*.cpp.
#   - "LOCALMODEL" : local model tests under test/local_model_test/*.cpp.
#   - "GEN"        : generated model targets under generated_model/*_test.cpp.
#   - "<fragment>" : tests whose filename contains the fragment.
#
# Dependency notes:
#   - jemalloc is used when available and USE_JEMALLOC is enabled.
#   - Source scanning is restricted to C++ source directories to avoid frontend
#     virtual-environment files.
#
# ==============================================================================

cmake_minimum_required(VERSION 3.21)
project(DyFence LANGUAGES CXX)
include(GNUInstallDirs)

# ==========================================
# 1. macOS dependency hints
# ==========================================
if(APPLE)
    find_program(BREW_EXECUTABLE brew)

    if(BREW_EXECUTABLE)
        foreach(formula IN ITEMS libomp openssl@3 jemalloc)
            execute_process(
                COMMAND "${BREW_EXECUTABLE}" --prefix "${formula}"
                OUTPUT_VARIABLE BREW_FORMULA_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE BREW_FORMULA_RESULT
            )

            if(BREW_FORMULA_RESULT EQUAL 0 AND EXISTS "${BREW_FORMULA_PREFIX}")
                list(APPEND CMAKE_PREFIX_PATH "${BREW_FORMULA_PREFIX}")

                if(formula STREQUAL "libomp")
                    set(OpenMP_ROOT "${BREW_FORMULA_PREFIX}")
                elseif(formula STREQUAL "openssl@3" AND NOT DEFINED OPENSSL_ROOT_DIR)
                    set(OPENSSL_ROOT_DIR "${BREW_FORMULA_PREFIX}")
                endif()
            endif()
        endforeach()
    endif()
endif()

# ==========================================
# 2. Compiler options
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Release by default when CMAKE_BUILD_TYPE is not specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Architecture-specific flags.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH_FLAGS -Wno-ignored-attributes -Wno-unused-result)
else()
    # x86-64 flags used by the benchmark configuration.
    set(ARCH_FLAGS -maes -msse4.1 -mpclmul -march=native -funroll-loops -Wno-ignored-attributes -Wno-unused-result)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type: DEBUG (Optimizations OFF)")
    add_compile_options(${ARCH_FLAGS} -O0 -g -ggdb -fno-inline)
else()
    message(STATUS "Build type: RELEASE (Optimizations ON + LTO)")
    add_compile_options(${ARCH_FLAGS} -O3 -g -fno-omit-frame-pointer)
    
    # Enable link-time optimization when supported.
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        message(STATUS "IPO / LTO supported and enabled.")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO / LTO not supported: ${error}")
    endif()
endif()

# GNU LTO can inherit the outer Make jobserver across linker subprocesses.
# Isolate only linker children; retain IPO and normal compilation parallelism.
if(ipo_supported AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(PREPEND CMAKE_CXX_LINKER_LAUNCHER
        "${CMAKE_COMMAND}" -E env --unset=MAKEFLAGS)
endif()

# ==========================================
# 3. Dependencies
# ==========================================
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

# Use the pinned source snapshot by default.
option(DYFENCE_USE_BUNDLED_EMP "Build with the EMP source bundled in third_party" ON)
if(DYFENCE_USE_BUNDLED_EMP)
    set(EMP_TOOL_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/emp-tool")
    set(EMP_OT_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/emp-ot")
    set(EMP_ZK_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/emp-zk")
    foreach(emp_header
            "${EMP_TOOL_INC}/emp-tool/emp-tool.h"
            "${EMP_OT_INC}/emp-ot/emp-ot.h"
            "${EMP_ZK_INC}/emp-zk/emp-zk.h")
        if(NOT EXISTS "${emp_header}")
            message(FATAL_ERROR "Bundled EMP is incomplete: ${emp_header} is missing.")
        endif()
    endforeach()
    file(GLOB_RECURSE EMP_TOOL_SOURCES CONFIGURE_DEPENDS "${EMP_TOOL_INC}/emp-tool/*.cpp")
    add_library(dyfence_emp_tool STATIC ${EMP_TOOL_SOURCES})
    target_include_directories(dyfence_emp_tool PUBLIC ${EMP_TOOL_INC} ${EMP_OT_INC} ${EMP_ZK_INC})
    target_link_libraries(dyfence_emp_tool PUBLIC OpenSSL::Crypto OpenSSL::SSL pthread)
    set(EMP_TOOL_LIB dyfence_emp_tool)
    message(STATUS "DyFence EMP source: bundled")
else()
    find_path(EMP_TOOL_INC "emp-tool/emp-tool.h" REQUIRED)
    find_path(EMP_OT_INC "emp-ot/emp-ot.h" REQUIRED)
    find_path(EMP_ZK_INC "emp-zk/emp-zk.h" REQUIRED)
    find_library(EMP_TOOL_LIB NAMES emp-tool REQUIRED)
    message(STATUS "DyFence EMP source: system")
endif()

# ==========================================
# Optional jemalloc support
# ==========================================
# ANSI escape character used in CMake status messages.
string(ASCII 27 Esc)

option(USE_JEMALLOC "Use jemalloc for memory allocation" ON)

if(USE_JEMALLOC)
    find_library(JEMALLOC_LIB NAMES jemalloc)
    
    if(JEMALLOC_LIB)
        # Link jemalloc globally when it is available.
        message(STATUS "${Esc}[32m[Performance Opt] Found jemalloc at ${JEMALLOC_LIB}. Enabling globally!${Esc}[0m")
        link_libraries(${JEMALLOC_LIB}) 
    else()
        # Fall back to the system allocator.
        message(WARNING "${Esc}[33m[Performance Opt] jemalloc not found. Falling back to default system allocator.${Esc}[0m")
    endif()
else()
    # USE_JEMALLOC was disabled explicitly.
    message(STATUS "${Esc}[34m[Performance Opt] USE_JEMALLOC is OFF. Using default system allocator.${Esc}[0m")
endif()

# ==========================================
# 4. Source discovery
# ==========================================

# Recursively scan source directories that contain project C++ code.
file(GLOB_RECURSE LIBRARY_SOURCES 
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_type/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/exec/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/local_models/*.cpp"
)

# Non-recursive scan for root-level .cpp files.
# Do not use GLOB_RECURSE here, because frontend virtual environments may
# contain unrelated C++ sources.
file(GLOB ROOT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(ROOT_SOURCES)
    list(APPEND LIBRARY_SOURCES ${ROOT_SOURCES})
endif()

# Remove test sources from the core library source list.
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
if(TEST_SOURCES)
    list(REMOVE_ITEM LIBRARY_SOURCES ${TEST_SOURCES})
endif()
# Optional source-list diagnostics.
# message(STATUS "Library Sources: ${LIBRARY_SOURCES}")

# ==========================================
# 5. Core static library
# ==========================================
add_library(dyfence_core STATIC ${LIBRARY_SOURCES})
add_library(DyFence::dyfence_core ALIAS dyfence_core)

# Public include paths.
target_include_directories(dyfence_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>        # Allows #include "data_type/..."
    $<BUILD_INTERFACE:${EMP_TOOL_INC}>
    $<BUILD_INTERFACE:${EMP_OT_INC}>
    $<BUILD_INTERFACE:${EMP_ZK_INC}>
    $<BUILD_INTERFACE:${OpenMP_CXX_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dyfence>
)

# Library dependencies.
target_link_libraries(dyfence_core PUBLIC
    OpenMP::OpenMP_CXX
    ${EMP_TOOL_LIB}
    OpenSSL::Crypto
    OpenSSL::SSL
    pthread
)

# ==========================================
# 6. Test executables
# ==========================================

# ==============================================================================
# Testing Configuration
# ==============================================================================

# TEST_SELECT controls which test targets are configured.
set(TEST_SELECT "ALL" CACHE STRING "Select tests to build: ALL, DEV, FUNC, LOCALMODEL, GEN, or specific filename")

message(STATUS "------------------------------------------------")
message(STATUS "Build Test Mode: ${TEST_SELECT}")

# Keep handwritten tests and generated-model tests in separate lists.
set(TEST_SOURCES "")
set(GEN_TEST_SOURCES "")

if(TEST_SELECT STREQUAL "ALL")
    # Configure both handwritten tests and generated model tests.
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
    file(GLOB_RECURSE GEN_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/generated_model/*_test.cpp")

elseif(TEST_SELECT STREQUAL "DEV")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/dev_test/*.cpp")

elseif(TEST_SELECT STREQUAL "FUNC")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/functional_test/*.cpp")

elseif(TEST_SELECT STREQUAL "LOCALMODEL")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/local_model_test/*.cpp")

elseif(TEST_SELECT STREQUAL "GEN")
    # Configure generated model tests only.
    file(GLOB_RECURSE GEN_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/generated_model/*_test.cpp")

else()
    # Match a filename fragment in the regular test directory.
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*${TEST_SELECT}*.cpp")
endif()

# ==============================================================================
# Build handwritten single-file tests.
# ==============================================================================
if(TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_target_name ${test_src} NAME_WE)
        message(STATUS "  [+] Adding standard test target: ${test_target_name}")

        add_executable(${test_target_name} ${test_src})
        target_link_libraries(${test_target_name} PRIVATE dyfence_core)
    endforeach()
endif()

# ==============================================================================
# Build generated two-file model tests.
# ==============================================================================
if(GEN_TEST_SOURCES)
    foreach(test_src ${GEN_TEST_SOURCES})
        # Base filename, e.g. ResNet18_test.
        get_filename_component(base_name ${test_src} NAME_WE)
        
        # Generated model directory.
        get_filename_component(parent_dir ${test_src} DIRECTORY)
        get_filename_component(folder_name ${parent_dir} NAME)
        
        # Use a globally unique target name to avoid CMake name collisions.
        set(test_target_name "${folder_name}_${base_name}")
        
        # Matching generated implementation file.
        string(REPLACE "_test.cpp" ".cpp" model_src ${test_src})

        message(STATUS "  [+] Adding generated model target: ${test_target_name} -> Output to ${parent_dir}")

        add_executable(${test_target_name} ${test_src} ${model_src})

        target_link_libraries(${test_target_name} PRIVATE dyfence_core)
        target_include_directories(${test_target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

        # Place the binary next to the generated model and keep the short name.
        set_target_properties(${test_target_name} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY "${parent_dir}"
            OUTPUT_NAME "${base_name}"
        )
    endforeach()
endif()

# Warn when no matching test source was found.
if(NOT TEST_SOURCES AND NOT GEN_TEST_SOURCES)
    message(WARNING "  [-] No test source files found matching: ${TEST_SELECT}")
endif()

message(STATUS "------------------------------------------------")

# ==========================================
# 7. Install
# ==========================================

install(TARGETS dyfence_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    config.h
    utility.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dyfence
)

foreach(header_dir IN ITEMS backend data_type exec operations utils)
    install(DIRECTORY ${header_dir}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dyfence/${header_dir}
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
    )
endforeach()
