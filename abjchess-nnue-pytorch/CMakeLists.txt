cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
endif()

project(training_data_loader)

# The JQv4 common reader is a static dependency of the training data loader.
# Point JQV4_COMMON_ROOT at a checkout of that reader when building the loader.
set(JQV4_BUILD_TESTS OFF CACHE BOOL "Build jqv4 common tests" FORCE)
set(JQV4_BUILD_RECOVERY OFF CACHE BOOL "Build jqv4 recovery implementation" FORCE)
set(JQV4_BUILD_WRITER OFF CACHE BOOL "Build jqv4 writer implementation" FORCE)

# zstd is consumed as a static target and may be either an install tree
# (lib/cmake/zstd)
# or a direct CMake build tree (zstdConfig.cmake at its root).  Accept both
# spellings so callers can pass either -DZSTD_DIR/-Dzstd_DIR or -DZSTD_ROOT.
set(V8_ZSTD_DIR "" CACHE PATH
    "Directory containing zstdConfig.cmake (overrides package search)")
if(DEFINED ZSTD_DIR AND NOT "${ZSTD_DIR}" STREQUAL "")
    set(V8_ZSTD_DIR "${ZSTD_DIR}" CACHE PATH
        "Directory containing zstdConfig.cmake (overrides package search)"
        FORCE)
elseif(DEFINED zstd_DIR AND NOT "${zstd_DIR}" STREQUAL "")
    set(V8_ZSTD_DIR "${zstd_DIR}" CACHE PATH
        "Directory containing zstdConfig.cmake (overrides package search)"
        FORCE)
endif()

if(DEFINED ZSTD_ROOT AND NOT "${ZSTD_ROOT}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "${ZSTD_ROOT}")
    if(EXISTS "${ZSTD_ROOT}/zstdConfig.cmake")
        set(V8_ZSTD_DIR "${ZSTD_ROOT}" CACHE PATH
            "Directory containing zstdConfig.cmake (overrides package search)"
            FORCE)
    elseif(EXISTS "${ZSTD_ROOT}/lib/cmake/zstd/zstdConfig.cmake")
        set(V8_ZSTD_DIR "${ZSTD_ROOT}/lib/cmake/zstd" CACHE PATH
            "Directory containing zstdConfig.cmake (overrides package search)"
            FORCE)
    endif()
endif()

if(NOT "${V8_ZSTD_DIR}" STREQUAL "")
    set(zstd_DIR "${V8_ZSTD_DIR}" CACHE PATH
        "CMake config directory for zstd" FORCE)
endif()
find_package(zstd 1.5.7 EXACT CONFIG REQUIRED)
if(NOT TARGET zstd::libzstd_static)
    message(FATAL_ERROR
            "V8 JQv4 loader requires the static zstd::libzstd_static target")
endif()
set(JQV4_COMMON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../jieqi_v4_common" CACHE PATH
    "Path to the JQv4 common reader source tree")
if(NOT EXISTS "${JQV4_COMMON_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "JQV4_COMMON_ROOT must point to a JQv4 common source tree")
endif()
add_subdirectory(${JQV4_COMMON_ROOT}
                 ${CMAKE_CURRENT_BINARY_DIR}/jieqi_v4_common
                 EXCLUDE_FROM_ALL)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_library(training_data_loader SHARED training_data_loader.cpp)

# MinGW's Windows headers hide FILE_ID_INFO/FileIdInfo unless the target
# Windows version is declared before inclusion.  JQv4 recovery uses that API
# to pin source directories while opening a catalog.
if(WIN32)
    target_compile_definitions(jqv4_common PRIVATE
                               _WIN32_WINNT=0x0602
                               WINVER=0x0602)
    target_compile_definitions(training_data_loader PRIVATE
                               _WIN32_WINNT=0x0602
                               WINVER=0x0602)
endif()

target_include_directories(training_data_loader PRIVATE
    ${JQV4_COMMON_ROOT}/include)

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader PRIVATE
                      Threads::Threads
                      $<TARGET_FILE:jqv4_common>
                      zstd::libzstd_static)
add_dependencies(training_data_loader jqv4_common)

# The loader is loaded into a long-lived Python process.  A dynamically linked
# MinGW C++/pthread runtime is unsafe here: another package (or another DLL in
# System32) can satisfy the same basename with incompatible symbols.  Keep the
# Windows distribution self-contained while leaving native Unix builds alone.
option(V8_STATIC_MINGW_RUNTIME
       "Statically link the MinGW runtime into the Windows loader" ON)
if (WIN32 AND V8_STATIC_MINGW_RUNTIME
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  find_file(V8_WINPTHREAD_STATIC
            NAMES libwinpthread.a
            PATHS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
            NO_DEFAULT_PATH)
  if (NOT V8_WINPTHREAD_STATIC)
    message(FATAL_ERROR
            "V8_STATIC_MINGW_RUNTIME requires a static libwinpthread.a")
  endif()
  set_property(TARGET training_data_loader APPEND_STRING PROPERTY
               LINK_FLAGS
               " -static-libgcc -static-libstdc++")
  # target_link_libraries entries are emitted after the object file.  Keep the
  # archive here and force its objects into the image so the compiler driver
  # cannot fall back to its DLL import library for an unresolved pthread symbol.
  target_link_libraries(training_data_loader PRIVATE
                        "-Wl,--whole-archive"
                        "${V8_WINPTHREAD_STATIC}"
                        "-Wl,--no-whole-archive")
endif()

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
