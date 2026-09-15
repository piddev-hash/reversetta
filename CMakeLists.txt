cmake_minimum_required(VERSION 3.24)
project(reversetta VERSION 0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build AsmJit as a proper static library
set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)
set(ASMJIT_NO_AARCH64 ON CACHE BOOL "" FORCE) # disable the ARM backend

# Add the nested directory (CMake finds asmjit/asmjit/CMakeLists.txt)
add_subdirectory(asmjit)

add_executable(reversetta
    src/main.cpp
    src/emu.cpp
    src/translator.cpp
    src/host-fns.cpp
)

target_include_directories(reversetta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the library. CMake propagates the global <asmjit/asmjit.h> path itself.
target_link_libraries(reversetta PRIVATE asmjit::asmjit)
