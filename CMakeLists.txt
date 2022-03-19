cmake_minimum_required(VERSION 3.20)

project(dgquest LANGUAGES CUDA CXX C)

set(CMAKE_VERBOSE_MAKEFILE on)

################## Compiler ABI options
# Set cuda, c++ and c standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CUDA_EXTENSIONS off)
set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_C_EXTENSIONS off)

# Set optimization options
add_compile_options($<$<CONFIG:Release>:$<$<COMPILE_LANG_AND_ID:CXX,Intel>:-xHost>>)
add_compile_options($<$<CONFIG:Release>:$<$<COMPILE_LANG_AND_ID:C,Intel>:-xHost>>)

# Check if lto is supported
include(CheckIPOSupported)
check_ipo_supported()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE on)
################## Compiler ABI options


################## 3rd-parties
find_package(MPI REQUIRED)

include_directories("3rd-party/easyloggingpp/src")
add_compile_definitions(ELPP_THREAD_SAFE)
set(build_static_lib ON "Build easyloggingpp as a static library")
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory("3rd-party/easyloggingpp")
unset(CMAKE_POLICY_DEFAULT_CMP0077)

add_subdirectory("3rd-party/fmtlib")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
################## 3rd-parties


# Set circuit user program
set(USER_PROGRAM
        "testcases/supremecy-32.c"
        CACHE STRING "Path to the user program to be used")
message(STATUS "USER_PROGRAM is \"${USER_PROGRAM}\"")

include_directories(include)

add_library(gpu STATIC src/cpp/QuEST_gpu_kernel.cu)
target_link_libraries(gpu
        PUBLIC easyloggingpp fmt::fmt)
target_compile_options(gpu PRIVATE -Xptxas -v)

aux_source_directory("src" C_SRC_FILES)
aux_source_directory("src/cpp" CXX_SRC_FILES)
add_library(quest ${C_SRC_FILES} ${CXX_SRC_FILES})

target_link_libraries(quest
        PRIVATE gpu
        PUBLIC MPI::MPI_CXX numa
        PUBLIC easyloggingpp fmt::fmt
        )

add_executable(test ${USER_PROGRAM})

target_link_libraries(test
        PRIVATE quest
        )

add_executable(sampler sampler/main.cpp)

target_link_libraries(sampler
        PRIVATE quest)