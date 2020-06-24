cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

## Set MSYS generator
# set(CMAKE_GENERATOR "MSYS Makefiles")
# message(STATUS "generator is set to ${CMAKE_GENERATOR}")

## Project information 
project(Numerical_Recipe 
        VERSION 1.0
        DESCRIPTION "In-house numerical recipe with latest c++ code style"
        LANGUAGES CXX C
)

## Set for CXX standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC)
# Find Eigen3
find_package(Eigen3 CONFIG REQUIRED)
# Find fmt
find_package(fmt CONFIG REQUIRED)
endif()
# When some aforementioned libraries cannot found.
# Then download source from web. 
add_subdirectory(extern)

enable_testing()

add_executable(gauss test/test.gauss_quadrature.cc)
target_include_directories(gauss PRIVATE src ${Catch2_SOURCE_DIR}/single_include/catch2)
target_link_libraries(gauss PRIVATE Eigen3::Eigen fmt::fmt-header-only)
target_compile_definitions(gauss PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING=1)
if(MSVC)
    target_compile_options(gauss PRIVATE /W4 /WX)
else()
    target_compile_options(gauss PRIVATE -Wall -Wextra -Werror -pedantic -O3 -fPIC)
endif()

add_test(
    NAME testGaussLegendre
    COMMAND $<TARGET_FILE:gauss>
) 