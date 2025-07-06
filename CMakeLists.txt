# Minimalist CMake file for C++20 with Ninja

cmake_minimum_required(VERSION 3.16)
project(lin_reg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/lin_reg.cpp
)

add_executable(lin_reg ${SOURCES})