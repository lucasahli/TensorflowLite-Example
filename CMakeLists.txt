cmake_minimum_required(VERSION 3.19)
project(tensorflowTest C CXX)

set(TENSORFLOW_SOURCE_DIR "/Users/luca/Develop/tensorflow_src" CACHE PATH
        "Directory that contains the TensorFlow project"
        )
if(NOT TENSORFLOW_SOURCE_DIR)
    get_filename_component(TENSORFLOW_SOURCE_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../../../../"
            ABSOLUTE
            )
endif()

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
        EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)

# Add Conan libs
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(tensorflowTest main.cpp)

target_link_libraries(tensorflowTest tensorflow-lite ${CONAN_LIBS})