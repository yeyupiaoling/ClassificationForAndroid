# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/)

set(lib_DIR ${CMAKE_SOURCE_DIR}/src/main/jniLibs)
set(TNN_OPENCL_ENABLE ON CACHE BOOL "" FORCE)
set(TNN_ARM_ENABLE ON CACHE BOOL "" FORCE)

file(GLOB_RECURSE WRAPPER_SRCS ${CMAKE_SOURCE_DIR}/src/main/cpp/*.cc)
add_library(tnn_wrapper SHARED ${WRAPPER_SRCS})

add_library(tnn SHARED IMPORTED)
set_target_properties(tnn PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libTNN.so)

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        tnn_wrapper
        -ljnigraphics
        tnn
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})
