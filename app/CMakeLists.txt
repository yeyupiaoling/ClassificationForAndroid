cmake_minimum_required(VERSION 3.4.1)

# common
set(lib_DIR ${CMAKE_SOURCE_DIR}/src/main/jniLibs)
include_directories(${CMAKE_SOURCE_DIR}/include)
find_library(log-lib log android)
find_library(jnigraphics-lib jnigraphics)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -flax-vector-conversions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions -flax-vector-conversions")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -Wl,--gc-sections")

# MNN
add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libMNN.so)
add_library(mnncore SHARED src/main/cpp/mnnnetnative.cpp src/main/cpp/mnnportraitnative.cpp)
add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries(mnncore mnn ${log-lib} ${jnigraphics-lib})
