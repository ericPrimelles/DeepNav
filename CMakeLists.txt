cmake_minimum_required(VERSION 3.0.0)
project(TorchNav VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(TorchNav main.cpp)


list(APPEND CMAKE_PREFIX_PATH "/home/eric-pc/libtorch")
find_package( Torch REQUIRED )
find_package(RVO REQUIRED)
#find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

#target_link_libraries(TorchNav "${TORCH_LIBRARIES}" RL OpenGL::GL GLUT::GLUT)
target_link_libraries(TorchNav "${TORCH_LIBRARIES}" RL)


target_include_directories(TorchNav PUBLIC "${PROJECT_BINARY_DIR}"
"${PROJECT_SOURCE_DIR}/src")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
add_subdirectory(src)