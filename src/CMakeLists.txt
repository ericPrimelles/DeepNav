add_library(RL Environment.h Environment.cpp Circle.h)
target_link_libraries(RL "${TORCH_LIBRARIES}" RVO OpenGL::GL GLUT::GLUT)

