add_library(RL Environment.h Environment.cpp Circle.h DDPGAgent.h DDPGAgent.cpp MADDPG.h MADDPG.cpp)
target_link_libraries(RL "${TORCH_LIBRARIES}" RVO OpenGL::GL GLUT::GLUT)

