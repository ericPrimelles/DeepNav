add_library(RL Environment.h Environment.cpp Circle.h DDPGAgent.h DDPGAgent.cpp MADDPG.h MADDPG.cpp Buffer.h Buffer.cpp )
target_link_libraries(RL "${TORCH_LIBRARIES}" RVO OpenGL::GL GLUT::GLUT GLU)
#target_link_libraries(RL "${TORCH_LIBRARIES}" RVO)


