# DeepNav

DeepNav is an RL application focused in the global path planning in MA navigation system. While the local path planning is solved by using the ORCA algorithm, an adapatative global path is searched through Deep RL algorithms.

## Preequisites

RVO library installation: It's important to download, build and install the latest version of RVO.
  - [RVO Download](https://gamma.cs.unc.edu/RVO2/downloads/)
  - [RVO Building](https://gamma.cs.unc.edu/RVO2/documentation/2.0/compiling.html)

[LibTorch installation](https://pytorch.org/cppdocs/installing.html).

## Instructions

At this point only the circle scenario it's settled. Also, only the training step it's coded in main.cpp. 
[CMake build and run] https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html#build-and-run. Anyway, VSCode CMake extension is highly recomendend.

After build:
```
cd Build
./DeepNav
```
 