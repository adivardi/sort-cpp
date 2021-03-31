cmake_minimum_required(VERSION 3.5.1)
project(sort_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(PROJECT_CATKIN_DEPS
    roscpp
    grid_map_core
    grid_map_ros
    pcl_conversions
    pcl_ros
    tf
    visualization_msgs
    enway_msgs
)

find_package(catkin REQUIRED
  COMPONENTS
    ${PROJECT_CATKIN_DEPS}
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()

# # Try to find Boost
# find_package(Boost COMPONENTS program_options filesystem REQUIRED)
# if(Boost_FOUND)
#     include_directories(${Boost_INCLUDE_DIRS})
# else ()
#     message(FATAL_ERROR "Could not locate Boost")
# endif()

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${PROJECT_CATKIN_DEPS}
  DEPENDS
    OpenCV
)

add_definitions(${catkin_DEFINITIONS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# file(GLOB SOURCE_FILES src/*.cpp)

add_subdirectory(src/${PROJECT_NAME})

# add_executable(${CMAKE_PROJECT_NAME} src/main.cpp )
# add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

# target_link_libraries (
#         ${CMAKE_PROJECT_NAME}
#         ${OpenCV_LIBS}
#         ${Boost_LIBRARIES}
#         ${catkin_LIBRARIES}
# )

########################################################
# Installs
########################################################

## Mark executables and/or libraries for installation
# install(TARGETS ${CMAKE_PROJECT_NAME}
# 	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# 	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# 	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(
  DIRECTORY
    include/${PROJECT_NAME}/
  DESTINATION
    ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
