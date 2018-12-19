cmake_minimum_required(VERSION 2.8.3)
project(galileo_map_updater)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  dynamic_reconfigure
)

add_message_files(
  FILES
)

generate_dynamic_reconfigure_options(
  cfg/MapUpdater.cfg
)


generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)