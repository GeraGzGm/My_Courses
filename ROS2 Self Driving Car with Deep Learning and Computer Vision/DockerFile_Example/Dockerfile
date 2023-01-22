FROM osrf/ros:noetic-desktop

RUN apt-get update
#-y automatically says "yes" to the question for installing the packages
RUN apt-get install -y git && apt-get install -y python3-pip

#Create workspace for ros
RUN mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
RUN git clone https://github.com/noshluk2/ros1_wiki.git

#Give feedback in terminal
RUN echo "Configuration done :)"