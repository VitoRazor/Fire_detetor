cmake_minimum_required(VERSION 3.6.1)
project(VIPL-Detector)
#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../common)
# #######################################
# 【*】添加编译选项
#set (CMAKE_PREFIX_PATH"E:\\Software\\QT\\5.9.1\\msvc2015_64")
# Find includes in corresponding build directories 
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically whenneeded 
set(CMAKE_AUTOMOC ON)
# #######################################
# Find the QtWidgets library 
find_package(Qt5Widgets REQUIRED)
# #######################################
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
         set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
         set(CMAKE_CXX_FLAGS"${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
         set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
# #######################################
FIND_PACKAGE(OpenCV REQUIRED)
# #######################################
# 【*】添加头文件
include_directories(${Qt5Widgets_INCLUDES})
include_directories(
         ././QT_UI/
)
# #######################################
# 【*】链接UI文件
QT5_WRAP_UI(ui_FILES ./QT_UI/vipl_detector.ui
)
# #######################################
SET (FILE
     main.cpp
     VIPL_class.hpp
)
SET (QT_UI
         QT_UI/vipl_detector.h
         QT_UI/vipl_detector.cpp
)
# #######################################
ADD_EXECUTABLE(VIPL-Detector ${FILE} ${QT_UI} ${ui_FILES})
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/exe)  
set_target_properties(VIPL-Detector PROPERTIES OUTPUT_NAME "VIPL-Detector" DEBUG_POSTFIX "d") 
qt5_use_modules(VIPL-Detector Widgets)
# #######################################
# 【*】附加依赖项
#target_link_libraries(${PROJECT_NAME} VIPL-Detector.dll VIPL-Detector.lib)
TARGET_LINK_LIBRARIES(VIPL-Detector
#       Qt5::Widgets
         ${Qt5Widgets_LIBRARIES}
         ${OpenCV_LIBS}
)