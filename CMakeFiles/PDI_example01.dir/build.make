# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/joao/Downloads/example01

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/joao/Downloads/example01

# Include any dependencies generated for this target.
include CMakeFiles/PDI_example01.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PDI_example01.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PDI_example01.dir/flags.make

CMakeFiles/PDI_example01.dir/src/main.cpp.o: CMakeFiles/PDI_example01.dir/flags.make
CMakeFiles/PDI_example01.dir/src/main.cpp.o: src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/joao/Downloads/example01/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/PDI_example01.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/PDI_example01.dir/src/main.cpp.o -c /home/joao/Downloads/example01/src/main.cpp

CMakeFiles/PDI_example01.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PDI_example01.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/joao/Downloads/example01/src/main.cpp > CMakeFiles/PDI_example01.dir/src/main.cpp.i

CMakeFiles/PDI_example01.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PDI_example01.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/joao/Downloads/example01/src/main.cpp -o CMakeFiles/PDI_example01.dir/src/main.cpp.s

CMakeFiles/PDI_example01.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/PDI_example01.dir/src/main.cpp.o.requires

CMakeFiles/PDI_example01.dir/src/main.cpp.o.provides: CMakeFiles/PDI_example01.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/PDI_example01.dir/build.make CMakeFiles/PDI_example01.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/PDI_example01.dir/src/main.cpp.o.provides

CMakeFiles/PDI_example01.dir/src/main.cpp.o.provides.build: CMakeFiles/PDI_example01.dir/src/main.cpp.o

# Object files for target PDI_example01
PDI_example01_OBJECTS = \
"CMakeFiles/PDI_example01.dir/src/main.cpp.o"

# External object files for target PDI_example01
PDI_example01_EXTERNAL_OBJECTS =

PDI_example01: CMakeFiles/PDI_example01.dir/src/main.cpp.o
PDI_example01: CMakeFiles/PDI_example01.dir/build.make
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
PDI_example01: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
PDI_example01: CMakeFiles/PDI_example01.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable PDI_example01"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PDI_example01.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PDI_example01.dir/build: PDI_example01
.PHONY : CMakeFiles/PDI_example01.dir/build

CMakeFiles/PDI_example01.dir/requires: CMakeFiles/PDI_example01.dir/src/main.cpp.o.requires
.PHONY : CMakeFiles/PDI_example01.dir/requires

CMakeFiles/PDI_example01.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PDI_example01.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PDI_example01.dir/clean

CMakeFiles/PDI_example01.dir/depend:
	cd /home/joao/Downloads/example01 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/joao/Downloads/example01 /home/joao/Downloads/example01 /home/joao/Downloads/example01 /home/joao/Downloads/example01 /home/joao/Downloads/example01/CMakeFiles/PDI_example01.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PDI_example01.dir/depend

