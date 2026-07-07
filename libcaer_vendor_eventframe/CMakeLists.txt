#
# Copyright 2024 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.16)
project(libcaer_vendor_eventframe CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

option(
  BUILD_SHARED_LIBS
  "Global flag to cause add_library() to create shared libraries if on. \
  If set to true, this will cause all libraries to be built shared \
  unless the library was explicitly added as a static library."
  ON)


ament_vendor(libcaer_vendor_eventframe
  # Upstream libcaer, fetched at build time (tag 3.3.15) + local patches/.
  # To hack libcaer locally instead, point VCS_URL at a local clone and commit.
  VCS_URL https://gitlab.com/inivation/dv/libcaer.git
  VCS_VERSION 3.3.15
  GLOBAL_HOOK
  PATCHES patches
  CMAKE_ARGS
    "-DCMAKE_C_FLAGS=-Wno-address-of-packed-member -Wno-sign-conversion -Wno-format-truncation -Wno-unused-parameter -Wno-conversion -Wp,-w"
)

ament_package(
  CONFIG_EXTRAS "libcaer_vendor_eventframe-extras.cmake"
)
