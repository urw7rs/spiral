# Copyright 2019 DeepMind Technologies Limited.
# 2 May 2020 - Modified by urw7rs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register

register(
    id="Libmypaint-v0", entry_point="spiral_gym.envs:LibMyPaint",
)

register(
    id="Fluid-v0", entry_point="spiral_gym.envs:FluidPaint",
)

register(
    id="Libmypaint-v1", entry_point="spiral_gym.envs:LibMyPaintCompound",
)

register(
    id="Fluid-v1", entry_point="spiral_gym.envs:FluidPaintCompound",
)
