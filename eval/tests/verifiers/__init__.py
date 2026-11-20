# Copyright 2026 Google LLC
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
"""Unit tests for rewardkit verifiers and reward hooks.

Stubs `rewardkit.criterion` as a pass-through decorator: the real one makes
the function uncallable and prevents the tests from staying hermetic.
"""

import sys
import types

_rewardkit = types.ModuleType("rewardkit")
_rewardkit.criterion = lambda **_: (lambda fn: fn)
sys.modules["rewardkit"] = _rewardkit
