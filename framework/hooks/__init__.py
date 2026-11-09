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
"""Hook dispatch: what `runtime.py hook-dispatch` runs for each harness event.

`events` turns a harness payload into a canonical event, `stop_filter` decides
whether a SubagentStop is a finished subagent, `dispatch` filters the event
and shapes the reply, and `actions` runs the hook itself.
"""

from framework.hooks import actions, dispatch, events, stop_filter

__all__ = ["actions", "dispatch", "events", "stop_filter"]
