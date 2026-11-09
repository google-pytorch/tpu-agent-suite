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

FROM us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host@sha256:7ff58a98ae2655dbcf25fa0aeb3e8081281bfb3061f97581a98951e98371fff4
# The agent package's host requirements the base image lacks; the optimization heuristics read
# parquet with pandas and pyarrow.
RUN pip install --no-cache-dir 'matplotlib>=3.7' 'pandas>=2.0' 'pyarrow>=14.0'

WORKDIR /workspace
COPY work /workspace
