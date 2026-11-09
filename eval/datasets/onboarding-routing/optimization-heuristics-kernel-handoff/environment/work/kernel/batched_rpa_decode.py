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
"""Decode attention over a paged KV cache: the served batched kernel's computation, in plain JAX."""

import jax
import jax.numpy as jnp


def get_inputs():
    k = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k[0], (64, 8, 128), jnp.bfloat16)
    kv = jax.random.normal(k[1], (2048, 128, 8, 128), jnp.bfloat16)
    pages = jax.random.randint(k[2], (64, 64), 0, 2048)
    lens = jnp.full((64, ), 1280, jnp.int32)
    return [q, kv, pages, lens], []


def computation(q, kv, pages, lens):
    keys = kv[pages].reshape(q.shape[0], -1, kv.shape[2], kv.shape[3])
    pos = jnp.arange(keys.shape[1])[None, :]
    s = jnp.einsum("bhd,bthd->bht", q, keys) / jnp.sqrt(q.shape[-1])
    s = jnp.where((pos < lens[:, None])[:, None, :], s, -1e30)
    return jnp.einsum("bht,bthd->bhd", jax.nn.softmax(s, axis=-1), keys)
