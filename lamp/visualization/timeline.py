# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Temporal state selection for delayed visualization."""

from __future__ import annotations

from lamp.core.types import Person, PersonState


def state_satisfies_delay(state: PersonState, delay: int) -> bool:
    return state.skeleton is not None and state.num_fuses >= delay / 2.0


def lookup_delayed_state(
    person: Person, current_ts_ns: int, delay: int
) -> PersonState | None:
    """Return the person's latest state at or before ``current_ts_ns``."""
    best_ts = -1
    best_state: PersonState | None = None
    for ts, state in person.ts_to_states.items():
        if ts <= current_ts_ns and state_satisfies_delay(state, delay) and ts > best_ts:
            best_ts = ts
            best_state = state
    return best_state
