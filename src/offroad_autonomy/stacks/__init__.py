"""Swappable driving stacks.

Each stack takes an ``Observation`` and returns a ``StackOutput`` (see
``offroad_autonomy.contract``). A run config picks one by import path, so a
new stack is a new package here with nothing in the harness to change.
Stacks may share the parts in ``perception``, ``planning`` and ``control``,
but nothing requires them to.
"""
