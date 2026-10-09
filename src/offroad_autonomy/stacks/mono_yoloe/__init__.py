"""One GMSL2 dashcam, YOLOE-26 road segmentation, a ground planner and Stanley."""


def build(stack_config: dict):
    """The harness's entry point, named in the run config as
    ``offroad_autonomy.stacks.mono_yoloe:build``.

    The stack is imported here, not at the top of the package, because it
    pulls in Torch and Ultralytics, and tools that only read this stack's
    config (the S-bend simulation, the ego-mask script) must not pay for that.
    """
    from offroad_autonomy.stacks.mono_yoloe.stack import build as build_stack

    return build_stack(stack_config)
