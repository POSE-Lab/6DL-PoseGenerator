from Renderer import (
    RenderRGBD,
    init_glfw,
)
import tyro
from utils import io, params
from utils.io import Render_args, merge_params
from rich.pretty import pprint


def run(args):

    data = io.parse_yaml(args.io.config)
    render_params = params.RenderParams(data)
    pprint(data)

    init_glfw(render_params.FBO_WIDTH, render_params.FBO_HEIGHT)
    RenderRGBD(render_params)


if __name__ == "__main__":
    args = tyro.cli(Render_args)
    run(args)
