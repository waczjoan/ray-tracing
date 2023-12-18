"""Tests for rays module functions."""

import torch

from ray_tracing.rays import Rays
from ray_tracing.mesh import Mesh

path_dir = 'tests/test_rays_data'

rays = Rays(
    origin=torch.load(f'{path_dir}/rays_origins.pt'),
    direction=torch.load(f'{path_dir}/rays_directions.pt'),
)


def find_intersection_points_with_mesh(
    plot: bool = True,
):
    """Test finding intersection points with mesh."""

    # Two mesh available to test
    vertices = torch.load(f'{path_dir}/vertices.pt')
    faces = torch.load(f'{path_dir}/faces.pt')

    mesh_two_icospheres = Mesh(
        vertices=vertices,
        faces=faces
    )

    out = rays.find_intersection_points_with_mesh(
        mesh=mesh_two_icospheres,
        plot=plot
    )


if __name__ == "__main__":
    find_intersection_points_with_mesh()
