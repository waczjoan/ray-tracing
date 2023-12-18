"""Lines module - contains a Rays class, representing lines in 3D space."""

import torch

from ray_tracing.mesh import Mesh
from ray_tracing.utils import (
    rays_triangles_intersection
)
from ray_tracing.plotting import (
    plot_rays_mesh_and_points
)


class Rays:
    """Lines in 3D space, each has an origin point and a direction vector."""

    def __init__(
            self,
            origin: torch.Tensor,
            direction: torch.Tensor
    ):
        """Initialize lines with origin points and direction vectors.

        Each line has a corresponding origin point and a direction vector.
        Args:
            origin: torch.Tensor with shape (N, 3),
             where N is the number of lines, and 3 corresponds to
             a set of three coordinates defining a point in 3D space.
            direction: torch.Tensor with shape (N, 3).

        """
        self.origin = origin
        self.direction = direction

    def find_intersection_points_with_mesh(
        self,
        mesh: Mesh,
        plot: bool = False,
    ):
        """
        Find points on the mesh, determined by intersecting rays with the mesh.

        Args:
            mesh: Mesh,
             triangle mesh defined by faces and vertices.
            plot: bool
             if True plot rays, mesh and intersection points
        Return:
            # TODO

        """
        triangle_vertices = mesh.vertices[mesh.faces]

        out = rays_triangles_intersection(
            ray_origins=self.origin,
            ray_directions=self.direction,
            triangle_vertices=triangle_vertices,
        )

        if plot:
            self.plot_intersection_points_with_mesh(
                mesh=mesh,
                points=out['pts'][out['valid_point']]
            )

        return out

    def plot_intersection_points_with_mesh(
        self,
        mesh: Mesh,
        points: torch.Tensor,
    ):
        """
        Plot intersection_points_with_mesh
        """

        plot_rays_mesh_and_points(
            rays_origins=self.origin,
            rays_directions=self.direction,
            vertices=mesh.vertices,
            faces=mesh.faces,
            points=points,
        )
