"""Lines module - contains a Rays class, representing lines in 3D space."""

import torch


class Mesh:
    """Mesh in 3D space, Mash triangle mesh is a type of polygon mesh."""

    def __init__(
            self,
            vertices: torch.Tensor,
            faces: torch.Tensor
    ):
        """Initialize Mesh based on triangles and theirs vertices vectors.

        Each triangle in the mesh is defined by vertices in 3D space.
        The mesh is defined with the help of indexes of triangles called faces.
        Args:
            vertices: torch.Tensor with shape (N, 3),
             where N is the number of vertices, and 3 corresponds to
             a set of three coordinates defining a point in 3D space.
            faces: torch.Tensor with shape (N, 3).

        """
        self.vertices = vertices
        self.faces = faces
