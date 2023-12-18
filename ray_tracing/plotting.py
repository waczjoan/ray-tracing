import torch
import plotly.graph_objs as go
import plotly.io as pio
from pathlib import Path

pio.renderers.default = "browser"


def plot_rays_mesh_and_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    points: torch.Tensor,
):
    vertices = vertices.detach().cpu().numpy()

    # Create a layout for the 3D scene
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )
    data = []

    pts_on_mesh_sc = [go.Scatter3d(
        x=points[:, 0].detach().cpu().numpy(),
        y=points[:, 1].detach().cpu().numpy(),
        z=points[:, 2].detach().cpu().numpy(),
        mode='markers',
        marker=dict(size=5, color='red'),
        name='pts_on_mesh',
    )]

    lines = []
    for i in range(len(rays_origins)):
        ray_origin, ray_direction = rays_origins[i], rays_directions[i]
        ray_end = ray_origin + ray_direction
        lines.append(go.Scatter3d(
            x=[ray_origin[0], ray_end[0]],
            y=[ray_origin[1], ray_end[1]],
            z=[ray_origin[2], ray_end[2]],
            mode='lines',
            line=dict(width=3, color='blue'),
            name=f'Ray_{i}'
        ))

    data = data + lines

    triangles = []
    for face in faces:
        f = list(face)
        triangles.append(
            go.Scatter3d(
                x=vertices[:, 0][f + [f[0]]],
                y=vertices[:, 1][f + [f[0]]],
                z=vertices[:, 2][f + [f[0]]],
                mode='lines',
                marker=dict(size=5, color='black'),
                showlegend=False
            )
        )

    data = data + triangles + pts_on_mesh_sc

    fig = go.Figure(
        data=data,
        layout=layout
    )

    dir = 'output_tests'
    Path(dir).mkdir(parents=True, exist_ok=True)

    fig.write_html(
        f'{dir}/intersection_points_with_mesh.html', auto_open=True
    )
