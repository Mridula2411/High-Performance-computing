import io
import os
import argparse
import numpy as np
from dataloader import DataLoader
from volumerender import render_view

import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

# --- Data Loading ---
def load_datacube(filepath):
    dataloader = DataLoader(virtual_stack=False)
    filepath_str = str(filepath)
    if filepath_str.endswith((".h5", ".hdf5")):
        datacube = dataloader.load_h5(filepath)
    elif filepath_str.endswith((".tif", ".tiff")):
        datacube = dataloader.load_tiff(filepath)
    else:
        raise ValueError("Unsupported file format. Provide HDF5 or TIFF.")
    return datacube

# --- App Creation ---
def create_app(datacube, points, default_N=128):
    # Using a Slate/Dark theme for a high-end renderer look
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    
    projection = np.log(np.mean(datacube, axis=0) + 1e-6)

    # Sidebar UI
    sidebar = dbc.Col([
        html.H2("Renderer", className="display-6 text-primary"),
        html.Hr(),
        html.P("Controls", className="lead"),
        
        html.Label("Rotation Angle"),
        dcc.Slider(
            id="angle-slider", min=0, max=360, step=2, value=0,
            marks={0: '0°', 90: '90°', 180: '180°', 270: '270°', 360: '360°'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode="drag"
        ),
        
        html.Div([
            dbc.Button("Play / Pause", id="play-btn", color="primary", className="me-2", n_clicks=0),
            dbc.Badge("Looping", color="info", id="status-badge", pill=True),
        ], className="d-grid gap-2 mb-4"),

        html.Label("Rendering Quality (N)"),
        dcc.Dropdown(
            id="resolution",
            options=[{"label": f"Low ({n})", "value": n} for n in [64, 128, 180]],
            value=default_N,
            clearable=False,
            className="text-dark"
        ),
        
        html.Hr(),
        html.Label("Export Settings"),
        dbc.Input(id="dpi-input", type="number", value=300, min=72, step=10, className="mb-2"),
        dbc.Button("Download Projection", id="download-btn", color="secondary", outline=True, size="sm"),
        dcc.Download(id="download-projection"),
        
    ], width=3, className="bg-light p-4", style={"height": "100vh", "overflowY": "auto"})

    # Main Display Area
    content = dbc.Col([
        dbc.Row([
            dbc.Col([
                html.H4("3D Volume View"),
                # Removed dcc.Loading to prevent the flicker/wait screen
                dcc.Graph(id="view-graph", style={"height": "70vh"}, config={'displayModeBar': False}),
            ], width=8),
            dbc.Col([
                html.H4("Mean Projection"),
                dcc.Graph(id="projection-graph", style={"height": "40vh"}, config={'displayModeBar': False}),
            ], width=4),
        ], className="mt-4"),
    ], width=9)

    app.layout = dbc.Container([
        dbc.Row([sidebar, content]),
        dcc.Interval(id="play-interval", interval=200, disabled=True),
        dcc.Store(id="playing-state", data=False),
    ], fluid=True)

    # --- Callbacks ---

    @app.callback(
        Output("view-graph", "figure"),
        Input("angle-slider", "value"),
        Input("resolution", "value"),
    )
    def update_view(angle_deg, resolution):
        # The 'previous' figure stays visible until this function returns the new one
        angle = float(angle_deg) * np.pi / 180.0
        img = render_view(datacube, points, angle, N=int(resolution))
        
        # Ensure image is valid
        img_255 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        
        fig = px.imshow(img_255, binary_format="jpg", binary_compression_level=4)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    @app.callback(
        Output("projection-graph", "figure"),
        Input("projection-graph", "id") # Only load once
    )
    def init_projection(_):
        fig = px.imshow(projection, color_continuous_scale="magma")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    @app.callback(
        Output("playing-state", "data"),
        Output("play-btn", "color"),
        Output("play-interval", "disabled"),
        Input("play-btn", "n_clicks"),
        State("playing-state", "data"),
        prevent_initial_call=True
    )
    def toggle_play(n_clicks, is_playing):
        new_state = not is_playing
        color = "danger" if new_state else "primary"
        return new_state, color, not new_state

    @app.callback(
        Output("angle-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("angle-slider", "value"),
        prevent_initial_call=True,
    )
    def advance_frame(n, current_angle):
        return (int(current_angle) + 4) % 360

    @app.callback(
        Output("download-projection", "data"),
        Input("download-btn", "n_clicks"),
        State("dpi-input", "value"),
        prevent_initial_call=True,
    )
    def download_projection(n_clicks, dpi):
        buf = io.BytesIO()
        plt.figure(figsize=(6, 6))
        plt.imshow(projection, cmap="magma")
        plt.axis("off")
        plt.savefig(buf, format="png", dpi=int(dpi), bbox_inches='tight')
        plt.close()
        return dcc.send_bytes(buf.getvalue(), filename="render_projection.png")

    return app

# --- Entry Point ---
def main(filepath: str, host: str = "127.0.0.1", port: int = 8050):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    print(f"Loading data from {filepath}...")
    datacube = load_datacube(filepath)
    Nx, Ny, Nz = datacube.shape
    points = (np.linspace(-Nx/2, Nx/2, Nx), 
              np.linspace(-Ny/2, Ny/2, Ny), 
              np.linspace(-Nz/2, Nz/2, Nz))

    app = create_app(datacube, points)
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    main(args.filepath, port=args.port)