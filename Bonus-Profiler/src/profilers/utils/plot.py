import plotly.graph_objects as go


def create_evolution_plot(timestamps, data, title="Resource Usage Over Time", y_label="Usage", labels=None):
    """
    A generalized Plotly evolution plot.
    
    Args:
        timestamps: List of time offsets.
        data: List of lists (e.g., [[core0, core1], [core0, core1]]).
        title: The plot title.
        y_label: The unit/label for the Y-axis. 
        labels: Custom names for each data stream (e.g., ["Core 0", "Core 1"] or ["VRAM"]).
    """
    fig = go.Figure()
    
    if not data or not timestamps:
        print("No data recorded to plot.")
        return

    # Determine how many individual lines (streams) to plot
    num_streams = len(data[0])
    
    # If no labels are provided, generate generic ones
    if labels is None:
        labels = [f"Stream {i}" for i in range(num_streams)]

    for i in range(num_streams):
        stream_data = [sample[i] for sample in data]
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=stream_data, 
            mode='lines',
            name=labels[i]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified" 
    )
    
    fig.show()