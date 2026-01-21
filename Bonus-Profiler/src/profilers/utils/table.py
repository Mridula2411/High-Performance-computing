from tabulate import tabulate

def create_summary_table(timestamps, data, headers=None):
    """
    Produces a professional summary table with recorded values.
    
    Args:
        timestamps: List of time offsets.
        data: List of lists (recorded metrics).
        headers: Column names (e.g., ["Core 0", "Core 1"] or ["VRAM (MiB)"]).
    """
    if not data or not timestamps:
        return "No data recorded."

    # Combine timestamps and data into rows for tabulate
    table_data = []
    for t, row_values in zip(timestamps, data):
        # Round values for a cleaner look, similar to lecture stats [cite: 715, 811]
        rounded_row = [round(val, 2) for val in row_values]
        table_data.append([round(t, 2)] + rounded_row)

    # If no headers provided, generate generic ones based on data shape
    if headers is None:
        headers = [f"Metric {i}" for i in range(len(data[0]))]
    
    # Add 'Time (s)' to the start of headers
    full_headers = ["Time (s)"] + headers

    # Use 'grid' or 'fancy_grid' for a "pretty" look
    return tabulate(table_data, headers=full_headers, tablefmt="grid")