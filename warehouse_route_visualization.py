import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json

def generate_layout(num_aisles=19, num_rows=12, aisle_labels=None, aisle_spacing=1, row_spacing=1, shelf_height=0.6):
    """
    Generate warehouse layout DataFrame.
    Returns: DataFrame with columns: shelf_id, aisle_label, row, x, y
    """
    if aisle_labels is None:
        aisle_labels = [f"A{str(i).zfill(2)}" for i in range(num_aisles, 0, -1)]
    data = []
    # Depot at (0,0)
    data.append(["Depot", "Depot", 0, 0, 0])
    for a_idx, aisle in enumerate(aisle_labels):
        x = a_idx * aisle_spacing + 1  # +1 so depot is at 0
        for r in range(1, num_rows+1):
            y = num_rows - r + 1  # row 1 at top
            shelf_id = f"{aisle}_{r}"
            data.append([shelf_id, aisle, r, x, y])
    df = pd.DataFrame(data, columns=["shelf_id", "aisle_label", "row", "x", "y"])
    return df

def load_orders(filename):
    """
    Reads CSV of order lines (order_id,item_id,shelf_id) and returns dict: order_id -> list of shelf_ids
    """
    df = pd.read_csv(filename)
    orders = {}
    for oid, group in df.groupby('order_id'):
        orders[oid] = list(group['shelf_id'])
    return orders

def shelf_to_coord(shelf_id, layout_df):
    """
    Returns (x, y) for shelf_id from layout_df
    """
    row = layout_df[layout_df['shelf_id'] == shelf_id]
    if row.empty:
        raise ValueError(f"Shelf {shelf_id} not found in layout.")
    return float(row['x'].values[0]), float(row['y'].values[0])

def manhattan(a, b):
    """
    Manhattan distance between two (x, y) points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_neighbor_route(start_coord, list_of_coords):
    """
    Returns route as list of coords, starting and ending at start_coord
    """
    unvisited = list(list_of_coords)
    route = [start_coord]
    current = start_coord
    while unvisited:
        nxt = min(unvisited, key=lambda p: manhattan(current, p))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(start_coord)
    return route

def two_opt_improve(route, coords_map):
    """
    2-opt improvement for Manhattan route. Keeps start/end fixed.
    """
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route, coords_map) < route_distance(best, coords_map):
                    best = new_route
                    improved = True
        route = best
    return best

def route_distance(route, coords_map):
    """
    Computes total Manhattan distance of an ordered route (list of shelf_ids)
    """
    total = 0
    for i in range(len(route) - 1):
        total += manhattan(coords_map[route[i]], coords_map[route[i+1]])
    return total

def visualize_layout(layout_df, route_shelf_ids, coords_map, highlight_points=None, show_crosshair_point=None, filename=None):
    """
    Draws warehouse layout, shelves, route, highlights, crosshair, and saves/shows image.
    """
    num_aisles = layout_df['aisle_label'].nunique() if 'aisle_label' in layout_df else 19
    num_rows = layout_df['row'].max() if 'row' in layout_df else 12
    aisle_labels = list(layout_df['aisle_label'].unique())
    fig, ax = plt.subplots(figsize=(14, 8))
    # Draw shelves
    for _, row in layout_df.iterrows():
        if row['shelf_id'] == 'Depot':
            ax.add_patch(plt.Rectangle((row['x']-0.3, row['y']-0.3), 0.6, 0.6, color='red', zorder=3))
            ax.text(row['x']-0.5, row['y']-0.7, 'START', fontsize=12, color='red', fontweight='bold')
        else:
            ax.add_patch(plt.Rectangle((row['x']-0.3, row['y']-0.3), 0.6, 0.6, edgecolor='black', facecolor='white', lw=1, zorder=2))
    # Draw aisle labels (top)
    for idx, aisle in enumerate(aisle_labels):
        x = layout_df[layout_df['aisle_label'] == aisle]['x'].iloc[0]
        ax.text(x, num_rows+1, aisle, ha='center', va='bottom', fontsize=11, rotation=30, fontweight='bold', color='#1565c0')
    # Draw row labels (left)
    for r in range(1, num_rows+1):
        ax.text(-1, num_rows-r+1, str(r), ha='right', va='center', fontsize=11, fontweight='bold', color='#2e7d32')
    # Draw route (L-shaped Manhattan)
    if route_shelf_ids:
        route_coords = [coords_map[sid] for sid in route_shelf_ids]
        for i in range(len(route_coords)-1):
            x1, y1 = route_coords[i]
            x2, y2 = route_coords[i+1]
            ax.plot([x1, x2], [y1, y1], color='green', lw=2, zorder=5)
            ax.plot([x2, x2], [y1, y2], color='green', lw=2, zorder=5)
            ax.scatter([x2], [y2], color='blue', s=80, zorder=6)
            ax.text(x2, y2+0.3, f"{i+1}", ha='center', va='bottom', fontsize=9, color='blue')
    # Highlight picked shelves
    if highlight_points:
        for sid in highlight_points:
            x, y = coords_map[sid]
            ax.add_patch(plt.Rectangle((x-0.3, y-0.3), 0.6, 0.6, edgecolor='blue', facecolor='#e3f2fd', lw=2, zorder=4))
    # Show crosshair point
    if show_crosshair_point:
        x, y = coords_map[show_crosshair_point]
        ax.scatter([x], [y], color='red', s=120, zorder=10)
        ax.axhline(y, color='red', linestyle='--', lw=1, zorder=9)
        ax.axvline(x, color='red', linestyle='--', lw=1, zorder=9)
        ax.text(x+0.5, y+0.5, f"{show_crosshair_point} ({x:.1f},{y:.1f})", fontsize=10, color='red', fontweight='bold')
    ax.set_xlim(-2, num_aisles+2)
    ax.set_ylim(-2, num_rows+3)
    ax.axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Warehouse Route Optimization')
    parser.add_argument('--order', type=str, help='Order ID to plot')
    parser.add_argument('--csv', type=str, help='Orders CSV file')
    args = parser.parse_args()
    # Generate layout
    layout_file = 'warehouse_layout.csv'
    if os.path.exists(layout_file):
        layout_df = pd.read_csv(layout_file)
    else:
        layout_df = generate_layout()
        layout_df.to_csv(layout_file, index=False)
    # Load orders
    orders_file = args.csv if args.csv else 'orders.csv'
    if os.path.exists(orders_file):
        orders = load_orders(orders_file)
        order_id = args.order if args.order else list(orders.keys())[0]
        shelf_ids = orders[order_id]
    else:
        # Sample order
        shelf_ids = ['A19_2', 'A15_7', 'A10_4', 'A05_11', 'A01_6']
        order_id = 'SAMPLE'
        # Save sample orders
        df_orders = pd.DataFrame({
            'order_id': [order_id]*len(shelf_ids),
            'item_id': [f'SKU{i+1:03}' for i in range(len(shelf_ids))],
            'shelf_id': shelf_ids
        })
        df_orders.to_csv('orders.csv', index=False)
    # Map shelf_ids to coords
    coords_map = {row['shelf_id']: (row['x'], row['y']) for _, row in layout_df.iterrows()}
    coords_map['Depot'] = (0, 0)
    pick_coords = [coords_map[sid] for sid in shelf_ids]
    # Route optimization
    route_nn_coords = nearest_neighbor_route(coords_map['Depot'], pick_coords)
    # Map coords back to shelf_ids
    def coord_to_shelf(coord):
        for sid, xy in coords_map.items():
            if np.allclose(xy, coord):
                return sid
        return None
    route_nn_shelf = [coord_to_shelf(c) for c in route_nn_coords]
    route_opt_shelf = two_opt_improve(route_nn_shelf, coords_map)
    # Distances
    baseline_dist = route_distance(route_nn_shelf, coords_map)
    opt_dist = route_distance(route_opt_shelf, coords_map)
    print(f"Order: {order_id}")
    print(f"Baseline NN route: {' -> '.join(route_nn_shelf)} | Distance: {baseline_dist}")
    print(f"Optimized 2-opt route: {' -> '.join(route_opt_shelf)} | Distance: {opt_dist}")
    print(f"Improvement: {100*(baseline_dist-opt_dist)/baseline_dist:.1f}%")
    # Visualize
    visualize_layout(layout_df, route_opt_shelf, coords_map, highlight_points=shelf_ids, show_crosshair_point=shelf_ids[0], filename='warehouse_route.png')
    # Save metrics
    metrics = {
        'order_id': order_id,
        'baseline_distance': baseline_dist,
        'optimized_distance': opt_dist,
        'improvement_percent': round(100*(baseline_dist-opt_dist)/baseline_dist, 1),
        'route_sequence': route_opt_shelf
    }
    with open('route_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def find_shortest_path(shelf_ids, num_aisles=19, num_rows=12):
    """
    Public wrapper function for route optimization.
    Args:
        shelf_ids: list of shelves to pick (e.g. ["A19_2", "A15_7", ...])
        num_aisles, num_rows: layout config (default: 19 aisles x 12 rows)
    Returns:
        (optimized_route, total_distance)
    """
    # Generate or load layout
    layout_file = 'warehouse_layout.csv'
    if os.path.exists(layout_file):
        layout_df = pd.read_csv(layout_file)
    else:
        layout_df = generate_layout(num_aisles=num_aisles, num_rows=num_rows)
        layout_df.to_csv(layout_file, index=False)

    # Build coords map
    coords_map = {row['shelf_id']: (row['x'], row['y']) for _, row in layout_df.iterrows()}
    coords_map['Depot'] = (0, 0)

    # Get coordinates for shelves
    pick_coords = [coords_map[sid] for sid in shelf_ids]

    # Run NN + 2-opt
    route_nn_coords = nearest_neighbor_route(coords_map['Depot'], pick_coords)

    def coord_to_shelf(coord):
        for sid, xy in coords_map.items():
            if np.allclose(xy, coord):
                return sid
        return None

    route_nn_shelf = [coord_to_shelf(c) for c in route_nn_coords]
    route_opt_shelf = two_opt_improve(route_nn_shelf, coords_map)

    # Compute distance
    opt_dist = route_distance(route_opt_shelf, coords_map)

    return route_opt_shelf, opt_dist


if __name__ == '__main__':
    main()
