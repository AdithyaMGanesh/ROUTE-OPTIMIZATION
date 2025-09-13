# Warehouse Route Optimization Visualization

This project visualizes and optimizes order picking routes in a warehouse grid using Python. It generates a realistic warehouse layout, computes picking routes using Manhattan distance (no diagonal moves), and produces clear matplotlib visualizations.

## Features
- Warehouse grid: 19 aisles (A19..A01), 12 rows, shelves as vertical stacks
- Depot/start location at bottom-left
- Order input via CSV (`orders.csv`)
- Route optimization: Nearest Neighbor + 2-opt improvement
- Visualization: shelves, aisles, depot, picking route, crosshair highlights
- Outputs: PNG image (`warehouse_route.png`), route metrics JSON (`route_metrics.json`)

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib

Install dependencies:
```
pip install numpy pandas matplotlib
```

## Usage
1. Place your order data in `orders.csv` (format: order_id,item_id,shelf_id)
2. Run the script:
```
python warehouse_route_visualization.py --order ORD002 --csv orders.csv
```
- If no CSV is present, a sample order is generated automatically.
- The script prints route details and saves visualization/metrics files.

## Output Files
- `warehouse_route.png`: Warehouse layout and optimized picking route
- `route_metrics.json`: Route statistics (distance, improvement, sequence)

## Customization
- Edit `orders.csv` to test different orders
- Change warehouse size or layout in the script (`generate_layout` function)
- Use CLI arguments for different orders or CSV files

## Example Order CSV
```
order_id,item_id,shelf_id
ORD002,SKU101,A19_2
ORD002,SKU102,A17_11
ORD002,SKU103,A13_5
ORD002,SKU104,A09_8
ORD002,SKU105,A05_3
ORD002,SKU106,A01_10
```

## Visualization Details
- Shelves: white rectangles with black borders
- Depot: red square labeled "START"
- Route: green L-shaped segments (horizontal then vertical)
- Picked shelves: blue highlight
- Crosshair: red dot and dashed lines at selected shelf

## License
MIT
