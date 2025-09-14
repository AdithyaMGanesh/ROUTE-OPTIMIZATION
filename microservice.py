from flask import Flask, request, jsonify
from warehouse_route_visualization import find_shortest_path  # make sure this exists

app = Flask(__name__)

# ---- API ROUTE ----
@app.route("/optimize-route", methods=["POST"])
def optimize_route():
    try:
        data = request.get_json()
        shelf_ids = data.get("pick_list")

        if not shelf_ids:
            return jsonify({"error": "pick_list is required"}), 400

        optimized_path, total_distance = find_shortest_path(shelf_ids)

        return jsonify({
            "optimized_path": optimized_path,
            "total_distance": total_distance
        })

    except Exception as e:
        import traceback
        print("DEBUG - Error in optimize-route:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=6001)
