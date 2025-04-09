# eco-courier-hub

Creating an eco-courier hub platform built on Python to optimize delivery routes and schedules involves several components, including data processing, machine learning, and user interface elements. The following example outlines a basic version of this system using Python with focus on modularity, error handling, and documentation through comments.

We'll use libraries like `scikit-learn` for machine learning and `Flask` for a simple web API/Interface.

```python
# eco_courier_hub.py
import json
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

def load_data(file_path):
    """Load delivery data from a given JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON.")
        return None

def optimize_routes(deliveries, num_clusters):
    """Optimize delivery routes using KMeans clustering."""
    try:
        locations = np.array([[d['latitude'], d['longitude']] for d in deliveries])
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(locations)
        clusters = kmeans.predict(locations)
        return clusters
    except Exception as e:
        print(f"Error in route optimization: {str(e)}")
        return None

def create_delivery_schedule(clusters, deliveries):
    """Create a delivery schedule based on clusters."""
    schedule = {}
    try:
        for i, cluster in enumerate(clusters):
            if cluster not in schedule:
                schedule[cluster] = []
            schedule[cluster].append(deliveries[i])
        return schedule
    except Exception as e:
        print(f"Error in creating delivery schedule: {str(e)}")
        return None

@app.route('/optimize', methods=['POST'])
def optimize():
    """API endpoint to receive and process delivery data."""
    try:
        data = request.get_json()
        num_clusters = data.get('num_clusters', 3)  # Default to 3 if not provided
        deliveries = data.get('deliveries', [])

        clusters = optimize_routes(deliveries, num_clusters)
        if clusters is None:
            return jsonify({"error": "Failed to optimize routes"}), 500

        schedule = create_delivery_schedule(clusters, deliveries)
        if schedule is None:
            return jsonify({"error": "Failed to create delivery schedule"}), 500

        return jsonify({"schedule": schedule}), 200
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    delivery_data_path = 'deliveries.json'
    deliveries = load_data(delivery_data_path)
    if deliveries:  # Only start the server if data is successfully loaded
        app.run(host='0.0.0.0', port=5000)
```

### Key Components:
- **Data Loading**: Using a JSON file to simulate delivery data input. There are basic file handling and JSON parsing error handling.
- **KMeans Clustering**: Utilizes the `scikit-learn` library for clustering delivery points into optimized route groups.
- **Schedule Creation**: Generates a dictionary mapping each cluster to the deliveries assigned to it.
- **Flask API**: Provides endpoints to receive delivery data and return optimized routes.

### Error Handling:
- Each part of the process captures exceptions and returns relevant error messages.
- The server only initiates if the data is successfully loaded to prevent unnecessary errors later.

This code is a simplified prototype and can be enhanced with additional features such as authentication, richer user interface, and persistent storage. Adjustments may also be needed for more complex routing algorithms, additional constraints, and real-world considerations (e.g., traffic data, package size restrictions).