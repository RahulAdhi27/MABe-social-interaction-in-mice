import json, os

core_feature_schema = {
    "kinematic": {
        "velocity": {
            "formula": "v_t = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)",
            "description": "Instantaneous speed of each mouse per frame"
        },
        "acceleration": {
            "formula": "a_t = v_t - v_{t-1}",
            "description": "Change in velocity between consecutive frames"
        },
        "turning_angle": {
            "formula": "Δθ_t = heading_t - heading_{t-1}",
            "description": "Angular change in movement direction"
        }
    },

    "spatial": {
        "inter_mouse_distance": {
            "formula": "d_t = ||centroid_A_t - centroid_B_t||",
            "description": "Distance between mouse-A and mouse-B centroids"
        },
        "approach_speed": {
            "formula": "Δd_t = d_t - d_{t-1}",
            "description": "Change of inter-mouse distance (negative = approach)"
        },
        "relative_orientation": {
            "formula": "θ_rel_t = heading_A_t - heading_B_t",
            "description": "Difference in facing directions between two mice"
        }
    },

    "postural": {
        "body_length": {
            "formula": "L_t = ||nose_t - tailbase_t||",
            "description": "Length of body axis, per frame"
        }
    }
}

path = os.path.expanduser("~/Desktop/AI/MABe/data/feature_schema.json")
with open(path, "w") as f:
    json.dump(core_feature_schema, f, indent=4)

print(f"Core feature schema written to: {path}")
