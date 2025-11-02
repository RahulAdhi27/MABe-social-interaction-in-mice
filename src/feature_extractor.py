import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import fastparquet
from itertools import combinations

BASE_PATH = os.path.expanduser("~/Desktop/AI/MABe")
DATA_DIR = os.path.join(BASE_PATH, "data")
TRACKING_DIR = os.path.join(BASE_PATH, "MABe-mouse-behavior-detection", "train_tracking")
ANNOTATION_DIR = os.path.join(BASE_PATH, "MABe-mouse-behavior-detection", "train_annotation")

manifest_path = os.path.join(DATA_DIR, "trimmed_summary_filtered.csv")
print(f"Loading manifest from: {manifest_path}")

df_manifest = pd.read_csv(manifest_path)

schema_path = os.path.join(DATA_DIR, "feature_schema.json")
print(f"Loading schema from: {schema_path}")
with open(schema_path, "r") as f:
    feature_schema = json.load(f)

def load_tracking_file(folder, filename):
    path = os.path.join(TRACKING_DIR, folder, filename)
    if not os.path.exists(path):
        print(f"file missing: {path}")
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path, engine="fastparquet")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        print(f"Unsupported file: {ext}")
        return None
    
def get_frame_column(df):
    return "video_frame" if "video_frame" in df.columns else "frame"

def trim_to_valid_range(df, frame_col, start, stop):
    return df[(df[frame_col]>=start)&(df[frame_col]<stop)].copy()

def compute_kinematic_features(df):
    frame_col = "video_frame" if "video_frame" in df.columns else "frame"
    id_col = "mouse_id"
    df = df.sort_values(by=[id_col, frame_col]).copy()
    results = []
    for mouse_id, grp in df.groupby(id_col):
        grp = grp.sort_values(by=frame_col)
        grp["dx"] = grp["x"].diff()
        grp["dy"] = grp["y"].diff()
        grp["velocity"] = np.sqrt(grp["dx"]**2 + grp["dy"]**2)
        grp["acceleration"] = grp["velocity"].diff()
        grp["heading"] = np.arctan2(grp["dy"], grp["dx"])
        grp["turning_angle"] = grp["heading"].diff()
        grp["mouse_id"] = mouse_id
        results.append(grp)
    df_out = pd.concat(results, ignore_index=True)
    df_out = df_out.drop(columns=["dx", "dy"], errors="ignore")
    return df_out

def compute_spatial_features(df):
    frame_col = "video_frame"
    id_col = "mouse_id"
    df_centroids = (df.groupby([frame_col, id_col])[["x", "y"]].mean().reset_index())
    results = []
    mouse_ids = df_centroids[id_col].unique()
    pairs = list(combinations(mouse_ids, 2))
    for(id_a, id_b) in pairs:
        df_a = df_centroids[df_centroids[id_col]==id_a].rename(columns={"x": "x_a", "y": "y_a"})
        df_b = df_centroids[df_centroids[id_col] == id_b].rename(columns={"x": "x_b", "y": "y_b"})
        merged = pd.merge(df_a, df_b, on=frame_col, how="inner")
        merged["inter_mouse_distance"] = np.sqrt((merged["x_a"] - merged["x_b"])**2 + (merged["y_a"] - merged["y_b"])**2)
        merged["approach_speed"] = merged["inter_mouse_distance"].diff()
        if "heading" in df.columns:
            heading_a = df[df[id_col] == id_a][[frame_col, "heading"]].rename(columns={"heading": "heading_a"})
            heading_b = df[df[id_col] == id_b][[frame_col, "heading"]].rename(columns={"heading": "heading_b"})
            merged = merged.merge(heading_a, on=frame_col, how="left")
            merged = merged.merge(heading_b, on=frame_col, how="left")
            merged["relative_orientation"] = merged["heading_a"] - merged["heading_b"]
        else:
            merged["relative_orientation"] = np.nan
        merged["mouse_A"] = id_a
        merged["mouse_B"] = id_b
        results.append(merged)
    if len(results) == 0:
        return df
    df_spatial = pd.concat(results, ignore_index=True)
    return df_spatial

def compute_postural_features(df):
    frame_col = "video_frame" if "video_frame" in df.columns else "frame"
    id_col = "mouse_id"
    df["bodypart"] = df["bodypart"].astype(str).str.lower().str.strip()
    nose_candidates = ["nose", "snout", "head", "nose_tip"]
    tail_candidates = ["tailbase", "tail_base", "basetail", "tailtip", "tail_tip"]
    parts_available = df["bodypart"].unique().tolist()
    nose_part = next((p for p in nose_candidates if p in parts_available), None)
    tail_part = next((p for p in tail_candidates if p in parts_available), None)
    if not nose_part or not tail_part:
        print(f"Skipping postural features (missing parts) | found: {parts_available[:5]}...")
        return pd.DataFrame(columns=[frame_col, id_col, "body_length"])
    df_sub = df[df["bodypart"].isin([nose_part, tail_part])].copy()
    df_pivot = (
        df_sub.pivot_table(index=[frame_col, id_col], columns="bodypart", values=["x", "y"])
    )
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]
    df_pivot = df_pivot.reset_index()
    x_nose_col = f"x_{nose_part}"
    x_tail_col = f"x_{tail_part}"
    y_nose_col = f"y_{nose_part}"
    y_tail_col = f"y_{tail_part}"
    if not all(c in df_pivot.columns for c in [x_nose_col, x_tail_col, y_nose_col, y_tail_col]):
        print(f"Missing coordinate columns for nose/tail in this file. Skipping.")
        return pd.DataFrame(columns=[frame_col, id_col, "body_length"])
    df_pivot["body_length"] = np.sqrt(
        (df_pivot[x_nose_col] - df_pivot[x_tail_col]) ** 2 +
        (df_pivot[y_nose_col] - df_pivot[y_tail_col]) ** 2
    )
    return df_pivot[[frame_col, id_col, "body_length"]]

def main():
    out_base = os.path.join(DATA_DIR, "features_intermediate")
    os.makedirs(out_base, exist_ok=True)
    records = []
    for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="extracting features"):
        folder = row["folder"]
        filename = row["file"]
        start, stop = row["final_start"], row["final_stop"]
        df_track = load_tracking_file(folder, filename)
        if df_track is None:
            continue
        frame_col = get_frame_column(df_track)
        df_trimmed = trim_to_valid_range(df_track, frame_col, start, stop)
        df_kin = compute_kinematic_features(df_trimmed)
        df_pos = compute_postural_features(df_trimmed)
        df_features = pd.merge(df_kin, df_pos, on=["video_frame", "mouse_id"], how="left")
        df_spa = compute_spatial_features(df_kin)
        out_dir = os.path.join(out_base, folder)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        out_file = os.path.join(out_dir, f"{base_name}_features.parquet")
        df_features.to_parquet(out_file, index=False)
        records.append({
            "folder": folder,
            "file": filename,
            "frames_processed": df_features.shape[0],
            "features_computed": len(df_features.columns),
            "output_path": out_file
        })
    df_log = pd.DataFrame(records)
    log_path = os.path.join(DATA_DIR, "feature_extraction_log.csv")
    df_log.to_csv(log_path, index=False)
    print(f"Feature extraction complete. Log saved to: {log_path}")

if __name__ == "__main__":
    main()
