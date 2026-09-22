import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from pathlib import Path

'''
You want to select a smaller representative subset using k-means prototypes, but you don’t want to guess k. So you:

Split events into body and tail using a threshold (e.g., max(u) ≥ 0.99).

For each stratum (body, tail), you run k-means for a sequence of candidate k values (k_grid).

For each k, you compute two “representation quality” metrics:

coverage_p95: how well centroids cover the stratum

hist_L1: how well weighted selected points reproduce the stratum’s density

You stop increasing k when improvements become small for a few steps (“patience”) → that k is your “optimal”.
'''

# -----------------------------
# Metrics
# -----------------------------
def coverage_p95(U_stratum: np.ndarray, centers: np.ndarray) -> float:
    """p95 distance from each point to nearest centroid (coverage)."""
    d = pairwise_distances(U_stratum, centers)          # (n, k)
    return float(np.percentile(d.min(axis=1), 95))

def hist_L1(U_full: np.ndarray, U_sel: np.ndarray, weights: np.ndarray, bins=10) -> float:
    """L1 distance between normalized histogram of full stratum and weighted selected points."""
    edges = [np.linspace(0, 1, bins + 1)] * U_full.shape[1]
    H_full, _ = np.histogramdd(U_full, bins=edges)
    H_full = H_full / (H_full.sum() + 1e-12)

    # weighted histogram for selected
    w = weights / (weights.sum() + 1e-12)
    H_sel = np.zeros_like(H_full, dtype=float)

    # bin index per selected point
    idx = []
    for j in range(U_sel.shape[1]):
        b = np.digitize(U_sel[:, j], edges[j]) - 1
        b = np.clip(b, 0, bins - 1)
        idx.append(b)
    idx = np.vstack(idx).T

    for i in range(U_sel.shape[0]):
        H_sel[tuple(idx[i])] += w[i]

    H_sel = H_sel / (H_sel.sum() + 1e-12)
    return float(np.abs(H_sel - H_full).sum())

# -----------------------------
# Prototype selection
# -----------------------------
def kmeans_prototypes(U: np.ndarray, indices: np.ndarray, k: int, random_state=42):
    """
    Fit k-means on U[indices], return:
      chosen_global_indices (len=k), weights (len=k), centers
    """
    k = min(k, len(indices))
    X = U[indices]
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    closest_local, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    chosen_global = indices[closest_local]
    weights = np.bincount(km.labels_, minlength=k)
    return chosen_global, weights, km.cluster_centers_

# -----------------------------
# Auto-k search (diminishing returns)
# -----------------------------
def auto_k_for_stratum(U: np.ndarray, indices: np.ndarray, k_grid,
                       bins=6,
                       eps_cover=0.06,
                       eps_hist=0.06,
                       patience=2,
                       stop_mode="either",   # "either" (OR) or "both" (AND)
                       random_state=42):
    if len(indices) == 0:
        return 0, pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)

    k_grid = sorted(set([k for k in k_grid if 1 <= k <= len(indices)]))
    U_full = U[indices]

    rows = []
    prev_cover = None
    prev_L1 = None
    best = None

    streak = 0

    for k in k_grid:
        chosen, weights, centers = kmeans_prototypes(U, indices, k, random_state=random_state)
        U_sel = U[chosen]

        cover = coverage_p95(U_full, centers)
        L1 = hist_L1(U_full, U_sel, weights, bins=bins)

        rel_cover = None
        rel_L1 = None
        if prev_cover is not None and prev_cover > 0:
            rel_cover = (prev_cover - cover) / prev_cover
        if prev_L1 is not None and prev_L1 > 0:
            rel_L1 = (prev_L1 - L1) / prev_L1

        rows.append({
            "k": k,
            "cover_p95": cover,
            "hist_L1": L1,
            "rel_cover_improve": rel_cover,
            "rel_L1_improve": rel_L1
        })

        best = (k, chosen, weights)

        if (rel_cover is not None) and (rel_L1 is not None):
            cover_small = (rel_cover < eps_cover)
            hist_small  = (rel_L1   < eps_hist)

            if stop_mode == "both":
                plateau = cover_small and hist_small
            else:
                plateau = cover_small or hist_small   # <-- key change

            if plateau:
                streak += 1
            else:
                streak = 0

            if streak >= patience:
                return k, pd.DataFrame(rows), chosen, weights

        prev_cover, prev_L1 = cover, L1

    k, chosen, weights = best
    return k, pd.DataFrame(rows), chosen, weights


def find_optimal_lower_upper(
    df: pd.DataFrame,
    u_cols=("U_RF", "U_RD", "U_NTR"),
    raw_cols=("Sim_RF", "Sim_RD", "Sim_NTR"),
    q_upper=0.99,
    k_grid_body=None,
    k_grid_tail=None,
    bins=6,
    eps_cover=0.06,
    eps_hist=0.06,
    patience=2,               # <-- add
    stop_mode_body="either",  # <-- add
    stop_mode_tail="both",    # <-- add (tail is more sensitive; keep stricter)
    random_state=42
):
    U = df.loc[:, u_cols].to_numpy()
    s = U.max(axis=1)

    idx_tail = np.where(s >= q_upper)[0]
    idx_body = np.where(s <  q_upper)[0]

    if k_grid_body is None:
        k_grid_body = list(range(200, 2601, 200))
    if k_grid_tail is None:
        k_grid_tail = list(range(50, 801, 50))

    k_body_opt, diag_body, chosen_body, w_body = auto_k_for_stratum(
        U, idx_body, k_grid_body,
        bins=bins, eps_cover=eps_cover, eps_hist=eps_hist,
        patience=patience, stop_mode=stop_mode_body,
        random_state=random_state
    )

    k_tail_opt, diag_tail, chosen_tail, w_tail = auto_k_for_stratum(
        U, idx_tail, k_grid_tail,
        bins=bins, eps_cover=eps_cover, eps_hist=eps_hist,
        patience=patience, stop_mode=stop_mode_tail,
        random_state=random_state
    )

    body_df = df.loc[chosen_body, list(raw_cols) + list(u_cols)].copy()
    body_df["Weights"] = w_body

    tail_df = df.loc[chosen_tail, list(raw_cols) + list(u_cols)].copy()
    tail_df["Weights"] = w_tail

    diagnostics = {
        "q_upper": q_upper,
        "k_body_opt": k_body_opt,
        "k_tail_opt": k_tail_opt,
        "diag_body": diag_body,
        "diag_tail": diag_tail,
        "n_body_points": int(len(idx_body)),
        "n_tail_points": int(len(idx_tail))
    }
    return body_df, tail_df, diagnostics

def coreset_kmeans(
    variable_1_simulated_col_name,
    variable_2_simulated_col_name,
    variable_3_simulated_col_name,
    input_simulated_dataset_path,
    output_path_variable,
    n_clusters=1667,
    random_state=42,
    n_init=10,
):
    """
    Reduce a multivariate simulated-event dataset to representative *observed events*
    using K-Means in standardized 3-D space.

    One representative event (medoid-like point) is selected from each cluster.
    The representative event is the actual observation closest to that cluster's
    centroid *among observations assigned to that cluster*.

    The output weight is the number of original events represented by each selected
    event. ``Probability`` is the corresponding normalized cluster mass.

    Notes
    -----
    This is not weighted K-Means unless external sample weights are supplied to the
    clustering algorithm. Here, the weights are OUTPUT cluster masses used after
    clustering for weighted statistics/model aggregation.
    """

    cols = [
        variable_1_simulated_col_name,
        variable_2_simulated_col_name,
        variable_3_simulated_col_name,
    ]

    df = pd.read_csv(input_simulated_dataset_path)

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[cols].copy()

    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise ValueError(
            f"NaN values found in clustering columns {bad}. "
            "Remove or impute them before clustering."
        )

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Clustering columns contain inf or -inf values.")

    zero_variance = data.columns[data.std(axis=0) == 0].tolist()
    if zero_variance:
        raise ValueError(f"Zero-variance clustering columns: {zero_variance}")

    n_samples = len(data)
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed number of samples ({n_samples})."
        )

    # K-Means is distance-based, so standardize all three dimensions first.
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Select one REAL event from each cluster. Restricting the search to members of
    # that cluster guarantees that the representative event actually belongs to it.
    representative_indices = np.empty(n_clusters, dtype=int)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    for cluster_id in range(n_clusters):
        member_idx = np.flatnonzero(labels == cluster_id)

        # KMeans should not return empty clusters at convergence, but keep the guard
        # explicit so a malformed result cannot silently propagate.
        if member_idx.size == 0:
            raise RuntimeError(f"Cluster {cluster_id} is empty.")

        distances_sq = np.sum(
            (X[member_idx] - centers[cluster_id]) ** 2,
            axis=1,
        )
        representative_indices[cluster_id] = member_idx[np.argmin(distances_sq)]

    # Keep the complete original rows so event IDs / other forcing columns are not lost.
    selected = df.iloc[representative_indices].copy()
    selected.insert(0, "Original_Row", representative_indices)
    selected.insert(1, "Cluster_ID", np.arange(n_clusters))
    selected["Weight"] = cluster_sizes
    selected["Probability"] = cluster_sizes / n_samples

    # Useful consistency checks.
    if selected["Weight"].sum() != n_samples:
        raise RuntimeError("Cluster weights do not sum to the original sample count.")
    if not np.isclose(selected["Probability"].sum(), 1.0):
        raise RuntimeError("Representative probabilities do not sum to 1.")

    output_path = Path(output_path_variable)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    return selected


def visualize_representative_space_vs_entire_space(
    representative_space_path,
    entire_space_path,
    variable_1_simulated_col_name,
    variable_2_simulated_col_name,
    variable_3_simulated_col_name,
    variable_1_axis_name,
    variable_2_axis_name,
    variable_3_axis_name,
    weight_col="Weight",
    bins=60,
):
    """
    Compare the full simulated event space with one joint representative coreset.

    IMPORTANT: representative marginal histograms are weighted by cluster mass.
    An unweighted histogram of one point per cluster does not reproduce the original
    event distribution because every cluster would incorrectly count equally.
    """

    cols = [
        variable_1_simulated_col_name,
        variable_2_simulated_col_name,
        variable_3_simulated_col_name,
    ]

    df_large = pd.read_csv(entire_space_path)
    df_small = pd.read_csv(representative_space_path)

    for name, frame in [("entire dataset", df_large), ("representative dataset", df_small)]:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    if weight_col not in df_small.columns:
        raise ValueError(
            f"'{weight_col}' is missing from representative dataset. "
            "Weighted validation requires cluster weights."
        )

    weights = df_small[weight_col].to_numpy(dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Representative weights must be non-negative and sum to > 0.")

    xL = df_large[cols[0]].to_numpy()
    yL = df_large[cols[1]].to_numpy()
    zL = df_large[cols[2]].to_numpy()

    xS = df_small[cols[0]].to_numpy()
    yS = df_small[cols[1]].to_numpy()
    zS = df_small[cols[2]].to_numpy()

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[3, 1.4],
        height_ratios=[2, 3],
        wspace=0.35,
        hspace=0.35,
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[1, 0], projection="3d")
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_histz = fig.add_subplot(gs[0, 1])

    # Conventional coordinate mapping: x=var1, y=var2, z=var3.
    ax_3d.scatter(xL, yL, zL, s=18, alpha=0.15, label="All simulated")
    ax_3d.scatter(xS, yS, zS, s=8, alpha=0.8, label="Representative events")
    ax_3d.set_xlabel(variable_1_axis_name)
    ax_3d.set_ylabel(variable_2_axis_name)
    ax_3d.set_zlabel(variable_3_axis_name)
    ax_3d.legend(loc="upper left")

    # Use common bin edges for a valid visual comparison.
    x_edges = np.histogram_bin_edges(xL, bins=bins)
    y_edges = np.histogram_bin_edges(yL, bins=bins)
    z_edges = np.histogram_bin_edges(zL, bins=bins)

    ax_histx.hist(xL, bins=x_edges, density=True, alpha=0.55, label="All simulated")
    ax_histx.hist(
        xS,
        bins=x_edges,
        weights=weights,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Representative (weighted)",
    )
    ax_histx.set_xlabel(variable_1_axis_name)
    ax_histx.set_ylabel("Density")
    ax_histx.legend(fontsize=8)

    ax_histy.hist(
        yL,
        bins=y_edges,
        density=True,
        alpha=0.55,
        orientation="horizontal",
    )
    ax_histy.hist(
        yS,
        bins=y_edges,
        weights=weights,
        density=True,
        orientation="horizontal",
        histtype="step",
        linewidth=1.8,
    )
    ax_histy.set_ylabel(variable_2_axis_name)
    ax_histy.set_xlabel("Density")

    ax_histz.hist(zL, bins=z_edges, density=True, alpha=0.55)
    ax_histz.hist(
        zS,
        bins=z_edges,
        weights=weights,
        density=True,
        histtype="step",
        linewidth=1.8,
    )
    ax_histz.set_xlabel(variable_3_axis_name)
    ax_histz.set_ylabel("Density")

    plt.show()
    return fig


# Optional helper: use this only if you truly need an elbow diagnostic.
def plot_kmeans_inertia(
    input_simulated_dataset_path,
    variable_names,
    k_values,
    random_state=42,
    n_init=10,
):
    """Plot K-Means inertia versus k. This is diagnostic only; it does not choose k."""

    df = pd.read_csv(input_simulated_dataset_path)
    X_raw = df[list(variable_names)]

    if X_raw.isna().any().any():
        raise ValueError("NaN values found in clustering variables.")

    X = StandardScaler().fit_transform(X_raw)
    n_samples = len(X)

    valid_k = [int(k) for k in k_values if 1 <= int(k) <= n_samples]
    if not valid_k:
        raise ValueError("No valid k values were supplied.")

    inertias = []
    for k in valid_k:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(valid_k, inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means inertia diagnostic")
    plt.show()

    return pd.DataFrame({"k": valid_k, "inertia": inertias})
