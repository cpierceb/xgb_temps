import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cmcrameri import cm

data_raw = pd.read_excel('climate_data.xlsx', index_col=0)

samples = data_raw.loc['samples'].copy()
samples.index = samples.index.str.capitalize()
samples.index = samples.index.str.replace('Novisad', 'Novi Sad')

data_raw = data_raw.drop('samples')

data = data_raw.T.reset_index()
data = data.rename(columns={'index': 'city'})
data['city'] = data['city'].str.capitalize()
data['city'] = data['city'].replace('Novisad', 'Novi Sad')

data['samples'] = data['city'].map(samples)

print("Samples check:")
print(data[['city', 'samples']].to_string(index=False))

X = data.drop(['city', 'samples'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)
print("\nClusters:")
print(data.groupby('cluster')['city'].apply(list))

dist_matrix = euclidean_distances(X_scaled)
n_cities = len(data)
samp = data['samples'].values.astype(float)
cities = data['city'].values

# --- Per-city: neighbor ranking + distance profile ---
print("\n" + "="*60)
city_profiles = {}
for i, city in enumerate(cities):
    dists = dist_matrix[i].copy()
    dists[i] = np.inf
    sorted_idx = np.argsort(dists)

    # Ranked neighbor list with raw distances
    neighbor_str = " > ".join(
        f"{cities[j]} ({dists[j]:.2f})" for j in sorted_idx
    )

    # Cumulative weighted average profile
    profile = []
    for k in range(1, n_cities):
        nearest_idx = sorted_idx[:k]
        w = samp[nearest_idx]
        d = dists[nearest_idx]
        profile.append(round(np.average(d, weights=w), 3))
    city_profiles[city] = profile

    print(f"\n{city}")
    print(f"  Neighbors: {neighbor_str}")
    print(f"  Weighted avg dist (k=1..{n_cities-1}): {profile}")

# --- Overall weighted average across all cities ---
print("\n" + "="*60)
print("Overall weighted average distance (weighted by source city samples):")
results = []
for k in range(1, n_cities):
    col_vals = [city_profiles[city][k-1] for city in cities]
    overall = np.average(col_vals, weights=samp)
    results.append({'k': k, 'weighted_avg_distance': round(overall, 4)})

# print(pd.DataFrame(results).to_string(index=False))

# --- CITY_RANKINGS dict output ---
print("\nCITY_RANKINGS = {")
for i, city in enumerate(cities):
    dists = dist_matrix[i].copy()
    dists[i] = np.inf
    sorted_idx = np.argsort(dists)[:-1]  # drop last entry (self)
    neighbors = [cities[j].lower().replace(' ', '') for j in sorted_idx]
    city_key = city.lower().replace(' ', '')
    print(f"    '{city_key}': {neighbors},")
print("}")

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,          # your original climate feature names
    columns=['PC1', 'PC2']
).round(3)

print(loadings.sort_values('PC1', key=abs, ascending=False))


# ── Consistent city colour palette (batlow, 12 cities) ──────────────────────
_ALL_CITIES_12 = [
    "Amsterdam", "Basel", "Berlin", "Bern", "Biel", "Birmingham",
    "Freiburg", "Ghent", "Novi Sad", "Rennes", "Turku", "Zürich",
    # "Oslo", "Naples", "Lyon"
]
_n = len(_ALL_CITIES_12)
COLOR_MAP = {city: cm.batlow(i / (_n - 1)) for i, city in enumerate(_ALL_CITIES_12)}
 
MARKERS = {
    "Amsterdam" : "D", 
    "Basel" : "v", 
    "Berlin" : "o", 
    "Bern" : "^", 
    "Biel" : "s", 
    "Birmingham" : "<",
    "Freiburg" : "p", 
    "Ghent" : ">", 
    "Novi Sad" : "H", 
    "Rennes" : "P", 
    "Turku" : "D", 
    "Zürich" : "X",
}

 

def plot_kmeans_clusters(X_scaled, X, cities, cluster_labels, save_path=None, dpi=600):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.size': 12})

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    n_clusters = len(np.unique(cluster_labels))
    hull_color = cm.glasgow(0.5)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # --- Convex hulls ---
    for c in range(n_clusters):
        mask = cluster_labels == c
        pts  = coords[mask]
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = plt.Polygon(
                pts[hull.vertices],
                closed=True,
                facecolor=hull_color, alpha=0.10,
                edgecolor=hull_color, linewidth=1.2,
                zorder=1,
            )
            ax.add_patch(poly)
        elif len(pts) == 2:
            ax.plot(pts[:, 0], pts[:, 1],
                    color=hull_color, linewidth=1.2, alpha=0.4, zorder=1)

    # --- Scatter ---
    # Normalize city names to handle encoding quirks (e.g. Zürich variants)
    def _norm(s):
        return s.strip().replace('\u00fc', 'ü').replace('Zurich', 'Zürich')

    for i, city in enumerate(cities):
        color = COLOR_MAP.get(_norm(city), '0.5')
        ax.scatter(coords[i, 0], coords[i, 1], marker = MARKERS.get(city),
                   color=color, s=70, zorder=3, linewidths=0)

    # --- Manual label offsets (dx, dy) in data coordinates ---
    label_offsets = {
        "Amsterdam":  ( 0.05,  0.10),
        "Basel":      (-0.15,  0.15),
        "Berlin":     (-0.55, -0.20),
        "Bern":       (-0.45, -0.25),
        "Biel":       ( 0.05,  0.10),
        "Birmingham": ( 0.05,  0.10),
        "Freiburg":   ( 0.00, -0.30),
        "Ghent":      (-0.70,  0.05),
        "Novi Sad":   ( 0.10, -0.15),
        "Rennes":     ( 0.12, -0.01),
        "Turku":      ( 0.05, -0.20),
        "Zürich":     ( 0.05,  0.12),
        # "Oslo":       ( 0.05,  0.12),
        # "Lyon":     ( 0.05,  0.12),
        # "Naples":     ( 0.05,  0.12), 
    }

    for i, city in enumerate(cities):
        city_n = _norm(city)
        color  = COLOR_MAP.get(city_n, '0.5')
        dx, dy = label_offsets.get(city_n, (0.05, 0.05))
        ax.text(
            coords[i, 0] + dx, coords[i, 1] + dy,
            city_n, fontsize=10, color=color, zorder=4,
        )

    ax.set_xlabel(f"PC 1  ({var[0]:.1f} % variance)")
    ax.set_ylabel(f"PC 2  ({var[1]:.1f} % variance)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    lim = abs(coords).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

# # --- PC loadings table (top right corner) ---
#     feature_labels = ["Dist. to coast", "Urban area", "Elevation", "Elevation SD"]
#     loadings = pca.components_  # (2, n_features)

#     col_x   = [0.72, 0.82, 0.92]   # x positions for PC1, PC2, feature name
#     row_y0  = 0.97
#     row_dy  = 0.055
#     fc      = '0.35'
#     fsize   = 9

#     # Header
#     for x, label in zip(col_x, ["PC1", "PC2", ""]):
#         ax.text(x, row_y0, label, transform=ax.transAxes,
#                 fontsize=fsize, va='top', ha='right', color=fc, style='italic')

#     # Rows
#     for j, feat in enumerate(feature_labels):
#         y = row_y0 - (j + 1) * row_dy
#         ax.text(col_x[0], y, f"{loadings[0,j]:+.2f}", transform=ax.transAxes,
#                 fontsize=fsize, va='top', ha='right', color=fc)
#         ax.text(col_x[1], y, f"{loadings[1,j]:+.2f}", transform=ax.transAxes,
#                 fontsize=fsize, va='top', ha='right', color=fc)
#         ax.text(col_x[2], y, feat, transform=ax.transAxes,
#                 fontsize=fsize, va='top', ha='right', color=fc)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    return fig, ax
 

 # ── Drop-in call (place after kmeans.fit_predict in your script) ─────────────
plot_kmeans_clusters(X_scaled, X, cities, data['cluster'].values, save_path='kmeans_clusters.png')