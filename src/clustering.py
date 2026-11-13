"""Country clustering: K-Means, hierarchical (Ward) clustering and PCA.

Countries are profiled by the mean of their six core indicators over recent
panel years. Life Ladder is **not** used to form clusters; it is only used
afterwards to describe them. Cluster names are generated from each cluster's
actual feature profile. Clusters are a descriptive summary of similarity in
the chosen indicators — not objective categories of national happiness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import CFG, INDICATORS, RANDOM_STATE, SHORT_LABELS, TARGET

TRAIT_WORDS = {
    "log_gdp_per_capita": ("higher income", "lower income"),
    "social_support": ("strong social support", "weaker social support"),
    "healthy_life_expectancy": ("longer healthy lives", "shorter healthy lives"),
    "freedom": ("high freedom", "lower freedom"),
    "generosity": ("high generosity", "low generosity"),
    "perceptions_of_corruption": ("high perceived corruption", "low perceived corruption"),
}


def country_profiles(panel: pd.DataFrame, years: list[int] | None = None, features: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Mean indicator values per country over ``years``.

    A gap in the window is filled only from the *same country's* most recent
    earlier observation (flagged). Countries still incomplete are excluded and
    reported — no values are borrowed from other countries.
    """
    years = years or list(CFG["clustering"]["profile_years"])
    features = features or INDICATORS
    win = panel[panel["year"].isin(years)]
    prof = win.groupby("country")[features + [TARGET]].mean()
    meta = panel.sort_values("year").groupby("country")[["region", "iso3"]].last()
    filled = {}
    hist = panel[panel["year"] < min(years)].sort_values("year")
    for f in features:
        missing = prof.index[prof[f].isna()]
        for c in missing:
            past = hist.loc[hist["country"] == c, ["year", f]].dropna()
            if not past.empty:
                prof.at[c, f] = past[f].iloc[-1]
                filled.setdefault(c, []).append(f"{f} ({int(past['year'].iloc[-1])})")
    complete = prof.dropna(subset=features)
    excluded = sorted(set(prof.index) - set(complete.index))
    complete = complete.join(meta).reset_index()
    info = {
        "years": years,
        "n_countries": int(len(complete)),
        "excluded": excluded,
        "filled_from_own_history": filled,
        "countries_in_window": int(len(prof)),
    }
    return complete, info


def scale(profiles: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, StandardScaler]:
    features = features or INDICATORS
    sc = StandardScaler().fit(profiles[features])
    return sc.transform(profiles[features]), sc


def evaluate_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "wcss": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_)),
            "min_cluster_size": int(np.bincount(km.labels_).min()),
        })
    df = pd.DataFrame(rows)
    # consensus rank: high silhouette, low Davies-Bouldin, high Calinski-Harabasz
    df["rank_silhouette"] = df["silhouette"].rank(ascending=False)
    df["rank_db"] = df["davies_bouldin"].rank(ascending=True)
    df["rank_ch"] = df["calinski_harabasz"].rank(ascending=False)
    df["consensus_rank"] = df[["rank_silhouette", "rank_db", "rank_ch"]].mean(axis=1)
    return df


def recommend_k(scores: pd.DataFrame, min_size: int = 5) -> tuple[int, str]:
    """Consensus of three internal indices; k=2 is only chosen if it wins
    clearly, because a two-way split is rarely informative for profiling."""
    ok = scores[scores["min_cluster_size"] >= min_size]
    if ok.empty:
        ok = scores
    best = ok.sort_values(["consensus_rank", "silhouette"], ascending=[True, False]).iloc[0]
    k = int(best["k"])
    reason = (
        f"k = {k} has the best average rank across silhouette (↑), Davies–Bouldin (↓) and Calinski–Harabasz (↑) "
        f"among solutions whose smallest cluster has ≥ {min_size} countries."
    )
    if k == 2:
        alt = ok[ok["k"] > 2].sort_values("consensus_rank").iloc[0] if (ok["k"] > 2).any() else None
        if alt is not None and alt["silhouette"] >= 0.9 * best["silhouette"]:
            k = int(alt["k"])
            reason += f" k = 2 was replaced by k = {k}, whose silhouette is within 10% of k = 2 and gives a more informative profile."
    return k, reason


@dataclass
class ClusteringResult:
    profiles: pd.DataFrame          # one row per country, with cluster labels + PCA coords
    k: int
    k_reason: str
    scores: pd.DataFrame
    cluster_summary: pd.DataFrame   # mean features per cluster (+ ladder)
    cluster_z: pd.DataFrame         # cluster means in global z-units
    names: dict[int, str]
    pca_variance: list[float]
    pca_loadings: pd.DataFrame
    linkage_matrix: np.ndarray
    hierarchical_ari: float
    silhouette: float
    davies_bouldin: float
    info: dict


def name_clusters(z: pd.DataFrame) -> dict[int, str]:
    names = {}
    comp = z[[c for c in INDICATORS if c in z]].copy()
    if "perceptions_of_corruption" in comp:
        comp["perceptions_of_corruption"] *= -1
    level = comp.mean(axis=1)
    for cl, row in z.iterrows():
        traits = []
        for f in row.abs().sort_values(ascending=False).index:
            if f not in TRAIT_WORDS or abs(row[f]) < 0.45 or len(traits) >= 3:
                continue
            traits.append(TRAIT_WORDS[f][0] if row[f] > 0 else TRAIT_WORDS[f][1])
        lv = level[cl]
        tier = ("Very high indicators" if lv > 0.9 else "High indicators" if lv > 0.35 else "Mid-range indicators" if lv > -0.35
                else "Low indicators" if lv > -0.9 else "Very low indicators")
        names[cl] = f"{tier} · " + (", ".join(traits) if traits else "close to global average")
    # guarantee uniqueness
    seen: dict[str, int] = {}
    for cl in sorted(names):
        n = names[cl]
        if n in seen:
            seen[n] += 1
            names[cl] = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
    return names


def run_clustering(panel: pd.DataFrame, k: int | None = None, years: list[int] | None = None, features: list[str] | None = None) -> ClusteringResult:
    features = features or INDICATORS
    profiles, info = country_profiles(panel, years, features)
    X, _ = scale(profiles, features)
    scores = evaluate_k(X, int(CFG["clustering"]["k_min"]), int(CFG["clustering"]["k_max"]))
    rec_k, reason = recommend_k(scores)
    if k is None:
        k = rec_k
    else:
        reason = f"k = {k} selected manually (recommended: {rec_k}). " + reason
    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_
    # order clusters by mean Life Ladder (descriptive ordering only)
    order = pd.Series(profiles[TARGET].to_numpy()).groupby(labels).mean().sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    profiles["cluster"] = labels

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    profiles["hier_cluster"] = agg.labels_
    ari = float(adjusted_rand_score(labels, agg.labels_))
    Z = linkage(X, method="ward")

    pca = PCA(n_components=min(len(features), 6), random_state=RANDOM_STATE).fit(X)
    pcs = pca.transform(X)
    profiles["pc1"], profiles["pc2"] = pcs[:, 0], pcs[:, 1]
    loadings = pd.DataFrame(pca.components_.T[:, :3], index=features, columns=["PC1", "PC2", "PC3"][: min(3, pca.n_components_)])

    summary = profiles.groupby("cluster")[features + [TARGET]].mean()
    summary["n_countries"] = profiles.groupby("cluster").size()
    mu, sd = profiles[features].mean(), profiles[features].std(ddof=0)
    z = (summary[features] - mu) / sd
    names = name_clusters(z)
    profiles["cluster_name"] = profiles["cluster"].map(names)
    summary["name"] = summary.index.map(names)
    summary["example_countries"] = [", ".join(profiles[profiles["cluster"] == c].nlargest(5, TARGET)["country"]) for c in summary.index]

    return ClusteringResult(
        profiles=profiles,
        k=k,
        k_reason=reason,
        scores=scores,
        cluster_summary=summary.reset_index(),
        cluster_z=z,
        names=names,
        pca_variance=pca.explained_variance_ratio_.tolist(),
        pca_loadings=loadings,
        linkage_matrix=Z,
        hierarchical_ari=ari,
        silhouette=float(silhouette_score(X, labels)),
        davies_bouldin=float(davies_bouldin_score(X, labels)),
        info=info,
    )


def short(f: str) -> str:
    return SHORT_LABELS.get(f, f)
