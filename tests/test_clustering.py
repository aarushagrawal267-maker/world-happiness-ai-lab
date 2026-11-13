import numpy as np
import pytest

from src.clustering import country_profiles, evaluate_k, recommend_k, run_clustering, scale
from src.config import INDICATORS, TARGET


@pytest.fixture(scope="module")
def result(panel):
    return run_clustering(panel)


def test_profiles_complete_and_unique(panel):
    prof, info = country_profiles(panel, [2020, 2021, 2022])
    assert prof[INDICATORS].notna().all().all()
    assert not prof["country"].duplicated().any()
    assert info["n_countries"] == len(prof) > 100


def test_k_evaluation_metrics(panel):
    prof, _ = country_profiles(panel)
    X, _ = scale(prof)
    s = evaluate_k(X, 2, 6)
    assert list(s["k"]) == [2, 3, 4, 5, 6]
    assert s["wcss"].is_monotonic_decreasing
    assert s["silhouette"].between(-1, 1).all() and (s["davies_bouldin"] > 0).all()
    k, reason = recommend_k(s)
    assert 2 <= k <= 6 and reason


def test_clustering_result(result):
    p = result.profiles
    assert p["cluster"].nunique() == result.k
    assert len(set(result.names.values())) == result.k
    assert 0 < result.silhouette <= 1
    assert sum(result.pca_variance) <= 1.0 + 1e-9
    assert -1 <= result.hierarchical_ari <= 1
    assert result.linkage_matrix.shape == (len(p) - 1, 4)


def test_clusters_ordered_by_ladder_but_ladder_not_used(result, panel):
    means = result.cluster_summary.sort_values("cluster")[TARGET].to_numpy()
    assert np.all(np.diff(means) <= 0)
    assert TARGET not in result.pca_loadings.index


def test_manual_k(panel):
    r = run_clustering(panel, k=4)
    assert r.k == 4 and r.profiles["cluster"].nunique() == 4
