"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from hierarchical_clustering.relaxed.datasets.hc_dataset import load_hc_data
from hierarchical_clustering.relaxed.utils.metrics import dasgupta_cost
from hierarchical_clustering.relaxed.utils.tree import to_nx_tree


dataset = ""
x, similarities = load_hc_data(dataset)
metrics = {}
for method in ["single", "complete", "average", "ward"]:
    metrics[method] = {}
    baseline_tree = to_nx_tree(linkage(squareform(1-similarities), method))
    dc = dasgupta_cost(baseline_tree, similarities)
    metrics[method]["DC"] = dc
print(metrics)