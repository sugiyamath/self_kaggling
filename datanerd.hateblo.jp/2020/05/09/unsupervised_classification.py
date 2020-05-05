import pprint

from fastcluster import ward
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def count_labels(target):
    out = {}
    for t in target:
        if t not in out:
            out[t] = 0
        out[t] += 1
    return out


def clustering(images, metric="euclidean", t=1.15):
    X = images
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_pca = PCA(n_components=100).fit_transform(X)
    X_dist = ward(pdist(X_pca, metric=metric))
    clusters = fcluster(X_dist, t=t)
    return clusters


def count_clusters(clusters, target):
    out = [{} for _ in range(max(clusters))]
    for c, t in zip(clusters, target):
        if t not in out[c - 1]:
            out[c - 1][t] = 0
        out[c - 1][t] += 1
    return out


def calc_rate(cnt_cl, clabs):
    out = cnt_cl[:]
    for c, tdict in enumerate(cnt_cl):
        maxval = max(tdict.items(), key=lambda x: x[1] / clabs[x[0]])
        sumval = sum(x[1] for x in tdict.items())
        out[c] = (c, maxval[0], maxval[1], sumval)
    return out


def normalize_rate(rate, clabs, target_names):
    out = {}
    tns = set()
    for c, tid, mval, sval in rate:
        tn = target_names[tid]
        if tn not in out:
            out[tn] = [[], 0.0, 0.0]
        out[tn][0].append(c)
        out[tn][1] += mval
        out[tn][2] += sval
        tns.add((tn, tid))
    for tn, tid in tns:
        out[tn][2] = out[tn][1]/out[tn][2]
        out[tn][1] = out[tn][1]/clabs[tid]
    return out

def simplify(nrate):
    out = {}
    for tn, vs in nrate.items():
        out[tn] = {"recall": vs[1], "precision": vs[2]}
    return out


def main():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    clabs = count_labels(people.target)
    clusters = clustering(people.images, "euclidean", 0.9)
    cnt_cl = count_clusters(clusters, people.target)
    rate = calc_rate(cnt_cl, clabs)
    nrate = normalize_rate(rate, clabs, people.target_names)
    nrate = simplify(nrate)
    pprint.pprint(nrate)


if __name__ == "__main__":
    main()
