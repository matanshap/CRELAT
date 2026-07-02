
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class Cluster:
    def __init__(self, data, n_cluters):
        self.data = data
        self.n_cluters = n_cluters

    def kmeans(self):
        kmeans = KMeans(
                init="random",
                n_clusters=self.n_cluters,
                n_init=10,
                max_iter=300,
                random_state=42)

        # print("data: ", data)
        values = [a[2] for a in self.data]
        values = [[c] for c in values]
        if len(values)==1:
            clusters_labels=[0]
            clusters_centers=[0.9]
        else:
            kmeans.fit(values)
            labels = kmeans.labels_.tolist()
            clusters_labels = labels

            clusters_centers =  kmeans.cluster_centers_
        return clusters_labels, clusters_centers


    def dbscan(self, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        StandardScaler().fit(self.data)
        data_scaled = StandardScaler().transform(self.data)
        dbscan.fit(data_scaled)

        return dbscan.labels_, dbscan.cluster_centers_