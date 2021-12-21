import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

LOGGER = logging.getLogger(__name__)


def plot_points(points, color, ax):
    ax.plot(points[:, 1], points[:, 0], '+', markerfacecolor=color,
            markeredgecolor=color, markersize=2)


def draw_contours(points, color, ax):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 1], points[simplex, 0], color)
    ax.plot(points[hull.vertices, 1], points[hull.vertices, 0], color)
    ax.plot(points[hull.vertices[0], 1], points[hull.vertices[0], 0], color)


def find_biggest_cluster(cluster_label_points):
    max_points_cluster_id = None
    max_points = 0
    for label, points in cluster_label_points.items():
        num_points = len(points)
        if num_points >= max_points:
            max_points_cluster_id = label
            max_points = num_points
    filtered_cluster = cluster_label_points[max_points_cluster_id]
    return filtered_cluster


def plot_dbscan_cluster(cluster_label_points, color=None, ax=None, clean_points=False):
    if not color:
        color = 'red'
    if not ax:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot()
    cluster_ids = list(cluster_label_points.keys())
    if len(cluster_ids) == 1 and -1 in cluster_ids:
        print('No clusters found by DBSCAN')
    else:
        if clean_points:
            filtered_cluster = find_biggest_cluster(cluster_label_points)
            plot_points(filtered_cluster, color, ax)
            # draw_contours(filtered_cluster, color, ax)
        else:
            for label, points in cluster_label_points.items():
                if label == -1:
                    continue
                plot_points(points, color, ax)
                # draw_contours(points, color, ax)

def dbscan_cluster(heatmap):
    # convert to black and white
    gray = np.uint8(heatmap * 255)
    # 204 because taking 0.8 <
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 204,
                                                 255, cv2.THRESH_BINARY)
    blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)
    # convert black pixels to coordinates
    X = np.column_stack(np.where(blackAndWhiteImage == 0))
    # Compute DBSCAN
    db = DBSCAN(eps=50, min_samples=31).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    LOGGER.info('Estimated number of clusters: %d' % n_clusters_)
    LOGGER.info('Estimated number of noise points: %d' % n_noise_)
    unique_labels = set(labels)
    cluster_label_points = dict()
    for label in unique_labels:
        class_member_mask = (labels == label)
        xy = X[class_member_mask & core_samples_mask]
        cluster_label_points[label] = xy
    return cluster_label_points
