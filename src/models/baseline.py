import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class Baseline(torch.nn.Module):
    def __init__(self, feature_extractor, extractor_dim=224, extractor_channels=3):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.grouping_model = None

        dummy_input = torch.randn(1, extractor_channels, extractor_dim, extractor_dim)
        with torch.no_grad():
            dummy_feature_map = self.feature_extractor(dummy_input)
        self.fe_dim = dummy_feature_map.flatten(start_dim=1).shape[1]

    def _get_feature_vector(self, inp):
        batch_size, num_images, c, h, w = inp.shape
        inp = inp.view(batch_size * num_images, c, h, w)
        feature_map = self.feature_extractor(inp)
        feature_vector = feature_map.view(batch_size, num_images, self.fe_dim)
        return feature_vector

    def get_groups_and_features(self, support_set, query_set, way):
        self.grouping_model = KMeans(n_clusters=way)
        support_features = self._get_feature_vector(support_set)
        query_features = self._get_feature_vector(query_set)

        support_features = support_features.view(-1, self.fe_dim).cpu().numpy()
        query_features = query_features.view(-1, self.fe_dim).cpu().numpy()

        all_features = np.concatenate([support_features, query_features], axis=0)
        all_groups = self.grouping_model.fit_predict(all_features)

        support_groups = all_groups[:support_features.shape[0]]
        query_groups = all_groups[support_features.shape[0]:]

        return support_groups, query_groups, support_features, query_features

    def calculate_accuracy(self, predicted_groups, true_labels):
        """Calculate clustering accuracy using the Hungarian method."""
        predicted_groups = np.array(predicted_groups)
        true_labels = np.array(true_labels)

        # Find the best cluster-to-label mapping
        way = len(np.unique(true_labels))
        cost_matrix = np.zeros((way, way))

        for i in range(way):
            for j in range(way):
                cost_matrix[i, j] = np.sum((predicted_groups == i) & (true_labels == j))

        row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)

        # Remap clusters to ground-truth labels
        cluster_mapping = {row: col for row, col in zip(row_ind, col_ind)}
        mapped_preds = np.array([cluster_mapping[p] for p in predicted_groups])

        # Compute accuracy
        accuracy = accuracy_score(true_labels, mapped_preds)
        return accuracy
