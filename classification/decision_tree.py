import numpy as np
from matplotlib import pyplot as plt


class Node(object):
    def __init__(self, feature=None, feature_index=None, threshold=None, left=None, right=None,
                 info_gain=None, value=None, value_name=None):
        # Decision node
        self.feature_index = feature_index
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # Leaf node
        self.value = value
        self.value_name = value_name


class DecisionTreeClassifier(object):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, *,
                 x_val: np.ndarray = None, y_val: np.ndarray = None, feature_names=None, class_names=None):
        # Save datasets
        # Num to Str mapping
        self.feature_names = feature_names
        self.class_names = class_names

        # Train sets
        self.x_train = x_train if x_train.ndim > 1 else x_train.reshape((-1, 1))  # Size of (samples, features)
        self.y_train = y_train if y_train.ndim > 1 else y_train.reshape((-1, 1))  # Size of (samples, outputs)

        # Validation sets
        if x_val is not None and y_val is not None:
            self.x_val = x_val if x_val.ndim > 1 else x_val.reshape((-1, 1))  # Size of (samples, features)
            self.y_val = y_val if y_val.ndim > 1 else y_val.reshape((-1, 1))  # Size of (samples, outputs)
        else:
            self.x_val = None
            self.y_val = None

        # Initialize the root of the tree
        self.root = None

        # Criteria
        self.splitting_criteria = None
        self.min_samples_split = None
        self.max_depth = None

        # Sizes
        self.n_samples, self.n_features = x_train.shape

    def __build_tree(self, dataset, curr_depth=0):
        """ recursive function to build the tree """

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # Split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Find the best split
            best_split = self.__get_best_split(dataset, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.__build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.__build_tree(best_split["dataset_right"], curr_depth + 1)

                # Get feature label
                feature_label = None if self.feature_names is None else self.feature_names[best_split["feature_index"]]

                # return decision node
                return Node(feature_label, best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # Compute leaf node
        leaf_value = self.mode(Y)

        # Return leaf node
        return Node(value=leaf_value, value_name=None if self.class_names is None else self.class_names[int(leaf_value)])

    def __get_best_split(self, dataset, num_features):
        """ Finds to find the best split """

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.__split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    @staticmethod
    def __split(dataset, feature_index, threshold):
        """ Splits the data """

        dataset_left = dataset[dataset[:, feature_index] <= threshold]
        dataset_right = dataset[dataset[:, feature_index] > threshold]
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        """ Computes information gain """

        # Compute weights
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        # Splitting quantity
        if self.splitting_criteria == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    @staticmethod
    def entropy(y):
        """ Computes entropy """

        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def gini_index(y):
        """ Computes gini index """

        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = np.sum(probabilities ** 2)
        return 1 - gini

    def plot_tree(self):
        plt.subplots(figsize=(8, 6))
        self.__gen_tree(self.root, x=0, y=0)
        plt.axis('off')
        plt.show()

    def __gen_tree(self, node: Node, depth=0, x=0, y=0):
        if node is None:
            return x, y

        if node.left is None and node.right is None:
            plt.text(x, y, f"{node.value_name if node.value_name else node.value}", ha='center', va='center')
            return x, y

        feature = f"x{node.feature_index}{f'[{node.feature}]' if node.feature else ''}"
        threshold = f"{node.threshold:.3f}"

        plt.text(x, y, f"{feature} <= {threshold}\nIG={node.info_gain:.3f}", ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        x_left, y_left = self.__gen_tree(node.left, depth+1, x-2**(10-depth), y-2)
        x_right, y_right = self.__gen_tree(node.right, depth+1, x+2**(10-depth), y-2)

        plt.plot([x, x_left], [y, y_left], 'b-')
        plt.plot([x, x_right], [y, y_right], 'b-')

        # Add 'yes' and 'no' labels to the links
        plt.text((x + x_left) / 2, (y + y_left) / 2, 'yes', ha='center', va='center', fontsize=12)
        plt.text((x + x_right) / 2, (y + y_right) / 2, 'no', ha='center', va='center', fontsize=12)

        return x, y

    def train(self, *, splitting_criteria="entropy", min_samples_split=2, max_depth=2):
        """ Trains the tree """
        # Set Hyper-parameters
        self.splitting_criteria = splitting_criteria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Train
        self.root = self.__build_tree(np.concatenate((self.x_train, self.y_train), axis=1))

    def predict(self, X):
        """ Predicts on new data """
        return [self.__go_through_nodes(x, self.root) for x in X]

    def __go_through_nodes(self, x, tree):
        """ Predicts a single data point """
        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.__go_through_nodes(x, tree.left)
        else:
            return self.__go_through_nodes(x, tree.right)

    @staticmethod
    def mode(arr):
        vals, counts = np.unique(arr, return_counts=True)
        index = np.argmax(counts)
        return vals[index]
