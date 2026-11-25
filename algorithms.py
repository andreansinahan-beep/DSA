"""
Loan Approval Prediction System - algorithms
CC 204: Data Structures and Algorithms (WVSU - CICT)

Three classifiers written from scratch:
    KNN           -> list of records + min-heap (heapq)
    DecisionTree  -> binary tree of Node objects, recursive Gini splitting
    NaiveBayes    -> hash maps (dict / Counter) of frequency counts

No machine learning library is used. numpy is used only for np.log1p.

Command line:
    python algorithms.py data/train.csv
"""

import csv
import heapq
import math
import os
import random
import statistics
import sys
import time
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Column definitions (tuples, because the column lists never change)
# ---------------------------------------------------------------------------
TARGET = "Loan_Status"

# Text columns from the CSV (missing values are filled with the MODE)
TEXT_COLUMNS = ("Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area")

# Number columns from the CSV (missing values are filled with the MEDIAN)
NUMBER_COLUMNS = ("ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                  "Loan_Amount_Term", "Credit_History")

# How the models treat each feature. Credit_History is stored as a number
# but only has two values (1 = good, 0 = bad), so the Decision Tree and
# Naive Bayes treat it as a category.
CATEGORICAL_FEATURES = TEXT_COLUMNS + ("Credit_History",)
NUMERIC_FEATURES = ("ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                    "Loan_Amount_Term", "TotalIncome")

REQUIRED_COLUMNS = TEXT_COLUMNS + NUMBER_COLUMNS + (TARGET,)

SEEDS = (43, 44, 45, 46, 47)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
def is_missing(value):
    """Return True if a cell is empty.

    Data structure: none (checks a single value). A tiny helper so that every
    part of the project agrees on what "missing" means.
    """
    return value is None or str(value).strip() == ""


def load_csv(path):
    """Read train.csv into a list of dictionaries.

    Data structure: LIST of DICTS. The list keeps the rows in file order, and
    each dict (hash map) gives O(1) access to a column by its name.
    Time: O(n*d) for n rows and d columns.
    """
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        absent = [c for c in REQUIRED_COLUMNS if c not in header]
        if absent:
            raise ValueError("The CSV is missing these columns: " + ", ".join(absent))
        rows = list(reader)
    if not rows:
        raise ValueError("The CSV has no data rows.")
    return rows


def count_missing(rows):
    """Count the empty cells in every column.

    Data structure: DICT mapping column name -> count, for O(1) updates.
    """
    counts = {column: 0 for column in rows[0]}
    for row in rows:
        for column in counts:
            if is_missing(row.get(column)):
                counts[column] += 1
    return counts


def preprocess(rows):
    """Clean the raw rows and return a NEW list (the input is not changed).

    1. Missing text values are filled with the mode (most common value).
    2. Missing numbers are filled with the median (middle value).
    3. Numbers are converted to float.
    4. TotalIncome = ApplicantIncome + CoapplicantIncome is added.
    Rows without a Loan_Status are dropped, because the answer we are trying
    to learn must never be guessed.

    Data structure: DICT of fill values (column -> value); a Counter (hash map)
    finds each mode in O(n). Finding each median sorts the values, O(n log n).
    """
    labelled = [r for r in rows if not is_missing(r.get(TARGET))]
    if len(labelled) < len(rows):
        print(f"Dropped {len(rows) - len(labelled)} rows with no {TARGET}.")

    fill = {}
    for column in TEXT_COLUMNS:
        present = [r[column].strip() for r in labelled if not is_missing(r[column])]
        if not present:
            raise ValueError(f"Column {column} has no values.")
        fill[column] = Counter(present).most_common(1)[0][0]
    for column in NUMBER_COLUMNS:
        present = [float(r[column]) for r in labelled if not is_missing(r[column])]
        if not present:
            raise ValueError(f"Column {column} has no values.")
        fill[column] = statistics.median(present)

    cleaned = []
    filled_cells = 0
    for row in labelled:
        new_row = dict(row)  # copy so the raw data stays untouched
        for column in TEXT_COLUMNS + NUMBER_COLUMNS:
            if is_missing(new_row[column]):
                new_row[column] = fill[column]
                filled_cells += 1
        for column in TEXT_COLUMNS:
            new_row[column] = str(new_row[column]).strip()
        for column in NUMBER_COLUMNS:
            new_row[column] = float(new_row[column])
        new_row[TARGET] = new_row[TARGET].strip()
        new_row["TotalIncome"] = new_row["ApplicantIncome"] + new_row["CoapplicantIncome"]
        cleaned.append(new_row)

    print(f"Preprocessing: filled {filled_cells} missing cells in {len(cleaned)} rows.")
    return cleaned


def split(rows, seed, test_size=0.2):
    """Stratified train/test split. Returns (train_rows, test_rows).

    "Stratified" means the Y/N proportion is kept the same in both parts.

    Data structure: DICT of LISTS (class label -> rows of that class). Each
    list is shuffled with a seeded random generator, so the same seed always
    gives the same split. Time: O(n).
    """
    groups = {}
    for row in rows:
        groups.setdefault(row[TARGET], []).append(row)

    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(groups):
        members = list(groups[label])
        rng.shuffle(members)
        n_test = round(len(members) * test_size)
        test.extend(members[:n_test])
        train.extend(members[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ---------------------------------------------------------------------------
# 1. K-Nearest Neighbors
# ---------------------------------------------------------------------------
class KNN:
    """K-Nearest Neighbors classifier.

    Data structures:
      * LIST of training points - KNN has no real model; it remembers the data.
      * MIN-HEAP (heapq.nsmallest) - picks the k smallest distances in
        O(n log k), which is cheaper than sorting all n distances O(n log n).
    """

    def __init__(self, k=7, credit_weight=5.0):
        """Store the settings. k = number of neighbors; credit_weight makes a
        difference in Credit_History count more than other features.

        Data structure: plain attributes, a list for points, dict for ranges.
        """
        self.k = k
        self.credit_weight = credit_weight
        self.ranges = {}   # feature -> (minimum, maximum)
        self.points = []   # list of (numeric_list, category_list, record)

    def fit(self, rows):
        """Learn each feature's min and max, then store normalized points.

        Data structure: DICT for the ranges, LIST for the points.
        Time: O(n*d), done once so that predicting does not repeat this work.
        """
        for feature in NUMERIC_FEATURES:
            column = [row[feature] for row in rows]
            self.ranges[feature] = (min(column), max(column))
        self.points = [(*self._encode(row), row) for row in rows]
        return self

    def _encode(self, record):
        """Turn a record into (normalized numbers, category values).

        Min-max normalization: (x - min) / (max - min), so that large columns
        such as income do not dominate the distance.

        Data structure: two LISTS in a fixed feature order.
        """
        numbers = []
        for feature in NUMERIC_FEATURES:
            low, high = self.ranges[feature]
            numbers.append((record[feature] - low) / (high - low) if high > low else 0.0)
        categories = [record[feature] for feature in CATEGORICAL_FEATURES]
        return numbers, categories

    def _distance(self, a, b):
        """Weighted Euclidean distance between two encoded records.

        Numbers add (difference)^2. Categories add 0 if equal and 1 if
        different. A Credit_History mismatch is multiplied by credit_weight.

        Data structure: walks two parallel LISTS with zip, O(d).
        """
        total = 0.0
        for x, y in zip(a[0], b[0]):
            total += (x - y) ** 2
        for feature, x, y in zip(CATEGORICAL_FEATURES, a[1], b[1]):
            if x != y:
                total += self.credit_weight if feature == "Credit_History" else 1.0
        return math.sqrt(total)

    def distance(self, record_a, record_b):
        """Distance between two cleaned records (dicts).

        Data structure: encodes both DICTS into lists, then compares them.
        """
        return self._distance(self._encode(record_a), self._encode(record_b))

    def neighbors(self, record):
        """Return the k nearest neighbors as a list of dicts with the keys
        'distance', 'label' and 'record', nearest first.

        Data structure: MIN-HEAP via heapq.nsmallest over (distance, index)
        tuples. The index breaks ties between equal distances.
        Time: O(n*d) for the distances + O(n log k) for the heap.
        """
        if not self.points:
            raise ValueError("Call fit() before predicting.")
        query = self._encode(record)
        distances = ((self._distance(query, point), index)
                     for index, point in enumerate(self.points))
        nearest = heapq.nsmallest(self.k, distances)
        return [{"distance": dist,
                 "label": self.points[index][2][TARGET],
                 "record": self.points[index][2]}
                for dist, index in nearest]

    def predict_one(self, record):
        """Predict 'Y' or 'N' by majority vote of the k neighbors.

        Data structure: Counter (hash map) of neighbor labels.
        """
        votes = Counter(n["label"] for n in self.neighbors(record))
        return "Y" if votes["Y"] > votes["N"] else "N"

    def predict(self, rows):
        """Predict every record. Data structure: returns a LIST of labels."""
        return [self.predict_one(row) for row in rows]


# ---------------------------------------------------------------------------
# 2. Decision Tree
# ---------------------------------------------------------------------------
class Node:
    """One node of the binary decision tree.

    Data structure: BINARY TREE NODE. An internal node holds a question
    (feature + threshold) and two children; a leaf holds only a label.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        """Create a node. left = answer is yes, right = answer is no."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        """A leaf is a node that carries a label (it has no children)."""
        return self.label is not None


def gini(counts, total):
    """Gini impurity = 1 - sum(p^2). 0 means the group is pure.

    Data structure: takes a Counter (hash map) of label -> count.
    """
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


class DecisionTree:
    """Decision Tree classifier built recursively with Gini impurity.

    Data structure: BINARY TREE of Node objects. Predicting follows a single
    path from the root to a leaf, so it costs only O(depth).
    """

    def __init__(self, max_depth=5, min_size=10):
        """max_depth limits the height; a node with fewer than min_size rows
        becomes a leaf. Both limits reduce overfitting.

        Data structure: only the root pointer is stored.
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    def fit(self, rows):
        """Build the tree from the training rows.

        Data structure: starts the recursive construction of the BINARY TREE.
        """
        self.root = self._build(rows, 0)
        return self

    def _build(self, rows, depth):
        """Recursively build and return the subtree for these rows.

        Base cases (make a leaf): maximum depth reached, too few rows, all
        rows have the same label, or no split lowers the impurity.

        Data structure: RECURSION (the call stack) creates the tree; each
        child receives its own LIST of rows.
        """
        counts = Counter(row[TARGET] for row in rows)
        majority = "Y" if counts["Y"] > counts["N"] else "N"

        if depth >= self.max_depth or len(rows) < self.min_size or len(counts) == 1:
            return Node(label=majority)

        best = self._best_split(rows, counts)
        if best is None:
            return Node(label=majority)

        feature, threshold = best
        yes_rows, no_rows = [], []
        for row in rows:
            (yes_rows if self._goes_left(row, feature, threshold) else no_rows).append(row)

        return Node(feature=feature, threshold=threshold,
                    left=self._build(yes_rows, depth + 1),
                    right=self._build(no_rows, depth + 1))

    def _best_split(self, rows, parent_counts):
        """Try every feature and threshold; return the (feature, threshold)
        with the lowest weighted Gini, or None if nothing improves.

        Data structures: SET for the unique values of a feature, sorted into a
        LIST; Counter for the label counts on each side.
        Time: O(d * v * n), where v is the number of unique values. In the
        worst case v = n, which gives O(d * n^2).
        """
        total = len(rows)
        best_score = gini(parent_counts, total)
        best = None

        for feature in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
            values = sorted({row[feature] for row in rows})
            if len(values) < 2:
                continue  # every row has the same value, nothing to split
            if feature in NUMERIC_FEATURES:
                # thresholds are the midpoints between neighboring values
                thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
            else:
                thresholds = values

            for threshold in thresholds:
                left = Counter(row[TARGET] for row in rows
                               if self._goes_left(row, feature, threshold))
                n_left = sum(left.values())
                n_right = total - n_left
                if n_left == 0 or n_right == 0:
                    continue
                right = parent_counts - left
                score = (n_left * gini(left, n_left) +
                         n_right * gini(right, n_right)) / total
                if score < best_score - 1e-12:
                    best_score, best = score, (feature, threshold)
        return best

    @staticmethod
    def _goes_left(record, feature, threshold):
        """Answer a node's question: numbers use <=, categories use ==.

        Data structure: one O(1) DICT lookup in the record.
        """
        if feature in NUMERIC_FEATURES:
            return record[feature] <= threshold
        return record[feature] == threshold

    def predict_one(self, record):
        """Walk from the root to a leaf and return its label.

        Data structure: TREE TRAVERSAL along one path, O(depth).
        """
        if self.root is None:
            raise ValueError("Call fit() before predicting.")
        node = self.root
        while not node.is_leaf():
            go_left = self._goes_left(record, node.feature, node.threshold)
            node = node.left if go_left else node.right
        return node.label

    def predict(self, rows):
        """Predict every record. Data structure: returns a LIST of labels."""
        return [self.predict_one(row) for row in rows]

    def rule_path(self, record):
        """Return the questions answered from root to leaf as one string, e.g.
        'Credit_History == 1.0 -> TotalIncome <= 5200 -> APPROVED'.

        Data structure: LIST of rule strings collected during a root-to-leaf
        TREE TRAVERSAL, then joined.
        """
        if self.root is None:
            raise ValueError("Call fit() before predicting.")
        node = self.root
        steps = []
        while not node.is_leaf():
            go_left = self._goes_left(record, node.feature, node.threshold)
            if node.feature in NUMERIC_FEATURES:
                sign = "<=" if go_left else ">"
                value = f"{node.threshold:g}"
            else:
                sign = "==" if go_left else "!="
                value = str(node.threshold)
            steps.append(f"{node.feature} {sign} {value}")
            node = node.left if go_left else node.right
        steps.append("APPROVED" if node.label == "Y" else "REJECTED")
        return " -> ".join(steps)

    def depth_and_nodes(self):
        """Return (depth, node_count). Depth counts the edges on the longest
        root-to-leaf path; node_count includes the leaves.

        Data structure: recursive (post-order) TREE TRAVERSAL, O(nodes).
        """
        def walk(node):
            """Return (depth, nodes) of one subtree."""
            if node.is_leaf():
                return 0, 1
            left_depth, left_nodes = walk(node.left)
            right_depth, right_nodes = walk(node.right)
            return 1 + max(left_depth, right_depth), 1 + left_nodes + right_nodes

        if self.root is None:
            return 0, 0
        return walk(self.root)


# ---------------------------------------------------------------------------
# 3. Naive Bayes
# ---------------------------------------------------------------------------
class NaiveBayes:
    """Naive Bayes classifier.

    Data structures: HASH MAPS (dict and Counter). Training only counts things,
    and predicting looks the counts up in O(1) each.
    Categorical features use Laplace-smoothed frequencies; numeric features use
    a Gaussian (bell curve) on log1p(x), which reduces the skew of incomes.
    """

    def __init__(self):
        """Create the empty hash maps that fit() will fill."""
        self.total = 0
        self.class_counts = Counter()   # label -> number of rows
        self.value_counts = {}          # label -> feature -> Counter(value)
        self.vocabulary = {}            # feature -> set of distinct values
        self.gaussians = {}             # label -> feature -> (mean, variance)

    def fit(self, rows):
        """Count class priors and feature values, then compute mean/variance.

        Data structure: nested DICTS of Counters; a SET per feature records
        its distinct values (needed by Laplace smoothing).
        Time: O(n*d).
        """
        self.total = len(rows)
        self.class_counts = Counter(row[TARGET] for row in rows)
        self.vocabulary = {f: set() for f in CATEGORICAL_FEATURES}
        self.value_counts = {c: {f: Counter() for f in CATEGORICAL_FEATURES}
                             for c in self.class_counts}
        logs = {c: {f: [] for f in NUMERIC_FEATURES} for c in self.class_counts}

        for row in rows:
            label = row[TARGET]
            for feature in CATEGORICAL_FEATURES:
                self.value_counts[label][feature][row[feature]] += 1
                self.vocabulary[feature].add(row[feature])
            for feature in NUMERIC_FEATURES:
                logs[label][feature].append(float(np.log1p(row[feature])))

        self.gaussians = {}
        for label, features in logs.items():
            self.gaussians[label] = {}
            for feature, values in features.items():
                mean = statistics.fmean(values)
                variance = statistics.fmean([(v - mean) ** 2 for v in values])
                # the floor prevents division by zero if all values are equal
                self.gaussians[label][feature] = (mean, max(variance, 1e-9))
        return self

    def log_scores(self, record):
        """Return {label: log P(label) + sum of log P(feature | label)}.

        Logarithms are added instead of multiplying many small probabilities,
        which could round down to zero.

        Data structure: DICT of scores; every count is an O(1) hash lookup.
        Time: O(c*d).
        """
        if not self.total:
            raise ValueError("Call fit() before predicting.")
        scores = {}
        for label, class_count in self.class_counts.items():
            score = math.log(class_count / self.total)  # class prior
            for feature in CATEGORICAL_FEATURES:
                seen = self.value_counts[label][feature][record[feature]]
                options = len(self.vocabulary[feature])
                # Laplace smoothing: add 1 so no probability is ever zero
                score += math.log((seen + 1) / (class_count + options))
            for feature in NUMERIC_FEATURES:
                mean, variance = self.gaussians[label][feature]
                x = float(np.log1p(record[feature]))
                score += (-0.5 * math.log(2 * math.pi * variance)
                          - (x - mean) ** 2 / (2 * variance))
            scores[label] = score
        return scores

    def predict_one(self, record):
        """Return the label with the highest score.

        Data structure: max over the DICT of scores.
        """
        scores = self.log_scores(record)
        return max(scores, key=scores.get)

    def predict(self, rows):
        """Predict every record. Data structure: returns a LIST of labels."""
        return [self.predict_one(row) for row in rows]


# The three models, in one DICT (name -> class) so they can be looped over
MODELS = {"KNN": KNN, "Decision Tree": DecisionTree, "Naive Bayes": NaiveBayes}


def train_models(rows):
    """Fit all three models on the given rows.

    Data structure: DICT of name -> fitted model object.
    """
    return {name: model_class().fit(rows) for name, model_class in MODELS.items()}


def majority_vote(predictions):
    """Return the label chosen by most models (3 voters, so no ties).

    Data structure: Counter (hash map) of votes.
    """
    votes = Counter(predictions)
    return "Y" if votes["Y"] > votes["N"] else "N"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """Return accuracy, precision, recall, F1 and TP, TN, FP, FN.
    "Y" (Approved) is the positive class.

    Data structure: Counter keyed by (actual, predicted) TUPLES, which counts
    the whole confusion matrix in one O(n) pass.
    """
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equally long.")
    pairs = Counter(zip(y_true, y_pred))
    tp, tn = pairs[("Y", "Y")], pairs[("N", "N")]
    fp, fn = pairs[("N", "Y")], pairs[("Y", "N")]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": precision,
            "recall": recall, "f1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def benchmark(rows, seeds=SEEDS):
    """Run one trial per seed (43-47): split, fit, predict, measure.

    Returns {"trials": [...], "summary": {...}}.
      trials  - one dict per (seed, algorithm)
      summary - per algorithm: MEAN of the four metrics, MEDIAN of the two
                times (a median ignores one unusually slow run), and the
                confusion matrix counts ADDED over all trials.
    fit_ms is the time to train; predict_ms is the time to predict the WHOLE
    test set. Input rows must already be preprocessed.

    Data structure: LIST of result dicts, then a DICT keyed by algorithm.
    """
    trials = []
    for seed in seeds:
        train, test = split(rows, seed)
        truth = [row[TARGET] for row in test]
        for name, model_class in MODELS.items():
            model = model_class()
            start = time.perf_counter()
            model.fit(train)
            fitted = time.perf_counter()
            predictions = model.predict(test)
            done = time.perf_counter()
            result = {"seed": seed, "algorithm": name,
                      "train_rows": len(train), "test_rows": len(test)}
            result.update(metrics(truth, predictions))
            result["fit_ms"] = (fitted - start) * 1000
            result["predict_ms"] = (done - fitted) * 1000
            trials.append(result)

    summary = {}
    for name in MODELS:
        own = [t for t in trials if t["algorithm"] == name]
        entry = {key: statistics.fmean(t[key] for t in own)
                 for key in ("accuracy", "precision", "recall", "f1")}
        for key in ("fit_ms", "predict_ms"):
            entry[key] = statistics.median(t[key] for t in own)
        for key in ("TP", "TN", "FP", "FN"):
            entry[key] = sum(t[key] for t in own)
        summary[name] = entry
    return {"trials": trials, "summary": summary}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv):
    """Run preprocessing, the benchmark and a tree report from the terminal.

    Data structure: uses the structures above; returns an exit code (int).
    """
    path = argv[1] if len(argv) > 1 else os.path.join("data", "train.csv")
    if not os.path.isfile(path):
        print(f"ERROR: '{path}' was not found.\n"
              "Download train.csv from Kaggle's 'Loan Prediction Problem Dataset' "
              "and place it in the data/ folder.")
        return 1
    try:
        raw = load_csv(path)
        print(f"Loaded {len(raw)} rows from {path}")
        rows = preprocess(raw)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    balance = Counter(row[TARGET] for row in rows)
    print(f"Class balance: Y = {balance['Y']}, N = {balance['N']}")
    train, test = split(rows, SEEDS[0])
    print(f"Stratified split: {len(train)} train / {len(test)} test")

    results = benchmark(rows)

    header = (f"{'Algorithm':<15}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
              f"{'F1':>9}{'Train ms':>11}{'Predict ms':>12}")
    print("\nPer-trial results")
    print("Seed  " + header)
    for t in results["trials"]:
        print(f"{t['seed']:<6}{t['algorithm']:<15}{t['accuracy']:>10.4f}"
              f"{t['precision']:>11.4f}{t['recall']:>9.4f}{t['f1']:>9.4f}"
              f"{t['fit_ms']:>11.3f}{t['predict_ms']:>12.3f}")

    print("\nSummary of 5 trials (metrics = mean, times = median)")
    print(header)
    for name, s in results["summary"].items():
        print(f"{name:<15}{s['accuracy']:>10.4f}{s['precision']:>11.4f}"
              f"{s['recall']:>9.4f}{s['f1']:>9.4f}{s['fit_ms']:>11.3f}"
              f"{s['predict_ms']:>12.3f}")

    print("\nConfusion matrix counts (added over 5 trials)")
    for name, s in results["summary"].items():
        print(f"{name:<15}TP={s['TP']}  TN={s['TN']}  FP={s['FP']}  FN={s['FN']}")

    tree = DecisionTree().fit(rows)
    depth, nodes = tree.depth_and_nodes()
    print(f"\nDecision Tree trained on all {len(rows)} rows (used by the app)")
    print(f"Depth: {depth}   Nodes: {nodes}   Root feature: {tree.root.feature or 'none (single leaf)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
