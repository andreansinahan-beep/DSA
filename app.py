"""
Loan Approval Prediction System - Streamlit web app.

Run with:  streamlit run app.py
All classification logic lives in algorithms.py; this file only shows it.
"""

import os

import matplotlib
matplotlib.use("Agg")  # draw charts without opening a desktop window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import algorithms as algo

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1"]
FIELD_LABELS = {
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Co-applicant Income",
    "Credit_History": "Credit History",
    "Loan_Amount_Term": "Loan Amount Term",
    "LoanAmount": "Loan Amount",
    "Loan_ID": "Loan ID",
    "Loan_Status": "Loan Status",
    "Property_Area": "Property Area",
    "Self_Employed": "Self-Employed",
    "TotalIncome": "Total Income",
}


def display_field_name(field):
    """Return a readable display label without changing the source field name."""
    return FIELD_LABELS.get(field, field.replace("_", " "))


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path, modified):
    """Load and clean the CSV. Returns (raw_rows, clean_rows).

    Data structure: two LISTS of DICTS. 'modified' is the file's timestamp, so
    the cache refreshes when the file is replaced.
    """
    raw = algo.load_csv(path)
    return raw, algo.preprocess(raw)


@st.cache_resource
def get_models(path, modified):
    """Train all three models once and reuse them.

    Data structure: DICT of name -> model (list + heap, binary tree, hash maps).
    """
    _, clean = load_data(path, modified)
    return algo.train_models(clean)


@st.cache_data
def load_sample_data(path, modified):
    """Load sample rows for the presentation demo buttons."""
    raw = algo.load_csv(path)
    return raw, algo.preprocess(raw)


def get_data_or_stop():
    """Return (raw, clean, modified), or show a message and stop the page.

    Data structure: TUPLE of the two row lists and the file timestamp.
    """
    if not os.path.isfile(DATA_PATH):
        st.error("data/train.csv was not found.")
        st.info("Download **train.csv** from Kaggle's **Loan Prediction Problem "
                "Dataset**, put it in the `data/` folder, then reload this page.")
        st.stop()
    modified = os.path.getmtime(DATA_PATH)
    try:
        raw, clean = load_data(DATA_PATH, modified)
    except ValueError as error:
        st.error(f"Could not read data/train.csv: {error}")
        st.stop()
    return raw, clean, modified


def find_demo_record(credit_history, expected_label, models):
    """Find a sample applicant whose model predictions unanimously match a label."""
    if not os.path.isfile(SAMPLE_PATH):
        return None
    _, clean = load_sample_data(SAMPLE_PATH, os.path.getmtime(SAMPLE_PATH))
    allowed = {
        "Gender": {"Male", "Female"},
        "Married": {"Yes", "No"},
        "Dependents": {"0", "1", "2", "3+"},
        "Education": {"Graduate", "Not Graduate"},
        "Self_Employed": {"No", "Yes"},
        "Property_Area": {"Urban", "Semiurban", "Rural"},
    }
    valid_terms = {120, 180, 240, 300, 360}
    for record in clean:
        if record["Credit_History"] != credit_history:
            continue
        if any(record[field] not in values for field, values in allowed.items()):
            continue
        if int(record["Loan_Amount_Term"]) not in valid_terms:
            continue
        if any(model.predict_one(record) != expected_label for model in models.values()):
            continue
        return record
    return None


def fill_demo_record(record):
    """Copy a cleaned sample applicant into the prediction form's stable keys."""
    for field in ("Gender", "Married", "Dependents", "Education",
                  "Self_Employed", "Credit_History", "Property_Area"):
        st.session_state[field] = record[field]
    st.session_state["ApplicantIncome"] = float(record["ApplicantIncome"])
    st.session_state["CoapplicantIncome"] = float(record["CoapplicantIncome"])
    st.session_state["LoanAmount"] = float(record["LoanAmount"])
    st.session_state["Loan_Amount_Term"] = int(record["Loan_Amount_Term"])


def status_word(label):
    """Turn 'Y'/'N' into a readable word. Data structure: none."""
    return "APPROVED" if label == "Y" else "REJECTED"


def result_card(title, label):
    """Draw a green (approved) or red (rejected) card.

    Data structure: none (builds an HTML string).
    """
    approved = label == "Y"
    accent = "#047857" if approved else "#b42318"
    surface = "#ecfdf3" if approved else "#fff1f0"
    icon = "&#10003;" if approved else "&#10005;"
    st.markdown(
        f"<div class='result-card {'approved' if approved else 'rejected'}'>"
        f"<div class='result-card-icon' style='color:{accent};background:{surface}'>{icon}</div>"
        f"<div class='result-card-content'><div class='result-card-title'>{title}</div>"
        f"<div class='result-card-status' style='color:{accent}'>{status_word(label)}</div></div>"
        f"</div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1: Prediction
# ---------------------------------------------------------------------------
def validate(record):
    """Return a LIST of error messages for the form (empty list = valid).

    Data structure: LIST, so that all problems are shown at once.
    """
    errors = []
    for field in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"):
        if record[field] is None:
            errors.append(f"{display_field_name(field)} is required.")
    for field in ("ApplicantIncome", "CoapplicantIncome"):
        if record[field] is not None and record[field] < 0:
            errors.append(f"{display_field_name(field)} cannot be negative.")
    for field in ("LoanAmount", "Loan_Amount_Term"):
        if record[field] is not None and record[field] <= 0:
            errors.append(f"{display_field_name(field)} must be greater than 0.")
    return errors


def prediction_page():
    """Show sample-backed form controls and side-by-side model predictions.

    Data structure: the applicant is a DICT with the same keys as a cleaned row.
    """
    st.title("Loan Approval Prediction")
    st.caption("Enter applicant information to compare three in-house prediction models.")
    _, _, modified = get_data_or_stop()

    st.subheader("Try a sample applicant")
    if os.path.isfile(SAMPLE_PATH):
        demo_left, demo_right = st.columns(2)
        for column, button_label, credit, expected in (
            (demo_left, "Load approved demo (Credit History = 1)", 1.0, "Y"),
            (demo_right, "Load rejected demo (Credit History = 0)", 0.0, "N"),
        ):
            with column:
                if st.button(button_label, use_container_width=True):
                    try:
                        models = get_models(DATA_PATH, modified)
                        record = find_demo_record(credit, expected, models)
                    except (OSError, ValueError) as error:
                        st.error(f"Could not load a sample applicant: {error}")
                        record = None
                    if record is None:
                        st.warning(
                            "No matching unanimous demo row was found in sample.csv "
                            "for the current dataset and models."
                        )
                    else:
                        fill_demo_record(record)
                        st.session_state["demo_notice"] = (
                            f"Loaded sample applicant {record.get('Loan_ID', '')}."
                        )
                        st.rerun()
        st.caption(
            "Each demo button finds a row in sample.csv that all three current "
            "models predict as requested."
        )
    else:
        st.caption("Demo presets are unavailable because data/sample.csv was not found.")

    demo_notice = st.session_state.pop("demo_notice", None)
    if demo_notice:
        st.success(demo_notice)

    with st.form("applicant"):
        st.markdown("#### Credit")
        credit_col, area_col = st.columns(2)
        with credit_col:
            credit = st.selectbox(
                "Credit History", [1.0, 0.0],
                format_func=lambda value: "1 = good" if value == 1.0 else "0 = bad",
                key="Credit_History",
            )
        with area_col:
            area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"], key="Property_Area"
            )

        st.markdown("#### Applicant")
        applicant_columns = st.columns(3)
        with applicant_columns[0]:
            gender = st.selectbox("Gender", ["Male", "Female"], key="Gender")
            education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"], key="Education"
            )
        with applicant_columns[1]:
            married = st.selectbox("Married", ["Yes", "No"], key="Married")
            self_employed = st.selectbox(
                "Self-Employed", ["No", "Yes"], key="Self_Employed"
            )
        with applicant_columns[2]:
            dependents = st.selectbox(
                "Dependents", ["0", "1", "2", "3+"], key="Dependents"
            )

        st.markdown("#### Finances")
        finance_left, finance_right = st.columns(2)
        with finance_left:
            income = st.number_input(
                "Applicant Income (monthly)", value=None, step=500.0,
                key="ApplicantIncome",
            )
            co_income = st.number_input(
                "Co-applicant Income (monthly)", value=None, step=500.0,
                key="CoapplicantIncome",
            )
        with finance_right:
            amount = st.number_input(
                "Loan Amount (thousands)", value=None, step=10.0, key="LoanAmount"
            )
            term = st.selectbox(
                "Loan Amount Term (months)", [120, 180, 240, 300, 360],
                index=4, key="Loan_Amount_Term",
            )
        st.caption(
            "Income uses the dataset's monthly-income convention (the CSV has no "
            "unit metadata). Enter 0 for Co-applicant Income if there is no co-applicant."
        )
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    applicant = {
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": income, "CoapplicantIncome": co_income,
        "LoanAmount": amount, "Loan_Amount_Term": term,
        "Credit_History": credit, "Property_Area": area,
    }
    errors = validate(applicant)
    if errors:
        for message in errors:
            st.error(message)
        return
    applicant["TotalIncome"] = income + co_income

    with st.spinner("Training the models..."):
        models = get_models(DATA_PATH, modified)
    predictions = {name: model.predict_one(applicant) for name, model in models.items()}

    st.subheader("Model predictions")
    for column, (name, label) in zip(st.columns(3), predictions.items()):
        with column:
            result_card(name, label)

    votes = list(predictions.values())
    final = algo.majority_vote(votes)
    if len(set(votes)) == 1:
        st.subheader("Overall verdict")
        result_card("Unanimous model decision", final)
        st.caption("3 of 3 models agree.")
    else:
        st.warning(
            f"The models disagree. The existing majority vote is "
            f"{status_word(final)} ({votes.count(final)} of 3)."
        )
        st.markdown("**Individual model results**")
        for name, label in predictions.items():
            st.write(f"- **{name}:** {status_word(label)}")

    st.subheader("Decision Tree rule path")
    rule_path = models["Decision Tree"].rule_path(applicant).replace("_", " ")
    st.code(rule_path, language=None, wrap_lines=True)

    st.subheader("7 nearest neighbors (KNN)")
    table = []
    for rank, item in enumerate(models["KNN"].neighbors(applicant), start=1):
        record = item["record"]
        table.append({
            "Rank": rank, "Loan ID": record.get("Loan_ID", ""),
            "Distance": round(item["distance"], 4), "Loan Status": item["label"],
            "Credit History": record["Credit_History"],
            "Total Income": record["TotalIncome"], "Loan Amount": record["LoanAmount"],
            "Education": record["Education"], "Property Area": record["Property_Area"],
        })
    st.dataframe(pd.DataFrame(table), height=380, use_container_width=True, hide_index=True)
    st.divider()
    st.caption(
        "School project — not a lending tool. West Visayas State University, CC 204."
    )


# ---------------------------------------------------------------------------
# Page 2: Dataset & Preprocessing
# ---------------------------------------------------------------------------
def dataset_page():
    """Show raw data, missing values, class balance and before/after rows.

    Data structure: the row LISTS are turned into pandas tables for display.
    """
    st.title("Dataset & Preprocessing")
    raw, clean, _ = get_data_or_stop()

    st.markdown("**1. Raw**  →  **2. Cleaned**  →  **3. Encoded**")
    st.caption(
        "Text categories and numeric values are represented in the formats used "
        "by the three models."
    )

    st.subheader("Raw rows (as loaded)")
    st.write(f"{len(raw)} rows, {len(raw[0])} columns")
    st.dataframe(
        pd.DataFrame(raw).rename(columns=display_field_name),
        height=380,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing values per column")
    labelled = [row for row in raw if not algo.is_missing(row[algo.TARGET])]
    before = algo.count_missing(labelled)
    after = algo.count_missing(clean)
    feature_columns = algo.TEXT_COLUMNS + algo.NUMBER_COLUMNS
    affected = [column for column in feature_columns if before[column] > 0]
    filled_cells = sum(before[column] - after.get(column, 0) for column in affected)
    st.write(f"{len(affected)} columns affected, {filled_cells} cells filled.")
    if affected:
        st.dataframe(
            pd.DataFrame([{
                "Column": display_field_name(column),
                "Before": before[column],
                "After": after.get(column, 0),
            } for column in affected]),
            height=380,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No missing feature values were found.")
    st.caption("Text columns are filled with the mode, number columns with the median. "
               "Total Income is a new column.")

    st.subheader("Class balance")
    labels = [row[algo.TARGET] for row in clean]
    counts = [labels.count("Y"), labels.count("N")]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(["Y (Approved)", "N (Rejected)"], counts, color=["#1b7f3b", "#b3261e"])
    ax.bar_label(
        bars,
        labels=[f"{count} ({count / total:.0%})" for count in counts],
        padding=4,
    )
    ax.set_ylabel("Number of applicants")
    ax.margins(y=0.12)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Effect of cleaning (before vs after)")
    # preprocess() drops unlabelled rows, so filter the raw rows the same way
    changed = [i for i, r in enumerate(labelled)
               if any(algo.is_missing(v) for v in r.values())][:5]
    if not changed:
        st.info("No row had a missing value; showing the first 5 rows.")
        changed = list(range(min(5, len(labelled))))
    before_after = st.columns(2)
    with before_after[0]:
        st.markdown("**Before**")
        st.dataframe(
            pd.DataFrame([labelled[i] for i in changed]).rename(
                columns=display_field_name
            ),
            height=380,
            use_container_width=True,
            hide_index=True,
        )
    with before_after[1]:
        st.markdown("**After**")
        st.dataframe(
            pd.DataFrame([clean[i] for i in changed]).rename(
                columns=display_field_name
            ),
            height=380,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page 3: Algorithm Benchmark
# ---------------------------------------------------------------------------
def confusion_figure(name, s):
    """Heatmap of one confusion matrix.

    Data structure: 2x2 LIST of lists (2D array): rows = actual, cols = predicted.
    """
    matrix = np.array([[s["TP"], s["FN"]], [s["FP"], s["TN"]]])
    tags = [["TP", "FN"], ["FP", "TN"]]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{tags[i][j]}\n{matrix[i, j]}", ha="center",
                    va="center", color=color, fontsize=12)
    ax.set_xticks([0, 1], ["Y", "N"])
    ax.set_yticks([0, 1], ["Y", "N"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    fig.tight_layout()
    return fig


def metrics_figure(summary):
    """Grouped bar chart of the four metrics.

    Data structure: loops over the summary DICT, one bar series per algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(METRIC_KEYS))
    width = 0.26
    for i, (name, s) in enumerate(summary.items()):
        bars = ax.bar(x + (i - 1) * width, [s[k] for k in METRIC_KEYS], width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, METRIC_NAMES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (mean of 5 trials)")
    ax.legend(ncols=3, loc="upper center")
    fig.tight_layout()
    return fig


def time_figure(summary):
    """Bar chart of training vs prediction time (log scale, because the times
    differ by a large factor).

    Data structure: two LISTS of times in the same algorithm order.
    """
    names = list(summary)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, key, label in ((-0.2, "fit_ms", "Training"),
                               (0.2, "predict_ms", "Prediction (whole test set)")):
        bars = ax.bar(x + offset, [summary[n][key] for n in names], 0.4, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_yscale("log")
    ax.margins(y=0.25)
    ax.set_ylabel("Milliseconds (median, log scale)")
    ax.legend()
    fig.tight_layout()
    return fig


def benchmark_page():
    """Run benchmark() and show the tables, charts and CSV export.

    Data structure: the result DICT is kept in st.session_state so it survives
    page reruns.
    """
    st.title("Algorithm Benchmark")
    _, clean, modified = get_data_or_stop()
    st.write("5 trials (seeds 43-47), each with a stratified 80/20 split.")

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running 5 trials..."):
            st.session_state["benchmark"] = (modified, algo.benchmark(clean))

    stored = st.session_state.get("benchmark")
    if not stored or stored[0] != modified:
        st.subheader("Export")
        st.caption("Run the benchmark to enable export")
        st.button(
            "Save CSV files into results/",
            disabled=True,
            use_container_width=True,
        )
        st.download_button(
            "Download summary CSV",
            data="",
            file_name="benchmark_summary.csv",
            mime="text/csv",
            disabled=True,
            use_container_width=True,
        )
        st.download_button(
            "Download per-trial CSV",
            data="",
            file_name="benchmark_trials.csv",
            mime="text/csv",
            disabled=True,
            use_container_width=True,
        )
        return
    results = stored[1]
    summary = results["summary"]

    st.subheader("Summary")
    table = pd.DataFrame([{
        "Algorithm": name,
        **{label: s[key] for key, label in zip(METRIC_KEYS, METRIC_NAMES)},
        "Train (ms)": s["fit_ms"], "Predict (ms)": s["predict_ms"],
    } for name, s in summary.items()])
    st.dataframe(table.style.format(
        {**{m: "{:.4f}" for m in METRIC_NAMES},
         "Train (ms)": "{:.3f}", "Predict (ms)": "{:.3f}"}),
        height=380,
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Metrics are the mean and times are the median of 5 trials. "
               "Prediction time covers the whole test set.")
    accuracy_by_model = {
        name: values["accuracy"] for name, values in summary.items()
    }
    accuracy_spread = max(accuracy_by_model.values()) - min(accuracy_by_model.values())
    if accuracy_spread <= 0.05:
        comparison = (
            f"Accuracy is closely grouped across models "
            f"({min(accuracy_by_model.values()):.1%}–"
            f"{max(accuracy_by_model.values()):.1%}; "
            f"{accuracy_spread:.1%} spread)."
        )
    else:
        comparison = (
            f"Accuracy varies across models "
            f"({min(accuracy_by_model.values()):.1%}–"
            f"{max(accuracy_by_model.values()):.1%}; "
            f"{accuracy_spread:.1%} spread)."
        )
    tree_accuracy = accuracy_by_model["Decision Tree"]
    best_accuracy = max(accuracy_by_model.values())
    if tree_accuracy < best_accuracy:
        tree_note = (
            f"The Decision Tree scores {tree_accuracy:.1%}, "
            f"{best_accuracy - tree_accuracy:.1%} below the best accuracy; "
            "its rule path offers a more interpretable decision trail."
        )
    else:
        tree_position = (
            "highest" if tree_accuracy > min(accuracy_by_model.values())
            else "tied for highest"
        )
        tree_note = (
            f"The Decision Tree has the {tree_position} accuracy at "
            f"{tree_accuracy:.1%}, with its readable rule path."
        )
    st.caption(f"{comparison} {tree_note}")

    st.subheader("Confusion matrices (counts added over 5 trials)")
    for column, (name, s) in zip(st.columns(3), summary.items()):
        with column:
            fig = confusion_figure(name, s)
            st.pyplot(fig)
            plt.close(fig)

    for heading, builder in (("Metrics comparison", metrics_figure),
                             ("Training vs prediction time", time_figure)):
        st.subheader(heading)
        fig = builder(summary)
        st.pyplot(fig)
        plt.close(fig)

    trials = pd.DataFrame(results["trials"])
    with st.expander("Per-trial output (seeds 43-47)"):
        st.dataframe(
            trials, height=380, use_container_width=True, hide_index=True
        )

    st.subheader("Export")
    summary_csv = table.to_csv(index=False)
    trials_csv = trials.to_csv(index=False)
    if st.button("Save CSV files into results/", use_container_width=True):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for filename, text in (("benchmark_summary.csv", summary_csv),
                               ("benchmark_trials.csv", trials_csv)):
            with open(os.path.join(RESULTS_DIR, filename), "w",
                      newline="", encoding="utf-8") as file:
                file.write(text)
        st.success("Saved benchmark summary and per-trial CSV files in results/.")
    st.download_button("Download summary CSV", summary_csv,
                       "benchmark_summary.csv", "text/csv",
                       use_container_width=True)
    st.download_button("Download per-trial CSV", trials_csv,
                       "benchmark_trials.csv", "text/csv",
                       use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: About
# ---------------------------------------------------------------------------
def about_page():
    """Static project information. Data structure: Markdown text only."""
    st.title("About")
    st.subheader("Problem statement")
    st.write("Checking loan applications by hand takes time and can be inconsistent. "
             "Given an applicant's details, this system classifies the application as "
             "Approved (Y) or Rejected (N) and compares three classification algorithms "
             "implemented from scratch. It is a school project and not a real lending tool.")

    st.subheader("Algorithms and data structures")
    st.markdown("""
| Algorithm | How it decides | Data structure |
|---|---|---|
| K-Nearest Neighbors | Majority label of the 7 most similar applicants | List + min-heap (`heapq`) |
| Decision Tree | Follows yes/no questions chosen by Gini impurity | Binary tree of `Node` objects |
| Naive Bayes | Most probable class from frequency counts | Hash maps (`dict`, `Counter`) |
""")

    st.subheader("Complexity")
    st.markdown("""
| Algorithm | Training | Prediction (one record) | Space |
|---|---|---|---|
| KNN | O(1) | O(n·d + n log k) | O(n·d) |
| Decision Tree | O(d·n²) worst | O(depth) | O(nodes) |
| Naive Bayes | O(n·d) | O(c·d) | O(c·d·v) |

n = training rows, d = features, k = neighbors, c = classes, v = distinct values per feature
""")
    st.caption("KNN's O(1) training is the textbook case of only storing a reference to "
               "the data. This program also normalizes the data in fit(), which is O(n·d). "
               "The Decision Tree bound assumes a fixed maximum depth.")

    st.subheader("Group members")
    st.markdown("\n".join(f"- [Member {i}]" for i in range(1, 6)))


def apply_global_styles():
    """Apply small layout refinements on top of the configured Streamlit theme."""
    st.markdown("""
    <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 8px;
        }

        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #dce3ec;
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 8px 24px rgba(19, 43, 74, 0.06);
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] input {
            background-color: #ffffff !important;
            color: #1b2a41 !important;
            -webkit-text-fill-color: #1b2a41 !important;
        }

        [data-testid="stButton"] > button,
        [data-testid="stDownloadButton"] > button {
            border-radius: 8px;
            min-height: 42px;
            font-weight: 650;
        }

        [data-testid="stButton"] > button[kind="primary"] {
            background: #176b87;
            border-color: #176b87;
            color: #ffffff;
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #dce3ec;
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
        }

        .result-card {
            min-height: 116px;
            display: flex;
            align-items: center;
            gap: 14px;
            background: #ffffff;
            border: 1px solid #dce3ec;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(19, 43, 74, 0.055);
        }

        .result-card.approved { border-left: 5px solid #047857; }
        .result-card.rejected { border-left: 5px solid #b42318; }
        .result-card-icon {
            width: 44px;
            height: 44px;
            flex: 0 0 44px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            font-size: 20px;
            font-weight: 800;
        }
        .result-card-content { min-width: 0; }
        .result-card-title {
            color: #52647a;
            font-size: 0.86rem;
            font-weight: 650;
            margin-bottom: 4px;
        }
        .result-card-status {
            font-size: 1.35rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: 0.02em;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Set up the sidebar and open the chosen page.

    Data structure: DICT of page name -> function (a dispatch table).
    """
    apply_global_styles()
    pages = {
        "Prediction": prediction_page,
        "Dataset & Preprocessing": dataset_page,
        "Algorithm Benchmark": benchmark_page,
        "About": about_page,
    }
    st.sidebar.markdown(
        "<div style='font-size:1.15rem;font-weight:800;letter-spacing:.01em;"
        "margin:.25rem 0 1.25rem'>Loan Approval System</div>",
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio("Page", list(pages))
    pages[choice]()


if __name__ == "__main__":
    main()
