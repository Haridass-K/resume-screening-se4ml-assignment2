from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)

API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

MODEL_METRICS_FILE = Path(
    "artifacts/production/model_quality_metrics.csv"
)

DATA_METRICS_FILE = Path(
    "artifacts/production/data_quality_metrics.csv"
)

LOG_FILE = Path("logs/application.log")


# ============================================================
# Helper Functions
# ============================================================


def check_api_health() -> bool:
    """Check whether the FastAPI backend is available."""
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=5,
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


def predict_resume(resume_text: str) -> dict:
    """Send resume text to the FastAPI prediction endpoint."""
    response = requests.post(
        PREDICT_URL,
        json={"resume_text": resume_text},
        timeout=30,
    )

    if response.status_code == 200:
        return response.json()

    try:
        error_message = response.json().get(
            "detail",
            "Prediction failed.",
        )

    except ValueError:
        error_message = "Prediction failed."

    raise RuntimeError(error_message)


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file safely."""
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def read_log_lines() -> list[str]:
    """Read the application log file safely."""
    if not LOG_FILE.exists():
        return []

    try:
        return LOG_FILE.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

    except OSError:
        return []


def filter_logs(
    log_lines: list[str],
    selected_level: str,
    search_text: str,
) -> list[str]:
    """Filter logs by level and search text."""
    filtered_lines = log_lines

    if selected_level != "ALL":
        filtered_lines = [
            line
            for line in filtered_lines
            if f"| {selected_level} |" in line
        ]

    if search_text.strip():
        search_value = search_text.strip().lower()

        filtered_lines = [
            line
            for line in filtered_lines
            if search_value in line.lower()
        ]

    return filtered_lines


def count_log_levels(log_lines: list[str]) -> dict[str, int]:
    """Count INFO, WARNING, and ERROR log entries."""
    return {
        "INFO": sum("| INFO |" in line for line in log_lines),
        "WARNING": sum(
            "| WARNING |" in line
            for line in log_lines
        ),
        "ERROR": sum(
            "| ERROR |" in line
            for line in log_lines
        ),
    }


def format_metric_value(value) -> str:
    """Format metric values for display."""
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return str(value)


def create_metrics_chart(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare metric data for a Streamlit chart."""
    normalized_columns = {
        column.lower().strip(): column
        for column in dataframe.columns
    }

    metric_column = normalized_columns.get("metric")
    value_column = normalized_columns.get("value")

    if metric_column and value_column:
        chart_data = dataframe[
            [metric_column, value_column]
        ].copy()

        chart_data[value_column] = pd.to_numeric(
            chart_data[value_column],
            errors="coerce",
        )

        chart_data = chart_data.dropna(
            subset=[value_column]
        )

        chart_data = chart_data.set_index(
            metric_column
        )

        return chart_data

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        return pd.DataFrame()

    if len(dataframe) == 1:
        chart_data = dataframe[
            numeric_columns
        ].transpose()

        chart_data.columns = ["Value"]

        return chart_data

    return dataframe[numeric_columns].copy()


# ============================================================
# Header
# ============================================================

st.title("AI Resume Screening System")

st.write(
    "Production-style resume classification application using "
    "Streamlit, FastAPI, machine learning, SQLite, security "
    "validation, testing, and application logging."
)

api_available = check_api_health()

if api_available:
    st.success("FastAPI backend is running.")

else:
    st.warning(
        "FastAPI backend is not available. Start the API using "
        "`uvicorn app.main:app --reload`."
    )


# ============================================================
# Tabs
# ============================================================

(
    prediction_tab,
    evaluation_tab,
    validation_tab,
    logs_tab,
    about_tab,
) = st.tabs(
    [
        "Resume Prediction",
        "Model Evaluation",
        "Project Validation",
        "Application Logs",
        "About",
    ]
)


# ============================================================
# Tab 1: Resume Prediction
# ============================================================

with prediction_tab:
    st.header("Resume Prediction")

    st.write(
        "Paste resume content below and submit it to the "
        "FastAPI prediction service."
    )

    resume_text = st.text_area(
        "Resume Text",
        height=320,
        placeholder=(
            "Paste the candidate's professional summary, skills, "
            "experience, education, and project details here."
        ),
    )

    uploaded_file = st.file_uploader(
        "Alternatively upload a text file",
        type=["txt"],
    )

    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode(
            "utf-8",
            errors="replace",
        )

        if uploaded_text.strip():
            resume_text = uploaded_text

            st.info(
                f"Loaded text from: {uploaded_file.name}"
            )

            st.text_area(
                "Uploaded Resume Content",
                value=uploaded_text,
                height=250,
                disabled=True,
            )

    if st.button(
        "Predict Resume Category",
        type="primary",
        use_container_width=True,
    ):
        if not resume_text.strip():
            st.warning(
                "Please enter resume text before prediction."
            )

        elif not api_available:
            st.error(
                "Cannot connect to the FastAPI backend. "
                "Start the API and try again."
            )

        else:
            try:
                with st.spinner("Analysing resume..."):
                    result = predict_resume(resume_text)

                predicted_category = result.get(
                    "predicted_category",
                    "Unknown",
                )

                confidence = result.get(
                    "confidence",
                    0,
                )

                st.success(
                    "Prediction completed successfully."
                )

                category_column, confidence_column = (
                    st.columns(2)
                )

                with category_column:
                    st.metric(
                        "Predicted Category",
                        predicted_category,
                    )

                with confidence_column:
                    st.metric(
                        "Confidence",
                        f"{float(confidence):.2f}%",
                    )

                top_predictions = result.get(
                    "top_predictions",
                    [],
                )

                if top_predictions:
                    st.subheader("Top Predictions")

                    predictions_df = pd.DataFrame(
                        top_predictions
                    )

                    st.dataframe(
                        predictions_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the FastAPI "
                    "prediction service."
                )

            except requests.Timeout:
                st.error(
                    "The prediction request timed out. "
                    "Please try again."
                )

            except requests.RequestException as error:
                st.error(
                    f"API request failed: {error}"
                )

            except RuntimeError as error:
                st.error(str(error))


# ============================================================
# Tab 2: Model Evaluation
# ============================================================

with evaluation_tab:
    st.header("Model Evaluation")

    # --------------------------------------------------------
    # Model Quality Metrics
    # --------------------------------------------------------

    st.subheader("Model Quality Metrics")

    model_metrics_df = load_csv(
        MODEL_METRICS_FILE
    )

    if model_metrics_df.empty:
        st.warning(
            "Model quality metrics file was not found or "
            "could not be read."
        )

    else:
        st.dataframe(
            model_metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        normalized_columns = {
            column.lower().strip(): column
            for column in model_metrics_df.columns
        }

        metric_column = normalized_columns.get("metric")
        value_column = normalized_columns.get("value")

        if metric_column and value_column:
            metric_rows = model_metrics_df[
                [metric_column, value_column]
            ].dropna()

            metric_columns = st.columns(
                min(len(metric_rows), 4)
            )

            for index, (_, row) in enumerate(
                metric_rows.head(4).iterrows()
            ):
                with metric_columns[index]:
                    st.metric(
                        str(row[metric_column])
                        .replace("_", " ")
                        .title(),
                        format_metric_value(
                            row[value_column]
                        ),
                    )

        elif len(model_metrics_df) > 0:
            first_row = model_metrics_df.iloc[0]

            visible_columns = list(
                model_metrics_df.columns[:4]
            )

            metric_columns = st.columns(
                len(visible_columns)
            )

            for index, column_name in enumerate(
                visible_columns
            ):
                with metric_columns[index]:
                    st.metric(
                        column_name.replace(
                            "_",
                            " ",
                        ).title(),
                        format_metric_value(
                            first_row[column_name]
                        ),
                    )

        model_chart_data = create_metrics_chart(
            model_metrics_df
        )

        if not model_chart_data.empty:
            st.subheader(
                "Model Metrics Visualization"
            )

            st.bar_chart(model_chart_data)

    st.divider()

    # --------------------------------------------------------
    # Data Quality Metrics
    # --------------------------------------------------------

    st.subheader("Data Quality Metrics")

    data_metrics_df = load_csv(
        DATA_METRICS_FILE
    )

    if data_metrics_df.empty:
        st.warning(
            "Data quality metrics file was not found or "
            "could not be read."
        )

    else:
        st.dataframe(
            data_metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        data_chart_data = create_metrics_chart(
            data_metrics_df
        )

        if not data_chart_data.empty:
            st.subheader(
                "Data Quality Visualization"
            )

            st.bar_chart(data_chart_data)

        else:
            st.info(
                "No numeric data is available for "
                "the data quality chart."
            )


# ============================================================
# Tab 3: Project Validation
# ============================================================

with validation_tab:
    st.header("Project Validation")

    st.write(
        "The following software engineering and machine-learning "
        "validation activities were completed."
    )

    validation_items = [
        ("Automated Tests", "13 tests passed"),
        ("Code Formatting", "Black passed"),
        ("Import Sorting", "isort passed"),
        ("Static Code Analysis", "Flake8 passed"),
        ("Unit Testing", "Implemented"),
        ("Integration Testing", "Implemented"),
        ("API Testing", "Implemented"),
        ("Data Validation Testing", "Implemented"),
        ("ML Training Testing", "Implemented"),
        ("ML Inference Testing", "Implemented"),
        ("Security Validation", "Implemented"),
        ("Shadow Deployment", "Implemented"),
        ("Canary Release", "Implemented"),
        ("A/B Testing", "Implemented"),
    ]

    validation_df = pd.DataFrame(
        validation_items,
        columns=[
            "Validation Area",
            "Status",
        ],
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "Testing, code-quality validation, security checks, "
        "and production experimentation have been completed."
    )


# ============================================================
# Tab 4: Application Logs
# ============================================================

with logs_tab:
    st.header("Application Logs")

    st.write(
        "This page displays application activity recorded by "
        "Python's logging module."
    )

    log_lines = read_log_lines()

    if not log_lines:
        st.warning(
            f"No log entries were found at: `{LOG_FILE}`"
        )

    else:
        level_counts = count_log_levels(
            log_lines
        )

        info_column, warning_column, error_column = (
            st.columns(3)
        )

        with info_column:
            st.metric(
                "INFO Logs",
                level_counts["INFO"],
            )

        with warning_column:
            st.metric(
                "WARNING Logs",
                level_counts["WARNING"],
            )

        with error_column:
            st.metric(
                "ERROR Logs",
                level_counts["ERROR"],
            )

        st.divider()

        filter_column, search_column, lines_column = (
            st.columns([1, 2, 1])
        )

        with filter_column:
            selected_level = st.selectbox(
                "Log Level",
                [
                    "ALL",
                    "INFO",
                    "WARNING",
                    "ERROR",
                ],
            )

        with search_column:
            search_text = st.text_input(
                "Search Logs",
                placeholder=(
                    "Search by module, message, "
                    "date, or keyword"
                ),
            )

        with lines_column:
            number_of_lines = st.selectbox(
                "Latest Entries",
                [
                    25,
                    50,
                    100,
                    200,
                    500,
                ],
                index=2,
            )

        filtered_log_lines = filter_logs(
            log_lines,
            selected_level,
            search_text,
        )

        displayed_lines = filtered_log_lines[
            -number_of_lines:
        ]

        st.caption(
            f"Showing {len(displayed_lines)} of "
            f"{len(filtered_log_lines)} matching entries."
        )

        if displayed_lines:
            st.code(
                "\n".join(displayed_lines),
                language="text",
            )

        else:
            st.info(
                "No log entries match the selected filters."
            )

        refresh_column, download_column = (
            st.columns(2)
        )

        with refresh_column:
            if st.button(
                "Refresh Logs",
                use_container_width=True,
            ):
                st.rerun()

        with download_column:
            st.download_button(
                label="Download Log File",
                data="\n".join(log_lines),
                file_name="application.log",
                mime="text/plain",
                use_container_width=True,
            )

        st.info(
            "INFO represents normal application activity, "
            "WARNING represents expected validation problems, "
            "and ERROR represents unexpected application failures."
        )


# ============================================================
# Tab 5: About
# ============================================================

with about_tab:
    st.header("About the Project")

    st.subheader("System Architecture")

    st.code(
        """
Streamlit Frontend
        |
        v
FastAPI REST API
        |
        v
Security and Input Validation
        |
        v
Random Forest Classification Model
        |
        v
SQLite Prediction History Database
        """,
        language="text",
    )

    st.subheader("Production Features")

    st.markdown(
        """
- Modular production-style code structure
- FastAPI request and response schemas
- Input validation and security checks
- INFO, WARNING, and ERROR logging
- Centralized exception handling
- SQLite prediction history
- Unit, integration, API, data, and ML tests
- Model-quality and data-quality metrics
- Shadow, canary, and A/B production experiments
- Streamlit user interface
        """
    )

    st.subheader("Technology Stack")

    technology_df = pd.DataFrame(
        [
            ["Frontend", "Streamlit"],
            ["Backend", "FastAPI"],
            ["Machine Learning", "Scikit-learn"],
            ["Model", "Random Forest"],
            ["Database", "SQLite"],
            ["Testing", "PyTest"],
            [
                "Code Quality",
                "Black, isort, Flake8",
            ],
            [
                "Logging",
                "Python logging module",
            ],
        ],
        columns=[
            "Component",
            "Technology",
        ],
    )

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True,
    )
st.divider()

st.subheader("Group Information")

st.write("**Course:** Software Engineering for Machine Learning")

st.write("**Group Number:** 25")

group_df = pd.DataFrame(
    [
        [
            "2024AC05325",
            "Haridass K",
            "2024ac05325@wilp.bits-pilani.ac.in",
        ],
        [
            "2024AC05104",
            "Sathish T",
            "2024ac05104@wilp.bits-pilani.ac.in",
        ],
        [
            "2024AC05651",
            "Tejaal M",
            "2024ac05651@wilp.bits-pilani.ac.in",
        ],
        [
            "2024AC05728",
            "Sanjayan S",
            "2024ac05728@wilp.bits-pilani.ac.in",
        ],
    ],
    columns=[
        "BITS ID",
        "Name",
        "Email",
    ],
)

st.dataframe(
    group_df,
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Disclaimer")

st.info(
    "This application has been developed solely for academic purposes "
    "as part of the Software Engineering for Machine Learning course "
    "at BITS Pilani Work Integrated Learning Programme (WILP). "
    "It is intended for educational demonstration and evaluation only "
    "and should not be used as a production recruitment or resume "
    "screening system."
)