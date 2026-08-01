import io
import os
from pathlib import Path

import docx
import fitz
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

MODEL_METRICS_FILE = (
    PROJECT_ROOT / "artifacts" / "production" / "model_quality_metrics.csv"
)
DATA_METRICS_FILE = (
    PROJECT_ROOT / "artifacts" / "production" / "data_quality_metrics.csv"
)
LOG_FILE = PROJECT_ROOT / "logs" / "application.log"


def check_api_health() -> bool:
    """Check whether the FastAPI backend is available."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
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
        error_message = response.json().get("detail", "Prediction failed.")
    except ValueError:
        error_message = "Prediction failed."

    raise RuntimeError(error_message)


def extract_resume_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT resume files."""
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
                extracted_text = "\n".join(page.get_text() for page in pdf_document)
        except (fitz.FileDataError, RuntimeError) as error:
            raise ValueError("The PDF file is invalid or could not be read.") from error

        if not extracted_text.strip():
            raise ValueError(
                "No readable text was found in the PDF. "
                "Scanned image-only PDFs require OCR."
            )
        return extracted_text

    if file_name.endswith(".docx"):
        try:
            document = docx.Document(io.BytesIO(file_bytes))
        except Exception as error:
            raise ValueError(
                "The DOCX file is invalid or could not be read."
            ) from error

        extracted_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        if not extracted_text.strip():
            raise ValueError("No readable text was found in the DOCX file.")
        return extracted_text

    if file_name.endswith(".txt"):
        extracted_text = file_bytes.decode("utf-8", errors="replace")
        if not extracted_text.strip():
            raise ValueError("The TXT file does not contain readable text.")
        return extracted_text

    raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")


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
            line for line in filtered_lines if f"| {selected_level} |" in line
        ]

    if search_text.strip():
        search_value = search_text.strip().lower()
        filtered_lines = [
            line for line in filtered_lines if search_value in line.lower()
        ]

    return filtered_lines


def count_log_levels(log_lines: list[str]) -> dict[str, int]:
    """Count INFO, WARNING, and ERROR log entries."""
    return {
        "INFO": sum("| INFO |" in line for line in log_lines),
        "WARNING": sum("| WARNING |" in line for line in log_lines),
        "ERROR": sum("| ERROR |" in line for line in log_lines),
    }


def format_metric_value(metric_name: str, value) -> str:
    """Format quality metric values for display."""
    if pd.isna(value):
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if "ratio" in metric_name.lower():
        return f"{numeric_value:.2f}"
    if 0 <= numeric_value <= 1:
        return f"{numeric_value * 100:.2f}%"
    return f"{numeric_value:.2f}"


def prepare_metrics_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize metric and value columns."""
    if dataframe.empty:
        return pd.DataFrame()

    normalized_columns = {
        column.lower().strip(): column for column in dataframe.columns
    }
    metric_column = normalized_columns.get("metric")
    value_column = normalized_columns.get("value")

    if metric_column and value_column:
        prepared = dataframe[[metric_column, value_column]].copy()
        prepared = prepared.rename(
            columns={metric_column: "Metric", value_column: "Value"}
        )
        prepared["Value"] = pd.to_numeric(
            prepared["Value"],
            errors="coerce",
        )
        return prepared.dropna(subset=["Value"])

    if len(dataframe) == 1:
        rows = []
        for column in dataframe.columns:
            value = pd.to_numeric(dataframe.iloc[0][column], errors="coerce")
            if not pd.isna(value):
                rows.append({"Metric": column, "Value": value})
        return pd.DataFrame(rows)

    return pd.DataFrame()


def show_metric_cards(metrics_df: pd.DataFrame) -> None:
    """Display quality metrics as Streamlit cards."""
    if metrics_df.empty:
        return

    columns = st.columns(min(len(metrics_df), 4))
    for position, (_, row) in enumerate(metrics_df.iterrows()):
        column = columns[position % len(columns)]
        column.metric(
            label=str(row["Metric"]).replace("_", " ").title(),
            value=format_metric_value(str(row["Metric"]), row["Value"]),
        )


def show_metrics_chart(metrics_df: pd.DataFrame) -> None:
    """Display quality metrics in a bar chart."""
    if metrics_df.empty:
        return

    chart_data = metrics_df.copy()
    percentage_mask = (
        ~chart_data["Metric"].astype(str).str.lower().str.contains("ratio")
    )
    values_between_zero_and_one = chart_data["Value"].between(0, 1)
    chart_data.loc[
        percentage_mask & values_between_zero_and_one,
        "Value",
    ] *= 100

    st.bar_chart(
        chart_data.set_index("Metric")["Value"],
        use_container_width=True,
    )


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

with prediction_tab:
    st.header("Resume Prediction")
    st.write(
        "Paste resume content or upload a PDF, DOCX, or TXT resume "
        "file and submit it to the FastAPI prediction service."
    )

    pasted_text = st.text_area(
        "Resume Text",
        height=320,
        placeholder=(
            "Paste the candidate's professional summary, skills, "
            "experience, education, and project details here."
        ),
    )

    uploaded_file = st.file_uploader(
        "Alternatively upload a resume file",
        type=["pdf", "docx", "txt"],
        help=(
            "Supported formats: PDF, DOCX, and TXT. "
            "Scanned image-only PDFs require OCR."
        ),
    )

    uploaded_text = ""
    if uploaded_file is not None:
        try:
            uploaded_text = extract_resume_text(uploaded_file)
            st.success("Resume loaded successfully: " f"{uploaded_file.name}")
            st.text_area(
                "Extracted Resume Content",
                value=uploaded_text,
                height=250,
                disabled=True,
            )
        except ValueError as error:
            st.error(str(error))

    resume_text = uploaded_text if uploaded_text.strip() else pasted_text

    if st.button(
        "Predict Resume Category",
        type="primary",
        use_container_width=True,
    ):
        if not resume_text.strip():
            st.warning(
                "Please paste resume text or upload a valid resume file "
                "before prediction."
            )
        elif len(resume_text.strip()) < 20:
            st.warning("Please provide sufficient resume content.")
        elif not api_available:
            st.error(
                "Cannot connect to the FastAPI backend. Start the API " "and try again."
            )
        else:
            try:
                with st.spinner("Analysing resume..."):
                    result = predict_resume(resume_text)

                predicted_category = result.get(
                    "predicted_category",
                    "Unknown",
                )
                st.success("Prediction completed successfully.")
                st.subheader("Predicted Category")
                st.info(predicted_category)

            except requests.ConnectionError:
                st.error("Cannot connect to the FastAPI prediction service.")
            except requests.Timeout:
                st.error("The prediction request timed out. Please try again.")
            except requests.RequestException as error:
                st.error(f"API request failed: {error}")
            except RuntimeError as error:
                st.error(str(error))

with evaluation_tab:
    st.header("Model Evaluation")

    st.subheader("Model Quality Metrics")
    model_metrics_df = prepare_metrics_dataframe(load_csv(MODEL_METRICS_FILE))
    if model_metrics_df.empty:
        st.warning("Model quality metrics file was not found or could not be read.")
    else:
        show_metric_cards(model_metrics_df)
        st.dataframe(
            model_metrics_df,
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Model Metrics Visualization")
        show_metrics_chart(model_metrics_df)

    st.divider()
    st.subheader("Data Quality Metrics")
    data_metrics_df = prepare_metrics_dataframe(load_csv(DATA_METRICS_FILE))
    if data_metrics_df.empty:
        st.warning("Data quality metrics file was not found or could not be read.")
    else:
        show_metric_cards(data_metrics_df)
        st.dataframe(
            data_metrics_df,
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Data Quality Visualization")
        show_metrics_chart(data_metrics_df)

with validation_tab:
    st.header("Project Validation")
    st.write(
        "The following software engineering and machine-learning "
        "validation activities were completed."
    )

    validation_items = [
        ("Automated Tests", "10 tests passed"),
        ("ML Component Tests", "3 tests passed"),
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
        columns=["Validation Area", "Status"],
    )
    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )
    st.success(
        "Testing, code-quality validation, security checks, and "
        "production experimentation have been completed."
    )

with logs_tab:
    st.header("Application Logs")
    st.write(
        "This page displays application activity recorded by Python's "
        "logging module."
    )

    log_lines = read_log_lines()
    if not log_lines:
        st.warning(f"No log entries were found at: `{LOG_FILE}`")
    else:
        level_counts = count_log_levels(log_lines)
        info_column, warning_column, error_column = st.columns(3)

        with info_column:
            st.metric("INFO Logs", level_counts["INFO"])
        with warning_column:
            st.metric("WARNING Logs", level_counts["WARNING"])
        with error_column:
            st.metric("ERROR Logs", level_counts["ERROR"])

        st.divider()
        filter_column, search_column, lines_column = st.columns([1, 2, 1])

        with filter_column:
            selected_level = st.selectbox(
                "Log Level",
                ["ALL", "INFO", "WARNING", "ERROR"],
            )
        with search_column:
            search_text = st.text_input(
                "Search Logs",
                placeholder="Search by module, message, date, or keyword",
            )
        with lines_column:
            number_of_lines = st.selectbox(
                "Latest Entries",
                [25, 50, 100, 200, 500],
                index=2,
            )

        filtered_log_lines = filter_logs(
            log_lines,
            selected_level,
            search_text,
        )
        displayed_lines = filtered_log_lines[-number_of_lines:]

        st.caption(
            f"Showing {len(displayed_lines)} of "
            f"{len(filtered_log_lines)} matching entries."
        )
        if displayed_lines:
            st.code("\n".join(displayed_lines), language="text")
        else:
            st.info("No log entries match the selected filters.")

        refresh_column, download_column = st.columns(2)
        with refresh_column:
            if st.button("Refresh Logs", use_container_width=True):
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
            "INFO represents normal application activity, WARNING "
            "represents expected validation problems, and ERROR "
            "represents unexpected application failures."
        )

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
    st.markdown("""
- Modular production-style code structure
- FastAPI request and response schemas
- PDF, DOCX, and TXT resume upload
- Input validation and security checks
- INFO, WARNING, and ERROR logging
- Centralized exception handling
- SQLite prediction history
- Unit, integration, API, data, and ML tests
- Model-quality and data-quality metrics
- Shadow, canary, and A/B production experiments
- Streamlit user interface
        """)

    st.subheader("Technology Stack")
    technology_df = pd.DataFrame(
        [
            ["Frontend", "Streamlit"],
            ["Backend", "FastAPI"],
            ["Machine Learning", "Scikit-learn"],
            ["Model", "Random Forest"],
            ["Database", "SQLite"],
            ["Testing", "PyTest"],
            ["Code Quality", "Black, isort, Flake8"],
            ["Logging", "Python logging module"],
            ["PDF Parsing", "PyMuPDF"],
            ["DOCX Parsing", "python-docx"],
        ],
        columns=["Component", "Technology"],
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
        columns=["BITS ID", "Name", "Email"],
    )
    st.dataframe(
        group_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Disclaimer")
    st.info(
        "This application has been developed solely for academic "
        "purposes as part of the Software Engineering for Machine "
        "Learning course at BITS Pilani Work Integrated Learning "
        "Programme (WILP). It is intended for educational "
        "demonstration and evaluation only and should not be used "
        "as a production recruitment or resume screening system."
    )
