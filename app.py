import io
import urllib.parse
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# AI ONBOARDING CONTROL TOWER
# Python + Streamlit
# ============================================================


# ============================================================
# 1. PROCESS CONFIGURATION
# ============================================================

CORE_CHECKLIST = [
    "Employee Info Form",
    "Federal W-4",
    "Iowa W-4",
    "Direct Deposit",
    "I-9",
    "I-9 Supporting Docs",
    "Property Agreement",
    "Internet Use Policy",
    "Payroll Authorization",
    "Email/Computer Request",
    "Handbook Ack",
]

PAYROLL_PREREQS = [
    "Federal W-4",
    "Iowa W-4",
    "Direct Deposit",
]

RAW_REQUIRED = [
    "Employee ID",
    "Employee Name",
    "Department",
    "Start Date",
    "Offer Accepted",
    *CORE_CHECKLIST,
    "SafeColleges",
    "HRIS Employee ID",
    "Payroll Screens",
    "HR Notified IT",
    "IT Email/System Access",
]

OPTIONAL_CONTACT_FIELDS = [
    "Employee Email",
    "Supervisor Name",
    "Supervisor Email",
    "Preferred Reminder Channel",
]

STAGE_ORDER = [
    "Not Started",
    "Checklist In Progress",
    "Ready for HRIS Setup",
    "Compliance Pending",
    "Ready for IT Notification",
    "IT Provisioning",
    "Exception / Payroll Risk",
    "Fully Operational",
]

RISK_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
    "Complete",
]

WORKFLOW_DEPTH = {
    "Not Started": 0,
    "Checklist In Progress": 1,
    "Ready for HRIS Setup": 2,
    "Exception / Payroll Risk": 3,
    "Compliance Pending": 4,
    "Ready for IT Notification": 5,
    "IT Provisioning": 6,
    "Fully Operational": 7,
}

PRIORITY_ORDER = {
    "Red": 0,
    "Yellow": 1,
    "Green": 2,
}

DEMO_ACTIONS = [
    "Complete all core checklist items",
    "Complete payroll documents only",
    "Create HRIS employee ID + payroll screens",
    "Complete SafeColleges",
    "HR notifies IT",
    "IT completes email/system access",
]

DEMO_USERNAME = "HR.Admin"
DEMO_PASSWORD = "HR2026"


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def _norm(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_complete(value):
    text = _norm(value).lower()

    if text in {
        "",
        "no",
        "n",
        "false",
        "0",
        "pending",
        "not started",
        "missing",
        "incomplete",
        "open",
    }:
        return False

    return True


def _yes(value):
    return _norm(value).lower() in {
        "yes",
        "y",
        "true",
        "1",
        "complete",
        "completed",
        "done",
    }


def _created(value):
    text = _norm(value).lower()

    return text not in {
        "",
        "no",
        "n",
        "false",
        "0",
        "pending",
        "not created",
        "not started",
        "missing",
    }


def _coerce_date(series):
    """
    Converts standard dates and Excel serial dates into datetime values.
    """
    result = pd.to_datetime(series, errors="coerce")

    numeric_values = pd.to_numeric(series, errors="coerce")

    excel_mask = result.isna() & numeric_values.notna()

    if excel_mask.any():
        result.loc[excel_mask] = pd.to_datetime(
            numeric_values.loc[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return result


def validate_schema(df):
    missing = [column for column in RAW_REQUIRED if column not in df.columns]
    return missing


def _calculate_days_to_start(start_dates, as_of_date):
    return (
        start_dates.dt.normalize()
        - pd.Timestamp(as_of_date)
    ).dt.days


# ============================================================
# 3. CORE CONTROL TOWER ANALYSIS
# ============================================================

def analyze_rows(df, as_of_date=None):

    df = df.copy()

    if as_of_date is None:
        as_of_date = date.today()

    # Make sure required columns exist
    for column in RAW_REQUIRED:
        if column not in df.columns:
            df[column] = ""

    start_dates = _coerce_date(df["Start Date"])

    df["Start Date"] = start_dates
    df["Start Date Parse Status"] = np.where(
        start_dates.notna(),
        "Valid",
        "Missing / Invalid",
    )

    df["Imported Days to Start"] = _calculate_days_to_start(
        start_dates,
        as_of_date,
    )

    df["Days to Start"] = df["Imported Days to Start"]

    readiness_percentages = []
    readiness_ready = []
    stages = []
    bottlenecks = []
    blocked_tasks = []
    next_actions = []
    responsible_parties = []
    risks = []
    exceptions = []
    missing_items_list = []
    explanations = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # Core checklist
        # ----------------------------------------------------

        missing_core = [
            item
            for item in CORE_CHECKLIST
            if not _is_complete(row.get(item, ""))
        ]

        completed_core = len(CORE_CHECKLIST) - len(missing_core)

        readiness = (
            completed_core / len(CORE_CHECKLIST) * 100
            if CORE_CHECKLIST
            else 0
        )

        readiness_percentages.append(round(readiness, 1))
        readiness_ready.append(len(missing_core) == 0)

        # ----------------------------------------------------
        # Payroll
        # ----------------------------------------------------

        missing_payroll = [
            item
            for item in PAYROLL_PREREQS
            if not _is_complete(row.get(item, ""))
        ]

        # ----------------------------------------------------
        # System / workflow status
        # ----------------------------------------------------

        hris_created = _created(row.get("HRIS Employee ID", ""))
        payroll_created = _created(row.get("Payroll Screens", ""))
        hr_notified_it = _yes(row.get("HR Notified IT", ""))
        it_access = _yes(row.get("IT Email/System Access", ""))

        safe_colleges_complete = _is_complete(
            row.get("SafeColleges", "")
        )

        # ----------------------------------------------------
        # Determine stage
        # ----------------------------------------------------

        if (
            len(missing_core) == 0
            and safe_colleges_complete
            and hris_created
            and payroll_created
            and hr_notified_it
            and it_access
        ):
            stage = "Fully Operational"

        elif (
            hris_created
            and payroll_created
            and len(missing_payroll) > 0
        ):
            stage = "Exception / Payroll Risk"

        elif hr_notified_it and not it_access:
            stage = "IT Provisioning"

        elif (
            len(missing_core) == 0
            and safe_colleges_complete
            and not hr_notified_it
        ):
            stage = "Ready for IT Notification"

        elif (
            len(missing_core) == 0
            and not safe_colleges_complete
        ):
            stage = "Compliance Pending"

        elif (
            len(missing_core) > 0
            and hris_created
            and payroll_created
        ):
            stage = "Compliance Pending"

        elif (
            len(missing_core) == 0
            and safe_colleges_complete
            and not hris_created
        ):
            stage = "Ready for HRIS Setup"

        elif completed_core == 0:
            stage = "Not Started"

        else:
            stage = "Checklist In Progress"

        stages.append(stage)

        # ----------------------------------------------------
        # Bottleneck / next action
        # ----------------------------------------------------

        if stage == "Fully Operational":

            bottleneck = "None"
            blocked_task = "None"
            next_action = "No action required"
            responsible = "None"

        elif stage == "Exception / Payroll Risk":

            bottleneck = "Payroll prerequisites"
            blocked_task = "Payroll readiness"
            next_action = (
                "Complete missing payroll documents: "
                + ", ".join(missing_payroll)
            )
            responsible = "HR / New Hire"

        elif stage == "IT Provisioning":

            bottleneck = "IT system access"
            blocked_task = "System access"
            next_action = "Complete email and system access"
            responsible = "IT"

        elif stage == "Ready for IT Notification":

            bottleneck = "HR notification"
            blocked_task = "IT provisioning"
            next_action = "HR should notify IT"
            responsible = "HR"

        elif stage == "Compliance Pending":

            if not safe_colleges_complete:
                bottleneck = "SafeColleges"
                blocked_task = "Compliance completion"
                next_action = "Complete SafeColleges training"
                responsible = "New Hire"
            else:
                bottleneck = (
                    missing_core[0]
                    if missing_core
                    else "Compliance review"
                )
                blocked_task = "Onboarding readiness"
                next_action = (
                    "Complete remaining compliance items"
                )
                responsible = "HR / New Hire"

        elif stage == "Ready for HRIS Setup":

            bottleneck = "HRIS setup"
            blocked_task = "Employee system setup"
            next_action = "Create HRIS employee ID and payroll screens"
            responsible = "HR"

        elif stage == "Checklist In Progress":

            bottleneck = (
                missing_core[0]
                if missing_core
                else "Core checklist"
            )
            blocked_task = "HRIS setup"
            next_action = (
                "Complete missing item: "
                + bottleneck
            )
            responsible = "HR / New Hire"

        else:

            bottleneck = "Onboarding initiation"
            blocked_task = "Core checklist"
            next_action = "Begin onboarding checklist"
            responsible = "HR"

        bottlenecks.append(bottleneck)
        blocked_tasks.append(blocked_task)
        next_actions.append(next_action)
        responsible_parties.append(responsible)

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        days = row.get("Days to Start")

        if stage == "Fully Operational":
            risk = "Complete"

        elif pd.isna(days):
            risk = "Medium"

        elif days < 0:
            risk = "Critical"

        elif days <= 7:
            risk = "High"

        elif days <= 14:
            risk = "Medium"

        else:
            risk = "Low"

        risks.append(risk)

        # ----------------------------------------------------
        # Exception
        # ----------------------------------------------------

        if (
            hris_created
            and payroll_created
            and len(missing_payroll) > 0
        ):
            exception = "Exception / Payroll Risk"

        else:
            exception = ""

        exceptions.append(exception)

        # ----------------------------------------------------
        # Missing items
        # ----------------------------------------------------

        missing_items_list.append(
            ", ".join(missing_core)
            if missing_core
            else "None"
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        if stage == "Fully Operational":
            explanation = (
                "The employee has completed the core onboarding "
                "requirements and required downstream system steps."
            )

        elif stage == "Exception / Payroll Risk":
            explanation = (
                "HRIS and payroll setup are progressing before "
                "all payroll prerequisites are complete."
            )

        elif stage == "IT Provisioning":
            explanation = (
                "HR has notified IT, but the employee's email "
                "and system access are not yet complete."
            )

        elif stage == "Ready for IT Notification":
            explanation = (
                "Core onboarding and compliance requirements are "
                "complete. The next dependency is HR notifying IT."
            )

        elif stage == "Compliance Pending":
            explanation = (
                "Core onboarding is progressing, but compliance "
                "requirements remain before downstream setup."
            )

        elif stage == "Ready for HRIS Setup":
            explanation = (
                "Core requirements are complete and the employee "
                "is ready for HRIS and payroll setup."
            )

        elif stage == "Checklist In Progress":
            explanation = (
                "The onboarding checklist is incomplete. "
                f"The current bottleneck is {bottleneck}."
            )

        else:
            explanation = (
                "The onboarding workflow has not yet started."
            )

        explanations.append(explanation)

    # --------------------------------------------------------
    # Add derived columns
    # --------------------------------------------------------

    df["Core Checklist %"] = readiness_percentages
    df["Core Checklist Ready?"] = readiness_ready
    df["Current Stage"] = stages
    df["Primary Bottleneck"] = bottlenecks
    df["Blocked Downstream Task"] = blocked_tasks
    df["Next Action"] = next_actions
    df["Responsible Party"] = responsible_parties
    df["Risk Level"] = risks
    df["Exception Flag"] = exceptions
    df["Missing Core Items"] = missing_items_list
    df["Control Tower Explanation"] = explanations

    return df


# ============================================================
# 4. PRIORITIZATION
# ============================================================

def prioritize_df(df):

    df = df.copy()

    priorities = []

    for _, row in df.iterrows():

        days = row.get("Days to Start")
        stage = row.get("Current Stage", "")

        if stage == "Fully Operational":
            priority = "Green"

        elif (
            pd.notna(days)
            and days <= 7
        ):
            priority = "Red"

        else:
            priority = "Yellow"

        priorities.append(priority)

    df["Priority"] = priorities

    def priority_value(value):
        return PRIORITY_ORDER.get(value, 9)

    df["_Priority Sort"] = df["Priority"].map(priority_value)

    df["_Workflow Sort"] = (
        df["Current Stage"]
        .map(WORKFLOW_DEPTH)
        .fillna(0)
    )

    df["_Days Sort"] = pd.to_numeric(
        df["Days to Start"],
        errors="coerce",
    ).fillna(9999)

    df["_Employee Sort"] = (
        df["Employee Name"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    df = df.sort_values(
        by=[
            "_Priority Sort",
            "_Days Sort",
            "_Workflow Sort",
            "_Employee Sort",
        ]
    )

    df = df.drop(
        columns=[
            "_Priority Sort",
            "_Workflow Sort",
            "_Days Sort",
            "_Employee Sort",
        ],
        errors="ignore",
    )

    return df.reset_index(drop=True)


# ============================================================
# 5. COMMUNICATION HELPERS
# ============================================================

def _directory_email(row):
    return _norm(row.get("Employee Email", ""))


def resolve_recipient(row):

    preferred = _norm(
        row.get("Preferred Reminder Channel", "")
    ).lower()

    employee_email = _directory_email(row)
    supervisor_email = _norm(
        row.get("Supervisor Email", "")
    )

    if preferred == "supervisor" and supervisor_email:
        return supervisor_email

    if preferred == "employee" and employee_email:
        return employee_email

    if employee_email:
        return employee_email

    if supervisor_email:
        return supervisor_email

    return "hr.onboarding@example.com"


def suggested_subject(row):
    employee_name = _norm(row.get("Employee Name", "Employee"))

    return (
        f"Onboarding follow-up required — {employee_name}"
    )


def deterministic_reminder(row):

    employee_name = _norm(row.get("Employee Name", "Employee"))
    stage = _norm(row.get("Current Stage", ""))
    next_action = _norm(row.get("Next Action", ""))
    days = row.get("Days to Start")

    if pd.isna(days):
        timing = "Please review the onboarding timeline."
    elif days < 0:
        timing = (
            "The employee's start date has already passed, "
            "so this item should be addressed immediately."
        )
    elif days <= 7:
        timing = (
            "The employee is scheduled to start within 7 days."
        )
    else:
        timing = (
            f"The employee is scheduled to start in approximately "
            f"{int(days)} days."
        )

    return (
        f"Hello,\n\n"
        f"This is a follow-up regarding the onboarding status "
        f"for {employee_name}.\n\n"
        f"Current stage: {stage}\n"
        f"Next action: {next_action}\n\n"
        f"{timing}\n\n"
        f"Please review the outstanding onboarding item and "
        f"complete the required action.\n\n"
        f"Thank you,\n"
        f"HR Operations"
    )


# ============================================================
# 6. GOOGLE SHEETS
# ============================================================

def _extract_sheet_id(sheet_url_or_id):

    value = _norm(sheet_url_or_id)

    if not value:
        return None

    if "docs.google.com/spreadsheets/d/" in value:

        try:
            return value.split(
                "/spreadsheets/d/"
            )[1].split("/")[0]
        except Exception:
            return None

    return value


def read_public_google_sheet(
    sheet_url_or_id,
    worksheet_name="Employee_Data",
):

    sheet_id = _extract_sheet_id(sheet_url_or_id)

    if not sheet_id:
        raise ValueError(
            "A valid Google Sheet URL or ID is required."
        )

    encoded_sheet = urllib.parse.quote(
        worksheet_name
    )

    url = (
        "https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/gviz/tq?tqx=out:csv"
        f"&sheet={encoded_sheet}"
    )

    response = requests.get(
        url,
        timeout=20,
    )

    response.raise_for_status()

    return pd.read_csv(
        io.StringIO(response.text)
    )


# ============================================================
# 7. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Onboarding Control Tower",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 8. NATIVE STREAMLIT VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------
       GLOBAL
    ------------------------------------------------------ */

    .stApp {
        background-color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #111111;
        letter-spacing: -0.02em;
    }

    p {
        color: #3f3f3f;
    }

    /* ------------------------------------------------------
       BUTTONS
    ------------------------------------------------------ */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #111111;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #f4c542;
        color: #111111;
    }

    /* ------------------------------------------------------
       METRICS
    ------------------------------------------------------ */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #666666;
        font-size: 0.85rem;
    }

    [data-testid="stMetricValue"] {
        color: #111111;
        font-weight: 700;
    }

    /* ------------------------------------------------------
       TABLES
    ------------------------------------------------------ */

    [data-testid="stDataFrame"] {
        border: 1px solid #dddddd;
        border-radius: 8px;
    }

    /* ------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------ */

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* ------------------------------------------------------
       PROGRESS
    ------------------------------------------------------ */

    [data-testid="stProgressBar"] > div > div > div {
        background-color: #f4c542;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. SESSION STATE
# ============================================================

DEFAULT_SESSION = {
    "authenticated": False,
    "main_view": "Overview",
    "selected_employee_id": None,
    "google_sheet_url": "",
    "google_sheet_loaded": False,
    "google_sheet_last_refresh": None,
    "login_attempted": False,
    "employee_df": None,
    "test_cases_df": None,
    "contact_directory_df": None,
    "data_source": None,
}

for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 10. LOGIN
# ============================================================

def show_login():

    st.sidebar.empty()

    st.write("")

    left, center, right = st.columns(
        [1, 1.3, 1]
    )

    with center:

        st.markdown(
            "## ◈ AI ONBOARDING CONTROL TOWER"
        )

        st.caption(
            "HR Operations Visibility & Workflow Intelligence"
        )

        st.write("")

        with st.container(border=True):

            st.subheader("Sign in")

            username = st.text_input(
                "Username",
                placeholder="Enter your HR username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )

            remember = st.checkbox(
                "Remember this session"
            )

            st.write("")

            sign_in = st.button(
                "Sign in",
                type="primary",
                use_container_width=True,
            )

            if sign_in:

                st.session_state.login_attempted = True

                if (
                    username == DEMO_USERNAME
                    and password == DEMO_PASSWORD
                ):

                    st.session_state.authenticated = True

                    st.session_state.login_attempted = False

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        st.caption(
            "Demo credentials are configured for classroom prototype use."
        )


# ============================================================
# 11. SIDEBAR
# ============================================================

NAV_OPTIONS = [
    "Overview",
    "Employee Directory",
    "Risk Monitor",
    "Analytics",
    "Communications",
    "Rule Validation",
    "Data Management",
    #"Employee Profile",
    "About",
]

NAV_ICONS = {
    "Overview": "⌂",
    "Employee Directory": "◉",
    "Risk Monitor": "!",
    "Analytics": "▥",
    "Communications": "✉",
    "Rule Validation": "✓",
    "Data Management": "⇧",
    "About": "i",
}


def show_sidebar():

    with st.sidebar:

        st.markdown("## ◈")
        st.markdown("### AI Onboarding")
        st.caption("CONTROL TOWER")

        st.divider()

        # changing below 
        current_navigation_view = st.session_state.main_view

        if current_navigation_view == "Employee Profile":
            current_navigation_view = "Employee Directory"

        selected = st.radio(
            "Navigation",
            NAV_OPTIONS,
            index=NAV_OPTIONS.index(
                current_navigation_view
            ),
            format_func=lambda item: (
                f"{NAV_ICONS.get(item, '')}   {item}"
            ),
            label_visibility="collapsed",
        )
#changing above
        #if selected != st.session_state.main_view:
            #st.session_state.main_view = selected
            #st.rerun()
        if (
            st.session_state.main_view != "Employee Profile"
            and selected != st.session_state.main_view
        ):
            st.session_state.main_view = selected
            st.rerun()
            ### changing above 2.0

        st.divider()

        st.caption("CONTROL DATE")

        control_date = st.date_input(
            "As of",
            value=date.today(),
            label_visibility="collapsed",
        )

        st.session_state.control_date = control_date

        st.divider()

        st.caption("SIGNED IN")

        st.write("**HR Admin**")
        st.caption("HR Administrator")

        if st.button(
            "Sign out",
            use_container_width=True,
        ):
            st.session_state.authenticated = False
            st.rerun()


# ============================================================
# 12. DATA MANAGEMENT
# ============================================================

def load_excel(uploaded_file):

    excel = pd.ExcelFile(uploaded_file)

    sheet_names = excel.sheet_names

    if "Employee_Data" in sheet_names:
        employee_df = pd.read_excel(
            uploaded_file,
            sheet_name="Employee_Data",
        )
    else:
        employee_df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_names[0],
        )

    test_cases_df = None
    contact_directory_df = None

    if "Test_Cases" in sheet_names:
        test_cases_df = pd.read_excel(
            uploaded_file,
            sheet_name="Test_Cases",
        )

    if "Contact_Directory" in sheet_names:
        contact_directory_df = pd.read_excel(
            uploaded_file,
            sheet_name="Contact_Directory",
        )

    st.session_state.employee_df = employee_df
    st.session_state.test_cases_df = test_cases_df
    st.session_state.contact_directory_df = contact_directory_df
    st.session_state.data_source = (
        "Excel workbook"
    )


def load_google_data(sheet_url):

    employee_df = read_public_google_sheet(
        sheet_url,
        "Employee_Data",
    )

    test_cases_df = None
    contact_directory_df = None

    try:
        test_cases_df = read_public_google_sheet(
            sheet_url,
            "Test_Cases",
        )
    except Exception:
        pass

    try:
        contact_directory_df = read_public_google_sheet(
            sheet_url,
            "Contact_Directory",
        )
    except Exception:
        pass

    st.session_state.employee_df = employee_df
    st.session_state.test_cases_df = test_cases_df
    st.session_state.contact_directory_df = contact_directory_df
    st.session_state.data_source = (
        "Live Google Sheet"
    )
    st.session_state.google_sheet_loaded = True
    st.session_state.google_sheet_last_refresh = (
        datetime.now()
    )


def show_data_management():

    st.title("Data Management")

    st.write(
        "Connect the Control Tower to the employee onboarding "
        "dataset used by the HR team."
    )

    st.divider()

    excel_tab, google_tab = st.tabs(
        [
            "Excel Snapshot",
            "Live Google Sheet",
        ]
    )

    with excel_tab:

        st.subheader("Upload onboarding data")

        st.write(
            "Use an Excel workbook containing the "
            "`Employee_Data` sheet."
        )

        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=["xlsx", "xls"],
        )

        if uploaded_file is not None:

            if st.button(
                "Load Excel Data",
                type="primary",
            ):

                try:

                    load_excel(uploaded_file)

                    st.success(
                        "Employee data loaded successfully."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to load the workbook: {error}"
                    )

    with google_tab:

        st.subheader("Connect a Google Sheet")

        st.write(
            "The workbook must be shared as "
            "**Anyone with the link — Viewer** for the demo."
        )

        google_url = st.text_input(
            "Google Sheet URL",
            value=st.session_state.google_sheet_url,
            placeholder="Paste Google Sheets URL here",
        )

        st.session_state.google_sheet_url = google_url

        if st.button(
            "Load / Refresh Google Sheet",
            type="primary",
        ):

            if not google_url.strip():

                st.warning(
                    "Please enter a Google Sheet URL."
                )

            else:

                try:

                    load_google_data(
                        google_url
                    )

                    st.success(
                        "Google Sheet data loaded successfully."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "Unable to load the Google Sheet. "
                        "Check that the URL is correct and "
                        "the sheet is publicly viewable."
                    )

                    st.caption(
                        f"Technical detail: {error}"
                    )

    st.divider()

    if st.session_state.employee_df is not None:

        df = st.session_state.employee_df

        st.subheader("Current data source")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Employees loaded",
                len(df),
            )

        with c2:
            st.metric(
                "Columns",
                len(df.columns),
            )

        with c3:
            st.metric(
                "Source",
                st.session_state.data_source
                or "Unknown",
            )

        if (
            st.session_state.google_sheet_last_refresh
            is not None
        ):

            st.caption(
                "Last refreshed: "
                + st.session_state.google_sheet_last_refresh.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        missing = validate_schema(df)

        if missing:

            st.error(
                "Required fields are missing from the dataset."
            )

            st.write(missing)

        else:

            st.success(
                "Dataset structure is compatible with the "
                "Control Tower workflow."
            )

            with st.expander(
                "Preview employee data"
            ):

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

    else:

        st.info(
            "No employee dataset is currently loaded."
        )


# ============================================================
# 13. DATA HELPERS
# ============================================================

def employee_choices(df):

    if df is None or df.empty:
        return []

    return (
        df["Employee ID"]
        .astype(str)
        .tolist()
    )


def get_employee(df, employee_id):

    if df is None or df.empty:
        return None

    matches = df[
        df["Employee ID"].astype(str)
        == str(employee_id)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def format_date(value):

    if pd.isna(value):
        return "Not available"

    try:
        return pd.Timestamp(value).strftime(
            "%b %d, %Y"
        )
    except Exception:
        return str(value)


def risk_badge_text(risk):

    mapping = {
        "Critical": "🔴 Critical",
        "High": "🟠 High",
        "Medium": "🟡 Medium",
        "Low": "🟢 Low",
        "Complete": "✓ Complete",
    }

    return mapping.get(
        risk,
        risk,
    )


def priority_badge_text(priority):

    mapping = {
        "Red": "🔴 Immediate",
        "Yellow": "🟡 Monitor",
        "Green": "🟢 Complete",
    }

    return mapping.get(
        priority,
        priority,
    )


def show_readiness(df):

    value = float(
        df["Core Checklist %"]
        .iloc[0]
    )

    st.progress(
        min(max(value / 100, 0), 1)
    )

    st.caption(
        f"{value:.0f}% of core onboarding requirements complete"
    )


# ============================================================
# 14. PAGE HEADER
# ============================================================

def page_header(title, subtitle=None):

    st.title(title)

    if subtitle:
        st.caption(subtitle)

    st.divider()


# ============================================================
# 15. OVERVIEW
# ============================================================

def show_overview(df):

    total_employees = len(df)

    ready_count = int(
        (df["Current Stage"] == "Fully Operational")
        .sum()
    )

    critical_count = int(
        (df["Risk Level"] == "Critical")
        .sum()
    )

    pending_count = int(
        (
            df["Current Stage"]
            != "Fully Operational"
        ).sum()
    )

    avg_checklist = (
        df["Core Checklist %"].mean()
        if total_employees
        else 0
    )

    st.title(
        "Welcome to the AI Onboarding Control Tower"
    )

    st.caption(
        "A centralized view of employee onboarding readiness, "
        "workflow dependencies, and action items."
    )

    st.write("")

    # --------------------------------------------------------
    # Top KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Employees",
            total_employees,
        )

    with c2:
        st.metric(
            "Ready",
            ready_count,
        )

    with c3:
        st.metric(
            "Needs Attention",
            pending_count,
        )

    with c4:
        st.metric(
            "Critical",
            critical_count,
        )

    with c5:
        st.metric(
            "Avg. Readiness",
            f"{avg_checklist:.0f}%",
        )

    st.write("")

    # --------------------------------------------------------
    # Control Tower
    # --------------------------------------------------------

    st.subheader("Control Tower")

    left, middle, right = st.columns(
        [1.15, 1.15, 1]
    )

    with left:

        with st.container(border=True):

            st.markdown("### Onboarding Status")

            stage_counts = (
                df["Current Stage"]
                .value_counts()
                .to_dict()
            )

            for stage in STAGE_ORDER:

                count = stage_counts.get(
                    stage,
                    0,
                )

                if count > 0:

                    st.write(
                        f"**{stage}**  ·  {count}"
                    )

    with middle:

        with st.container(border=True):

            st.markdown("### Action Needed")

            attention_df = df[
                df["Priority"].isin(
                    ["Red", "Yellow"]
                )
            ].head(5)

            if attention_df.empty:

                st.success(
                    "No outstanding actions."
                )

            else:

                for _, row in attention_df.iterrows():

                    st.write(
                        f"**{row['Employee Name']}**"
                    )

                    st.caption(
                        f"{row['Risk Level']} · "
                        f"{row['Next Action']}"
                    )

                    st.divider()

    with right:

        with st.container(border=True):

            st.markdown("### Search Employee")

            search = st.text_input(
                "Search",
                placeholder=(
                    "Name, ID, or department"
                ),
                label_visibility="collapsed",
            )

            if search:

                search_text = search.lower()

                matches = df[
                    df["Employee Name"]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        na=False,
                    )
                    |
                    df["Employee ID"]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        na=False,
                    )
                    |
                    df["Department"]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        na=False,
                    )
                ]

                if matches.empty:

                    st.info(
                        "No employee found."
                    )

                else:

                    for _, row in matches.head(5).iterrows():

                        if st.button(
                            row["Employee Name"],
                            key=(
                                "overview_employee_"
                                + str(row["Employee ID"])
                            ),
                            use_container_width=True,
                        ):

                            st.session_state.selected_employee_id = (
                                row["Employee ID"]
                            )

                            st.session_state.main_view = (
                                "Employee Profile"
                            )

                            st.rerun()

    st.write("")

    # --------------------------------------------------------
    # Priority queue
    # --------------------------------------------------------

    st.subheader("Priority Queue")

    queue = df[
        df["Priority"].isin(
            ["Red", "Yellow"]
        )
    ].head(10)

    if queue.empty:

        st.success(
            "No employees currently require follow-up."
        )

    else:

        display = queue[
            [
                "Priority",
                "Employee ID",
                "Employee Name",
                "Department",
                "Start Date",
                "Days to Start",
                "Current Stage",
                "Next Action",
                "Responsible Party",
            ]
        ].copy()

        display["Start Date"] = display[
            "Start Date"
        ].apply(format_date)

        display["Priority"] = display[
            "Priority"
        ].apply(priority_badge_text)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 16. EMPLOYEE DIRECTORY
# ============================================================

def show_employee_directory(df):

    page_header(
        "Employee Directory",
        "Search onboarding records and open an employee control view.",
    )

    c1, c2, c3 = st.columns(
        [2, 1, 1]
    )

    with c1:

        search = st.text_input(
            "Search employee",
            placeholder=(
                "Employee name or ID"
            ),
        )

    with c2:

        departments = [
            "All departments"
        ] + sorted(
            df["Department"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        department = st.selectbox(
            "Department",
            departments,
        )

    with c3:

        risk_filter = st.selectbox(
            "Risk",
            [
                "All",
                "Critical",
                "High",
                "Medium",
                "Low",
                "Complete",
            ],
        )

    filtered = df.copy()

    if search:

        search_text = search.lower()

        filtered = filtered[
            filtered["Employee Name"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False,
            )
            |
            filtered["Employee ID"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False,
            )
        ]

    if department != "All departments":

        filtered = filtered[
            filtered["Department"]
            .astype(str)
            == department
        ]

    if risk_filter != "All":

        filtered = filtered[
            filtered["Risk Level"]
            == risk_filter
        ]

    st.caption(
        f"{len(filtered)} employee(s) displayed"
    )

    for _, row in filtered.iterrows():

        with st.container(border=True):

            left, middle, right = st.columns(
                [2.2, 2.2, 1]
            )

            with left:

                st.markdown(
                    f"### {row['Employee Name']}"
                )

                st.caption(
                    f"{row['Employee ID']} · "
                    f"{row['Department']}"
                )

                st.write(
                    f"Start date: **{format_date(row['Start Date'])}**"
                )

            with middle:

                st.write(
                    f"**{risk_badge_text(row['Risk Level'])}**"
                )

                st.write(
                    f"Stage: **{row['Current Stage']}**"
                )

                st.progress(
                    float(
                        row["Core Checklist %"]
                    ) / 100
                )

                st.caption(
                    f"{row['Core Checklist %']:.0f}% ready"
                )

            with right:

                if st.button(
                    "View employee",
                    key=(
                        "directory_view_"
                        + str(row["Employee ID"])
                    ),
                    use_container_width=True,
                ):

                    st.session_state.selected_employee_id = (
                        row["Employee ID"]
                    )

                    st.session_state.main_view = (
                        "Employee Profile"
                    )

                    st.rerun()


# ============================================================
# 17. EMPLOYEE PROFILE
# ============================================================

def show_employee_profile(df):

    employee_id = (
        st.session_state.selected_employee_id
    )

    if employee_id is None:

        st.info(
            "Select an employee from the Employee Directory."
        )

        if st.button("Go to Employee Directory"):

            st.session_state.main_view = (
                "Employee Directory"
            )

            st.rerun()

        return

    row = get_employee(
        df,
        employee_id,
    )

    if row is None:

        st.error(
            "Employee record could not be found."
        )

        return

    if st.button("← Back to Employee Directory"):

        st.session_state.main_view = (
            "Employee Directory"
        )

        st.rerun()

    st.write("")

    st.title(
        str(row["Employee Name"])
    )

    st.caption(
        f"{row['Employee ID']} · "
        f"{row['Department']}"
    )

    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Start Date",
            format_date(row["Start Date"]),
        )

    with c2:
        st.metric(
            "Readiness",
            f"{row['Core Checklist %']:.0f}%",
        )

    with c3:
        st.metric(
            "Risk",
            row["Risk Level"],
        )

    with c4:
        st.metric(
            "Stage",
            row["Current Stage"],
        )

    with c5:
        days = row["Days to Start"]

        if pd.isna(days):
            value = "N/A"
        else:
            value = str(int(days))

        st.metric(
            "Days to Start",
            value,
        )

    st.write("")

    # --------------------------------------------------------
    # Workflow intelligence
    # --------------------------------------------------------

    st.subheader("Workflow Intelligence")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Primary Bottleneck**")

        st.write(
            row["Primary Bottleneck"]
        )

    with c2:

        st.markdown("**Next Action**")

        st.write(
            row["Next Action"]
        )

    with c3:

        st.markdown("**Responsible Party**")

        st.write(
            row["Responsible Party"]
        )

    st.write("")

    st.markdown("**Readiness**")

    st.progress(
        float(
            row["Core Checklist %"]
        ) / 100
    )

    st.caption(
        f"{row['Core Checklist %']:.0f}% of core checklist complete"
    )

    st.info(
        row["Control Tower Explanation"]
    )

    st.write("")

    # --------------------------------------------------------
    # Checklist
    # --------------------------------------------------------

    st.subheader("Onboarding Checklist")

    checklist_rows = []

    for item in CORE_CHECKLIST:

        completed = _is_complete(
            row.get(item, "")
        )

        checklist_rows.append(
            {
                "Requirement": item,
                "Status": (
                    "Complete"
                    if completed
                    else "Pending"
                ),
            }
        )

    checklist_df = pd.DataFrame(
        checklist_rows
    )

    st.dataframe(
        checklist_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # System readiness
    # --------------------------------------------------------

    st.subheader("System Readiness")

    system_rows = []

    system_items = [
        "HRIS Employee ID",
        "Payroll Screens",
        "HR Notified IT",
        "IT Email/System Access",
    ]

    for item in system_items:

        value = row.get(item, "")

        system_rows.append(
            {
                "System Step": item,
                "Status": (
                    "Complete"
                    if _is_complete(value)
                    else "Pending"
                ),
            }
        )

    system_df = pd.DataFrame(
        system_rows
    )

    st.dataframe(
        system_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Employee details
    # --------------------------------------------------------

    with st.expander("Employee Details"):

        details = {
            "Employee ID": row.get(
                "Employee ID",
                "",
            ),
            "Employee Name": row.get(
                "Employee Name",
                "",
            ),
            "Department": row.get(
                "Department",
                "",
            ),
            "Employee Email": row.get(
                "Employee Email",
                "",
            ),
            "Supervisor": row.get(
                "Supervisor Name",
                "",
            ),
            "Supervisor Email": row.get(
                "Supervisor Email",
                "",
            ),
        }

        st.dataframe(
            pd.DataFrame(
                details.items(),
                columns=[
                    "Field",
                    "Value",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.write("")

    if st.button(
        "Draft Follow-Up Email",
        type="primary",
    ):

        st.session_state.communication_employee = (
            row["Employee ID"]
        )

        st.session_state.main_view = (
            "Communications"
        )

        st.rerun()


# ============================================================
# 18. RISK MONITOR
# ============================================================

def show_risk_monitor(df):

    page_header(
        "Risk Monitor",
        "Prioritized onboarding exceptions and upcoming deadlines.",
    )

    critical = int(
        (df["Risk Level"] == "Critical").sum()
    )

    high = int(
        (df["Risk Level"] == "High").sum()
    )

    medium = int(
        (df["Risk Level"] == "Medium").sum()
    )

    complete = int(
        (df["Risk Level"] == "Complete").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Critical",
            critical,
        )

    with c2:
        st.metric(
            "High",
            high,
        )

    with c3:
        st.metric(
            "Medium",
            medium,
        )

    with c4:
        st.metric(
            "Complete",
            complete,
        )

    st.write("")

    risk_filter = st.multiselect(
        "Show risk levels",
        [
            "Critical",
            "High",
            "Medium",
            "Low",
            "Complete",
        ],
        default=[
            "Critical",
            "High",
            "Medium",
        ],
    )

    filtered = df[
        df["Risk Level"].isin(
            risk_filter
        )
    ].copy()

    if filtered.empty:

        st.success(
            "No employees match the selected filters."
        )

        return

    display = filtered[
        [
            "Priority",
            "Employee ID",
            "Employee Name",
            "Department",
            "Start Date",
            "Days to Start",
            "Current Stage",
            "Primary Bottleneck",
            "Next Action",
            "Responsible Party",
            "Risk Level",
        ]
    ].copy()

    display["Start Date"] = display[
        "Start Date"
    ].apply(format_date)

    display["Priority"] = display[
        "Priority"
    ].apply(priority_badge_text)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 19. ANALYTICS
# ============================================================

def show_analytics(df):

    page_header(
        "Analytics",
        "Monitor onboarding readiness, workflow stage, and department trends.",
    )

    # --------------------------------------------------------
    # Stage chart
    # --------------------------------------------------------

    stage_counts = (
        df["Current Stage"]
        .value_counts()
        .reindex(
            STAGE_ORDER,
            fill_value=0,
        )
        .reset_index()
    )

    stage_counts.columns = [
        "Stage",
        "Employees",
    ]

    fig_stage = px.bar(
        stage_counts,
        x="Stage",
        y="Employees",
        title="Employees by Onboarding Stage",
    )

    fig_stage.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="",
        yaxis_title="Employees",
    )

    st.plotly_chart(
        fig_stage,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Risk chart
    # --------------------------------------------------------

    risk_counts = (
        df["Risk Level"]
        .value_counts()
        .reindex(
            RISK_ORDER,
            fill_value=0,
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Employees",
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Employees",
        title="Employees by Risk Level",
    )

    fig_risk.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="",
        yaxis_title="Employees",
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Department performance
    # --------------------------------------------------------

    st.subheader("Department Overview")

    department_df = (
        df.groupby("Department")
        .agg(
            Employees=(
                "Employee ID",
                "count",
            ),
            Average_Readiness=(
                "Core Checklist %",
                "mean",
            ),
            Critical_Employees=(
                "Risk Level",
                lambda x: (
                    x == "Critical"
                ).sum(),
            ),
        )
        .reset_index()
    )

    department_df[
        "Average_Readiness"
    ] = department_df[
        "Average_Readiness"
    ].round(1)

    department_df = department_df.rename(
        columns={
            "Average_Readiness":
                "Average Readiness %",
            "Critical_Employees":
                "Critical Employees",
        }
    )

    st.dataframe(
        department_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 20. COMMUNICATIONS
# ============================================================

def show_communications(df):

    page_header(
        "Communications",
        "Generate a follow-up message based on the employee's current workflow status.",
    )

    employee_ids = employee_choices(df)

    if not employee_ids:

        st.warning(
            "No employee data available."
        )

        return

    default_id = st.session_state.get(
        "communication_employee",
        st.session_state.selected_employee_id,
    )

    if (
        default_id in employee_ids
    ):

        default_index = employee_ids.index(
            default_id
        )

    else:

        default_index = 0

    selected_id = st.selectbox(
        "Employee",
        employee_ids,
        index=default_index,
        format_func=lambda x: (
            f"{x} — "
            f"{get_employee(df, x)['Employee Name']}"
        ),
    )

    row = get_employee(
        df,
        selected_id,
    )

    if row is None:
        return

    st.write("")

    left, right = st.columns(
        [1.4, 1]
    )

    with left:

        st.subheader("Message")

        recipient = resolve_recipient(
            row
        )

        subject = suggested_subject(
            row
        )

        message = deterministic_reminder(
            row
        )

        to = st.text_input(
            "To",
            value=recipient,
        )

        cc = st.text_input(
            "CC",
            value="",
        )

        subject_input = st.text_input(
            "Subject",
            value=subject,
        )

        message_input = st.text_area(
            "Message",
            value=message,
            height=300,
        )

        if st.button(
            "Create Gmail Draft",
            type="primary",
        ):

            params = {
                "view": "cm",
                "fs": "1",
                "to": to,
                "cc": cc,
                "su": subject_input,
                "body": message_input,
            }

            gmail_url = (
                "https://mail.google.com/mail/"
                "u/0/?"
                + urllib.parse.urlencode(
                    params
                )
            )

            st.link_button(
                "Open Gmail Draft",
                gmail_url,
            )

    with right:

        st.subheader("Workflow Context")

        st.metric(
            "Current Stage",
            row["Current Stage"],
        )

        st.metric(
            "Risk",
            row["Risk Level"],
        )

        st.write(
            "**Next action**"
        )

        st.info(
            row["Next Action"]
        )

        st.write(
            "**Responsible party**"
        )

        st.write(
            row["Responsible Party"]
        )

        st.write(
            "**Days to start**"
        )

        days = row["Days to Start"]

        if pd.isna(days):
            st.write("Not available")
        else:
            st.write(
                f"{int(days)} days"
            )


# ============================================================
# 21. RULE VALIDATION
# ============================================================

def show_rule_validation(df):

    page_header(
        "Rule Validation",
        "Compare prototype output against expected test-case outcomes.",
    )

    test_cases = (
        st.session_state.test_cases_df
    )

    if test_cases is None:

        st.info(
            "No Test_Cases sheet is loaded. "
            "Upload the Excel workbook containing "
            "the Test_Cases sheet."
        )

        return

    if test_cases.empty:

        st.info(
            "The Test_Cases sheet is empty."
        )

        return

    st.subheader("Validation Results")

    results = []

    for _, test in test_cases.iterrows():

        employee_id = (
            test.get("Employee ID")
            or test.get("HRIS Employee ID")
        )

        if pd.isna(employee_id):
            continue

        employee_id = str(
            employee_id
        )

        actual_matches = df[
            df["Employee ID"]
            .astype(str)
            == employee_id
        ]

        if actual_matches.empty:
            continue

        actual = actual_matches.iloc[0]

        expected_stage = _norm(
            test.get(
                "Expected Stage",
                test.get(
                    "Stage",
                    "",
                ),
            )
        )

        expected_bottleneck = _norm(
            test.get(
                "Expected Bottleneck",
                test.get(
                    "Bottleneck",
                    "",
                ),
            )
        )

        expected_next_action = _norm(
            test.get(
                "Expected Next Action",
                test.get(
                    "Next Action",
                    "",
                ),
            )
        )

        stage_pass = (
            not expected_stage
            or expected_stage
            == _norm(
                actual["Current Stage"]
            )
        )

        bottleneck_pass = (
            not expected_bottleneck
            or expected_bottleneck
            == _norm(
                actual["Primary Bottleneck"]
            )
        )

        #next_action_pass = (
         #   not expected_next_action
          #  or expected_next_action
           # == _norm(
            #    actual["Next Action"]
            #)
        #)

        # actual_next_action = _norm(
        #     actual["Next Action"]
        # )
        # normalized_expected_action = (
        #     expected_next_action
        #     .replace(
        #         "complete missing item:",
        #         "complete"
        #     )
        #     .strip()
        # )
        
        # normalized_actual_action = (
        #     actual_next_action
        #     .replace(
        #         "complete missing item:",
        #         "complete"
        #     )
        #     .strip()
        # )
        # next_action_pass = (
        #     not normalized_expected_action
        #     or normalized_expected_action
        #     == normalized_actual_action
        # )
    ##
    actual_next_action = _norm(
        actual["Next Action"]
    ).lower()
    
    normalized_expected_action = (
        expected_next_action
        .lower()
        .replace(
            "complete missing item:",
            "complete"
        )
        .strip()
    )
    
    normalized_actual_action = (
        actual_next_action
        .replace(
            "complete missing item:",
            "complete"
        )
        .strip()
    )
    
    next_action_pass = (
        not normalized_expected_action
        or normalized_expected_action
        == normalized_actual_action
    )


        # new code above

        overall = (
            stage_pass
            and bottleneck_pass
            and next_action_pass
        )

        results.append(
            {
                "Employee ID": employee_id,
                "Employee Name": actual[
                    "Employee Name"
                ],
                "Stage": (
                    "PASS"
                    if stage_pass
                    else "FAIL"
                ),
                "Bottleneck": (
                    "PASS"
                    if bottleneck_pass
                    else "FAIL"
                ),
                "Next Action": (
                    "PASS"
                    if next_action_pass
                    else "FAIL"
                ),
                "Overall": (
                    "PASS"
                    if overall
                    else "FAIL"
                ),
            }
        )

    if not results:

        st.warning(
            "No matching test cases were found."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    pass_rate = (
        results_df["Overall"]
        .eq("PASS")
        .mean()
        * 100
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Test Cases",
            len(results_df),
        )

    with c2:
        st.metric(
            "Passed",
            int(
                (
                    results_df["Overall"]
                    == "PASS"
                ).sum()
            ),
        )

    with c3:
        st.metric(
            "Pass Rate",
            f"{pass_rate:.0f}%",
        )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 22. ABOUT
# ============================================================

def show_about():

    page_header(
        "About the Control Tower",
        "Prototype overview and operating model.",
    )

    st.subheader(
        "AI Onboarding Control Tower 🗼⏱️"
    )

    st.write(
        """
        The AI Onboarding Control Tower is a prototype designed
        to improve visibility across the employee onboarding
        lifecycle.
        """
    )

    st.write(
        """
        Instead of requiring HR to manually review multiple
        onboarding steps, the Control Tower combines workflow
        status, deadlines, dependencies, and missing requirements
        into one operational view.
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("Visibility")

        st.write(
            "Provides a centralized view of employee onboarding "
            "status and readiness."
        )

    with c2:

        st.subheader("Risk Detection")

        st.write(
            "Highlights employees with approaching start dates, "
            "missing requirements, or workflow exceptions."
        )

    with c3:

        st.subheader("Action")

        st.write(
            "Identifies the next action and responsible party "
            "so HR can move the workflow forward."
        )

    st.divider()

    st.subheader(
        "Prototype Boundaries"
    )

    st.write(
        """
        This classroom prototype uses fictional employee data
        and simulated HR workflow fields. It demonstrates how
        AI-assisted workflow visibility could support HR
        operations; it is not connected to a production HRIS,
        payroll system, or enterprise identity-management system.
        """
    )


# ============================================================
# 23. APPLICATION DATA PREPARATION
# ============================================================

def prepare_application_data():

    raw_df = (
        st.session_state.employee_df
    )

    if raw_df is None:
        return None

    if raw_df.empty:
        return None

    missing = validate_schema(
        raw_df
    )

    if missing:

        st.error(
            "The uploaded dataset is missing required fields:"
        )

        st.write(missing)

        return None

    as_of_date = st.session_state.get(
        "control_date",
        date.today(),
    )

    analyzed = analyze_rows(
        raw_df,
        as_of_date=as_of_date,
    )

    prioritized = prioritize_df(
        analyzed
    )

    return prioritized


# ============================================================
# 24. MAIN ROUTER
# ============================================================

def main():

    if not st.session_state.authenticated:

        show_login()

        return

    show_sidebar()

    df = prepare_application_data()

    current_view = (
        st.session_state.main_view
    )

    # --------------------------------------------------------
    # Data requirement
    # --------------------------------------------------------

    if (
        df is None
        and current_view
        not in {
            "Data Management",
            "About",
        }
    ):

        st.title(
            "AI Onboarding Control Tower"
        )

        st.info(
            "Your onboarding dataset has not been loaded yet."
        )

        if st.button(
            "Open Data Management",
            type="primary",
        ):

            st.session_state.main_view = (
                "Data Management"
            )

            st.rerun()

        return

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    if current_view == "Overview":

        show_overview(df)

    elif current_view == "Employee Directory":

        show_employee_directory(df)

    elif current_view == "Employee Profile":

        show_employee_profile(df)

    elif current_view == "Risk Monitor":

        show_risk_monitor(df)

    elif current_view == "Analytics":

        show_analytics(df)

    elif current_view == "Communications":

        show_communications(df)

    elif current_view == "Rule Validation":

        show_rule_validation(df)

    elif current_view == "Data Management":

        show_data_management()

    elif current_view == "About":

        show_about()

    else:

        show_overview(df)


# ============================================================
# 25. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
