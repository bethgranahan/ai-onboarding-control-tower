import os
import json
import re
import tempfile
import html
import urllib.parse
import io
import requests
from datetime import date, datetime

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# PROCESS CONFIGURATION
# -----------------------------------------------------------------------------

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
    "Direct Deposit"
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

DERIVED_COLS = [
    "Days to Start",
    "Imported Days to Start",
    "Start Date Parse Status",
    "Core Checklist %",
    "Core Checklist Ready?",
    "Current Stage",
    "Primary Bottleneck",
    "Blocked Downstream Task",
    "Next Action",
    "Responsible Party",
    "Risk Level",
    "Exception Flag",
    "Missing Core Items",
    "Control Tower Explanation",
    "Priority",
    "Priority Reason",
    "Workflow Depth",
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
    "Complete"
]

DEMO_ACTIONS = [
    "Complete all core checklist items",
    "Complete payroll documents only",
    "Create HRIS employee ID + payroll screens",
    "Complete SafeColleges",
    "HR notifies IT",
    "IT completes email/system access",
]


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _norm(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def _is_complete(v):
    return _norm(v).lower() == "complete"


def _yes(v):
    return _norm(v).lower() == "yes"


def _created(v):
    return _norm(v).lower() == "created"


def _coerce_date(series):
    s = series.copy()

    out = pd.to_datetime(s, errors="coerce")

    numeric = pd.to_numeric(s, errors="coerce")

    serial_mask = (
        numeric.notna()
        & numeric.between(20000, 80000)
    )

    if serial_mask.any():
        serial_dates = pd.to_datetime(
            numeric.loc[serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

        out.loc[serial_mask] = serial_dates

    return pd.to_datetime(
        out,
        errors="coerce"
    ).dt.normalize()


def _calculate_days_to_start(start_dates, as_of):
    parsed = _coerce_date(start_dates)

    days = (
        parsed - as_of
    ).dt.days

    return parsed, days.astype("Int64")


def validate_schema(df):
    return [
        c for c in RAW_REQUIRED
        if c not in df.columns
    ]


# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------

def analyze_rows(df, as_of_date=None):

    df = df.copy()

    if as_of_date is None:
        as_of = pd.Timestamp.today().normalize()

    else:

        as_of = pd.to_datetime(
            as_of_date,
            errors="coerce"
        )

        if pd.isna(as_of):
            as_of = pd.Timestamp.today().normalize()

        as_of = as_of.normalize()

    if "Days to Start" in df.columns:

        df["Imported Days to Start"] = (
            pd.to_numeric(
                df["Days to Start"],
                errors="coerce"
            ).astype("Int64")
        )

    df["Start Date"], df["Days to Start"] = (
        _calculate_days_to_start(
            df["Start Date"],
            as_of
        )
    )

    df["Start Date Parse Status"] = np.where(
        df["Start Date"].isna(),
        "Could not parse Start Date",
        "OK"
    )

    outputs = []

    for _, row in df.iterrows():

        core_complete = sum(
            _is_complete(row.get(c))
            for c in CORE_CHECKLIST
        )

        core_pct = (
            core_complete
            / len(CORE_CHECKLIST)
        )

        core_ready = (
            core_complete
            == len(CORE_CHECKLIST)
        )

        missing_core = [
            c for c in CORE_CHECKLIST
            if not _is_complete(row.get(c))
        ]

        payroll_missing = [
            c for c in PAYROLL_PREREQS
            if not _is_complete(row.get(c))
        ]

        hris_created = _created(
            row.get("HRIS Employee ID")
        )

        safe_complete = _is_complete(
            row.get("SafeColleges")
        )

        hr_notified_it = _yes(
            row.get("HR Notified IT")
        )

        it_complete = _is_complete(
            row.get("IT Email/System Access")
        )

        exception = (
            hris_created
            and len(payroll_missing) > 0
        )

        if it_complete:

            stage = "Fully Operational"

        elif exception:

            stage = "Exception / Payroll Risk"

        elif hr_notified_it:

            stage = "IT Provisioning"

        elif (
            hris_created
            and core_ready
            and safe_complete
        ):

            stage = "Ready for IT Notification"

        elif hris_created:

            stage = "Compliance Pending"

        elif core_ready:

            stage = "Ready for HRIS Setup"

        elif (
            core_pct == 0
            and _norm(
                row.get("SafeColleges")
            ).lower()
            in ("", "not started")
        ):

            stage = "Not Started"

        else:

            stage = "Checklist In Progress"

        # ---------------------------------------------------------
        # Workflow logic
        # ---------------------------------------------------------

        if stage == "Fully Operational":

            bottleneck = "None"
            blocked = "None"
            next_action = "No action needed"
            owner = "None"

        elif stage == "Not Started":

            bottleneck = "Checklist not begun"
            blocked = "Onboarding workflow"
            next_action = (
                "New hire begins required checklist items"
            )
            owner = "New Hire"

        elif stage == "Exception / Payroll Risk":

            bottleneck = "Missing payroll information"
            blocked = "Payroll setup / pay accuracy"
            next_action = (
                "Submit missing tax/direct-deposit information; "
                "HR verifies payroll screens"
            )
            owner = "New Hire / HR"

        elif stage == "IT Provisioning":

            bottleneck = "IT access setup"
            blocked = "Operational readiness"
            next_action = (
                "IT completes email/system access"
            )
            owner = "IT"

        elif stage == "Ready for IT Notification":

            bottleneck = "HR-to-IT handoff"
            blocked = "IT email/system access"
            next_action = "HR notifies IT"
            owner = "HR"

        elif stage == "Compliance Pending":

            bottleneck = "SafeColleges"
            blocked = "IT email/system access"
            next_action = (
                "New hire completes SafeColleges"
            )
            owner = "New Hire"

        elif stage == "Ready for HRIS Setup":

            bottleneck = "HRIS setup"
            blocked = "HRIS/payroll setup"
            next_action = (
                "HR creates HRIS employee ID "
                "and payroll screens"
            )
            owner = "HR"

        else:

            if payroll_missing:

                bottleneck = "Payroll prerequisite"
                blocked = "HRIS/payroll setup"
                next_action = (
                    "New hire submits missing payroll documents"
                )
                owner = "New Hire"

            else:

                bottleneck = "Outstanding checklist items"
                blocked = "Remaining onboarding work"
                next_action = (
                    "New hire completes outstanding "
                    "checklist items"
                )
                owner = "New Hire"

        # ---------------------------------------------------------
        # Risk
        # ---------------------------------------------------------

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

        exception_flag = (
            "Payroll Risk Exception"
            if exception
            else ""
        )

        missing_text = (
            ", ".join(missing_core)
            if missing_core
            else "None"
        )

        if stage == "Fully Operational":

            explanation = (
                "All modeled onboarding gates are complete. "
                "The employee is fully operational."
            )

        else:

            explanation = (
                f"{stage}: the primary blocker is "
                f"{bottleneck}. This is preventing "
                f"{blocked} from moving forward. "
                f"Next action: {next_action}. "
                f"Owner: {owner}."
            )

        outputs.append({
            "Days to Start": days,
            "Core Checklist %": core_pct,
            "Core Checklist Ready?": (
                "Ready"
                if core_ready
                else "Not Ready"
            ),
            "Current Stage": stage,
            "Primary Bottleneck": bottleneck,
            "Blocked Downstream Task": blocked,
            "Next Action": next_action,
            "Responsible Party": owner,
            "Risk Level": risk,
            "Exception Flag": exception_flag,
            "Missing Core Items": missing_text,
            "Control Tower Explanation": explanation,
        })

    derived = pd.DataFrame(
        outputs,
        index=df.index
    )

    for c in derived.columns:
        df[c] = derived[c]

    return df


# -----------------------------------------------------------------------------
# PRIORITIZATION
# -----------------------------------------------------------------------------

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


def prioritize_df(df):

    df = df.copy()

    depth = (
        df["Current Stage"]
        .map(WORKFLOW_DEPTH)
        .fillna(0)
        .astype(int)
    )

    df["Workflow Depth"] = depth

    priorities = []
    reasons = []

    for _, row in df.iterrows():

        stage = _norm(
            row.get("Current Stage")
        )

        days = row.get(
            "Days to Start"
        )

        if stage == "Fully Operational":

            priorities.append("Green")

            reasons.append(
                "All modeled onboarding requirements are complete."
            )

        elif (
            not pd.isna(days)
            and days <= 7
        ):

            priorities.append("Red")

            if days < 0:

                reasons.append(
                    "Start date has passed and "
                    "onboarding work is still pending."
                )

            else:

                reasons.append(
                    f"Starts in {int(days)} day(s) "
                    "and onboarding work is still pending."
                )

        else:

            priorities.append("Yellow")

            if pd.isna(days):

                reasons.append(
                    "Pending onboarding work; "
                    "start date is unavailable."
                )

            else:

                reasons.append(
                    f"Pending onboarding work; "
                    f"start date is in {int(days)} day(s)."
                )

    df["Priority"] = priorities
    df["Priority Reason"] = reasons

    df["_priority_sort"] = (
        df["Priority"]
        .map(PRIORITY_ORDER)
        .fillna(9)
    )

    df["_days_sort"] = (
        pd.to_numeric(
            df["Days to Start"],
            errors="coerce"
        )
        .fillna(99999)
    )

    df = df.sort_values(
        by=[
            "_priority_sort",
            "_days_sort",
            "Workflow Depth",
            "Employee Name",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    return df.drop(
        columns=[
            "_priority_sort",
            "_days_sort"
        ],
        errors="ignore"
    )


def _fmt_start_date(v):

    if pd.isna(v):
        return "Unknown"

    try:
        return pd.to_datetime(
            v
        ).strftime("%m/%d/%Y")

    except Exception:
        return str(v)


# -----------------------------------------------------------------------------
# EMAIL / COMMUNICATION HELPERS
# -----------------------------------------------------------------------------

def _directory_email(contact_df, role):

    if (
        contact_df is None
        or len(contact_df) == 0
    ):
        return ""

    try:

        hit = contact_df[
            contact_df["Role"]
            .astype(str)
            .str.strip()
            .str.lower()
            == role.lower()
        ]

        return (
            _norm(hit.iloc[0]["Email"])
            if len(hit)
            else ""
        )

    except Exception:

        return ""


def resolve_recipient(
    row,
    login_email="",
    default_it_email="",
    contact_df=None
):

    owner = _norm(
        row.get("Responsible Party")
    )

    employee_email = _norm(
        row.get("Employee Email")
    )

    supervisor_email = _norm(
        row.get("Supervisor Email")
    )

    directory_hr = _directory_email(
        contact_df,
        "HR"
    )

    directory_it = _directory_email(
        contact_df,
        "IT"
    )

    hr_email = (
        _norm(login_email)
        or directory_hr
    )

    it_email = (
        _norm(default_it_email)
        or directory_it
    )

    to_email = ""
    cc_email = ""
    role = (
        owner
        or "Responsible stakeholder"
    )

    owner_l = owner.lower()

    if owner_l == "new hire":

        to_email = employee_email
        role = "New Hire"

    elif owner_l == "new hire / hr":

        to_email = employee_email
        cc_email = hr_email
        role = "New Hire (HR copied)"

    elif owner_l == "hr":

        to_email = hr_email
        role = "HR"

    elif owner_l == "it":

        to_email = it_email
        role = "IT"

    elif "supervisor" in owner_l:

        to_email = supervisor_email
        role = "Hiring Supervisor"

    return {
        "to": to_email,
        "cc": cc_email,
        "role": role,
        "resolved": bool(to_email),
    }


def suggested_subject(row):

    name = (
        _norm(row.get("Employee Name"))
        or "New Hire"
    )

    stage = _norm(
        row.get("Current Stage")
    )

    blocker = _norm(
        row.get("Primary Bottleneck")
    )

    if stage == "IT Provisioning":

        return (
            f"Onboarding access follow-up — {name}"
        )

    if stage == "Ready for IT Notification":

        return (
            f"Action needed: IT onboarding handoff — {name}"
        )

    if stage == "Exception / Payroll Risk":

        return (
            f"Action needed: payroll onboarding information — {name}"
        )

    return (
        f"Onboarding action needed — {name}: {blocker}"
    )


def deterministic_reminder(
    row,
    tone="Friendly",
    sender_name="",
    recipient_role=""
):

    name = (
        _norm(row.get("Employee Name"))
        or "there"
    )

    owner = _norm(
        row.get("Responsible Party")
    )

    action = _norm(
        row.get("Next Action")
    )

    stage = _norm(
        row.get("Current Stage")
    )

    blocked = _norm(
        row.get("Blocked Downstream Task")
    )

    bottleneck = _norm(
        row.get("Primary Bottleneck")
    )

    days = row.get(
        "Days to Start"
    )

    if stage == "Fully Operational":

        return (
            "No reminder is needed. "
            "This employee is fully operational."
        )

    urgency = ""

    if not pd.isna(days):

        if days < 0:

            urgency = (
                " The employee's start date has already "
                "passed, so this requires immediate follow-up."
            )

        elif days <= 7:

            urgency = (
                f" The employee starts in {int(days)} day(s), "
                "so timely follow-up is important."
            )

    role = (
        recipient_role
        or owner
    )

    sign = (
        f"\n\nThank you,\n{sender_name}"
        if _norm(sender_name)
        else "\n\nThank you"
    )

    if role.startswith("New Hire"):

        body = (
            f"Hi {name},\n\n"
            f"We're following up on your onboarding. "
            f"The next required step is: {action}."
            f"{urgency}\n\n"
            f"Until this is completed, {blocked} is blocked.\n\n"
            "Please complete the item when you can so "
            "the onboarding process can continue."
            f"{sign}"
        )

    elif role == "IT":

        body = (
            "Hello,\n\n"
            f"Please follow up on onboarding access for "
            f"{name}. The current next action is: {action}."
            f"{urgency}\n\n"
            f"Current workflow stage: {stage}.\n"
            f"Blocked downstream work: {blocked}."
            f"{sign}"
        )

    elif role == "HR":

        body = (
            "Hello,\n\n"
            f"Internal onboarding follow-up for {name}: {action}."
            f"{urgency}\n\n"
            f"Primary blocker: {bottleneck}.\n"
            f"Blocked downstream work: {blocked}."
            f"{sign}"
        )

    else:

        body = (
            "Hello,\n\n"
            f"This is an onboarding follow-up for {name}. "
            f"The next action is: {action}."
            f"{urgency}\n\n"
            f"Primary blocker: {bottleneck}.\n"
            f"Blocked downstream work: {blocked}."
            f"{sign}"
        )

    return body


# -----------------------------------------------------------------------------
# GOOGLE SHEETS
# -----------------------------------------------------------------------------

def _extract_sheet_id(url_or_id):

    text = str(
        url_or_id or ""
    ).strip()

    if not text:
        return ""

    m = re.search(
        r"/spreadsheets/d/([a-zA-Z0-9-_]+)",
        text
    )

    return (
        m.group(1)
        if m
        else text
    )


def read_public_google_sheet(
    sheet_url_or_id,
    worksheet_name="Employee_Data"
):

    sheet_id = _extract_sheet_id(
        sheet_url_or_id
    )

    if not sheet_id:

        raise ValueError(
            "Enter a Google Sheet URL or Sheet ID."
        )

    params = urllib.parse.urlencode({
        "tqx": "out:csv",
        "sheet": worksheet_name
    })

    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/gviz/tq?{params}"
    )

    r = requests.get(
        url,
        timeout=30
    )

    r.raise_for_status()

    if "<html" in r.text[:300].lower():

        raise RuntimeError(
            "Google returned a sign-in page. "
            "For this demo, share the Sheet as "
            "'Anyone with the link — Viewer'."
        )

    return pd.read_csv(
        io.StringIO(r.text)
    )


# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Onboarding Control Tower",
    page_icon="🧭",
    layout="wide"
)


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------

if "main_view" not in st.session_state:

    st.session_state["main_view"] = "Home"


if "selected_employee_id" not in st.session_state:

    st.session_state[
        "selected_employee_id"
    ] = None


if "google_sheet_url" not in st.session_state:

    st.session_state[
        "google_sheet_url"
    ] = ""


if "google_sheet_loaded" not in st.session_state:

    st.session_state[
        "google_sheet_loaded"
    ] = False


# -----------------------------------------------------------------------------
# CUSTOM HOME STYLING
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>

    .home-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #111111;
        margin-top: 35px;
        margin-bottom: 10px;
    }

    .home-subtitle {
        text-align: center;
        color: #555555;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .home-card {
        border: 2px solid #111111;
        border-radius: 18px;
        padding: 28px;
        min-height: 220px;
        text-align: center;
        box-shadow: 4px 4px 0px #111111;
        margin-bottom: 12px;
    }

    .home-card-yellow {
        background-color: #F4C542;
    }

    .home-card-black {
        background-color: #111111;
        color: white;
    }

    .home-card h2 {
        margin-bottom: 15px;
    }

    .home-card p {
        font-size: 16px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:

    st.header("Session Settings")

    sender_name = st.text_input(
        "Current HR user name",
        "Demo HR Coordinator"
    )

    sender_email = st.text_input(
        "Current HR user email",
        "hr.onboarding@example.com"
    )

    default_it_email = st.text_input(
        "Default IT contact email",
        "it.access@example.com"
    )

    as_of_text = st.date_input(
        "As-of date",
        value=date.today()
    ).isoformat()

    st.divider()

    st.subheader("Data Source")

    source_type = st.radio(
        "Choose source",
        [
            "Excel upload",
            "Live Google Sheet"
        ]
    )


# -----------------------------------------------------------------------------
# DATA VARIABLES
# -----------------------------------------------------------------------------

raw_df = None
test_df = pd.DataFrame()
contact_df = pd.DataFrame()
source_label = ""


# -----------------------------------------------------------------------------
# EXCEL UPLOAD
# -----------------------------------------------------------------------------

if source_type == "Excel upload":

    uploaded = st.file_uploader(
        "Upload onboarding Excel workbook",
        type=["xlsx", "xls"]
    )

    if uploaded is not None:

        try:

            xls = pd.ExcelFile(
                uploaded
            )

            emp_sheet = (
                "Employee_Data"
                if "Employee_Data"
                in xls.sheet_names
                else xls.sheet_names[0]
            )

            raw_df = pd.read_excel(
                uploaded,
                sheet_name=emp_sheet
            )

            if "Test_Cases" in xls.sheet_names:

                test_df = pd.read_excel(
                    uploaded,
                    sheet_name="Test_Cases"
                )

            if "Contact_Directory" in xls.sheet_names:

                contact_df = pd.read_excel(
                    uploaded,
                    sheet_name="Contact_Directory"
                )

            source_label = (
                f"Excel: {uploaded.name}"
            )

        except Exception as e:

            st.error(
                f"Could not read workbook: {e}"
            )


# -----------------------------------------------------------------------------
# GOOGLE SHEETS
# -----------------------------------------------------------------------------

else:

    st.markdown(
        "### 🔗 Live Google Sheet"
    )

    st.caption(
        "HR can update the Google Sheet directly. "
        "Click Refresh Data after making changes."
    )

    sheet_url = st.text_input(
        "Google Sheet URL",
        value=st.session_state[
            "google_sheet_url"
        ],
        placeholder=(
            "https://docs.google.com/spreadsheets/d/..."
        )
    )

    if sheet_url:

        st.session_state[
            "google_sheet_url"
        ] = sheet_url

    refresh = st.button(
        "🔄 Refresh Data",
        type="primary",
        use_container_width=True
    )

    # Automatically load once when URL is first entered
    should_load = (
        bool(sheet_url)
        and (
            refresh
            or not st.session_state[
                "google_sheet_loaded"
            ]
        )
    )

    if should_load:

        try:

            with st.spinner(
                "Refreshing onboarding data..."
            ):

                raw_df = read_public_google_sheet(
                    sheet_url,
                    "Employee_Data"
                )

                try:

                    test_df = read_public_google_sheet(
                        sheet_url,
                        "Test_Cases"
                    )

                except Exception:

                    test_df = pd.DataFrame()

                try:

                    contact_df = read_public_google_sheet(
                        sheet_url,
                        "Contact_Directory"
                    )

                except Exception:

                    contact_df = pd.DataFrame()

            st.session_state[
                "google_sheet_loaded"
            ] = True

            st.session_state[
                "google_sheet_last_refresh"
            ] = datetime.now().strftime(
                "%m/%d/%Y %I:%M:%S %p"
            )

            source_label = (
                "Live Google Sheet"
            )

            st.success(
                "Data refreshed successfully."
            )

        except Exception as e:

            st.session_state[
                "google_sheet_loaded"
            ] = False

            st.error(
                str(e)
            )

    elif (
        st.session_state[
            "google_sheet_loaded"
        ]
        and sheet_url
    ):

        # Load the existing URL even after navigation
        try:

            raw_df = read_public_google_sheet(
                sheet_url,
                "Employee_Data"
            )

            try:

                test_df = read_public_google_sheet(
                    sheet_url,
                    "Test_Cases"
                )

            except Exception:

                test_df = pd.DataFrame()

            try:

                contact_df = read_public_google_sheet(
                    sheet_url,
                    "Contact_Directory"
                )

            except Exception:

                contact_df = pd.DataFrame()

            source_label = (
                "Live Google Sheet"
            )

        except Exception as e:

            st.error(
                f"Could not load Google Sheet: {e}"
            )

    if (
        st.session_state.get(
            "google_sheet_last_refresh"
        )
    ):

        st.caption(
            "Last refresh: "
            + st.session_state[
                "google_sheet_last_refresh"
            ]
        )


# -----------------------------------------------------------------------------
# HOME PAGE FUNCTION
# -----------------------------------------------------------------------------

def show_home():

    st.markdown(
        """
        <div class="home-title">
            Welcome to the AI Onboarding Control Tower
        </div>

        <div class="home-subtitle">
            Monitor onboarding progress, identify actions needed,
            and quickly search employee status.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(
        3,
        gap="large"
    )

    # ---------------------------------------------------------
    # ONBOARDING STATUS
    # ---------------------------------------------------------

    with c1:

        st.markdown(
            """
            <div class="home-card home-card-yellow">
                <h2>📋 Onboarding Status</h2>
                <p>
                    View an employee's onboarding progress,
                    checklist, current stage, risk level,
                    and readiness.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Onboarding Status",
            use_container_width=True,
            key="home_status"
        ):

            st.session_state[
                "main_view"
            ] = "Employee Detail"

            st.rerun()

    # ---------------------------------------------------------
    # ACTION NEEDED
    # ---------------------------------------------------------

    with c2:

        st.markdown(
            """
            <div class="home-card home-card-black">
                <h2>⚠️ Action Needed</h2>
                <p>
                    See which employees need attention,
                    identify workflow bottlenecks,
                    and prioritize follow-up actions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Action Needed",
            use_container_width=True,
            key="home_action"
        ):

            st.session_state[
                "main_view"
            ] = "Priority Dashboard"

            st.rerun()

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    with c3:

        st.markdown(
            """
            <div class="home-card home-card-yellow">
                <h2>🔎 Search</h2>
                <p>
                    Search employees by Employee ID,
                    Employee Name, or Department.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Search Employee",
            use_container_width=True,
            key="home_search"
        ):

            st.session_state[
                "main_view"
            ] = "Search Employee"

            st.rerun()

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.caption(
        "Group 6 classroom prototype — fictional/mock employee data only"
    )


# -----------------------------------------------------------------------------
# IF NO DATA
# -----------------------------------------------------------------------------

if raw_df is None:

    # Keep the homepage clean even before data is loaded.
    if st.session_state["main_view"] == "Home":

        show_home()

    else:

        st.warning(
            "Please upload the onboarding Excel workbook "
            "or connect a Google Sheet to open this section."
        )

        st.info(
            "Use the Data Source section in the sidebar. "
            "For Google Sheets, make sure the Sheet is shared "
            "as 'Anyone with the link — Viewer'."
        )

    st.stop()


# -----------------------------------------------------------------------------
# VALIDATE DATA
# -----------------------------------------------------------------------------

missing = validate_schema(
    raw_df
)

if missing:

    st.error(
        "Missing required columns: "
        + ", ".join(missing)
    )

    st.stop()


# -----------------------------------------------------------------------------
# ANALYZE DATA
# -----------------------------------------------------------------------------

analyzed = prioritize_df(
    analyze_rows(
        raw_df,
        as_of_text
    )
)


# Employee ID as text
analyzed["Employee ID"] = (
    analyzed["Employee ID"]
    .astype(str)
    .str.strip()
)


# Set first employee if needed
if (
    st.session_state[
        "selected_employee_id"
    ] is None
    and len(analyzed) > 0
):

    st.session_state[
        "selected_employee_id"
    ] = str(
        analyzed.iloc[0][
            "Employee ID"
        ]
    )


# -----------------------------------------------------------------------------
# NAVIGATION
# -----------------------------------------------------------------------------

NAV_OPTIONS = [
    "Home",
    "Search Employee",
    "Employee Statistics",
    "Priority Dashboard",
    "Employee Detail",
    "Communications",
    "Rule Validation",
    "About",
]


# Home should remain visually clean.
# Navigation appears only after the user opens another section.

if st.session_state["main_view"] != "Home":

    view = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="main_view",
        horizontal=True,
        label_visibility="collapsed"
    )

else:

    view = "Home"


# -----------------------------------------------------------------------------
# EMPLOYEE HELPERS
# -----------------------------------------------------------------------------

def _employee_choices(df):

    return [
        f"{r['Employee ID']} — {r['Employee Name']}"
        for _, r in df.iterrows()
    ]


def _choice_index(
    choices,
    employee_id
):

    if not choices:

        return 0

    for i, choice in enumerate(
        choices
    ):

        if (
            choice.split(
                " — ",
                1
            )[0].strip()
            == str(employee_id)
        ):

            return i

    return 0


def _sync_employee_from_widget(
    widget_key
):

    choice = st.session_state.get(
        widget_key
    )

    if choice:

        st.session_state[
            "selected_employee_id"
        ] = (
            choice.split(
                " — ",
                1
            )[0].strip()
        )


# -----------------------------------------------------------------------------
# EMPLOYEE DETAIL
# -----------------------------------------------------------------------------

def show_employee_detail(row):

    employee_name = _norm(
        row.get("Employee Name")
    )

    employee_id = _norm(
        row.get("Employee ID")
    )

    department = _norm(
        row.get("Department")
    )

    employee_email = _norm(
        row.get("Employee Email")
    )

    start_date = _fmt_start_date(
        row.get("Start Date")
    )

    core_pct = row.get(
        "Core Checklist %",
        0
    )

    try:

        core_pct_display = (
            f"{float(core_pct) * 100:.0f}%"
        )

    except Exception:

        core_pct_display = "N/A"

    core_ready = _norm(
        row.get(
            "Core Checklist Ready?"
        )
    )

    risk = _norm(
        row.get("Risk Level")
    )

    priority = _norm(
        row.get("Priority")
    )

    priority_icon = {
        "Red": "🔴",
        "Yellow": "🟡",
        "Green": "🟢"
    }.get(
        priority,
        ""
    )

    st.markdown(
        f"## {employee_name}"
    )

    st.caption(
        f"Employee ID: {employee_id}"
    )

    st.markdown(
        "### Employee Information"
    )

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Employee ID",
        employee_id
    )

    info2.metric(
        "Department",
        department
        if department
        else "N/A"
    )

    info3.metric(
        "Start Date",
        start_date
    )

    info4.metric(
        "Risk Level",
        risk
        if risk
        else "N/A"
    )

    info5, info6, info7, info8 = st.columns(4)

    info5.metric(
        "Employee Email",
        employee_email
        if employee_email
        else "N/A"
    )

    info6.metric(
        "Core Checklist %",
        core_pct_display
    )

    info7.metric(
        "Core Checklist Ready?",
        core_ready
        if core_ready
        else "N/A"
    )

    info8.metric(
        "Priority",
        (
            f"{priority_icon} {priority}"
            if priority
            else "N/A"
        )
    )

    st.divider()

    st.markdown(
        "### Onboarding Status"
    )

    st.markdown(
        f"**Current Stage:** "
        f"{_norm(row.get('Current Stage'))}"
    )

    st.markdown(
        f"**Primary Bottleneck:** "
        f"{_norm(row.get('Primary Bottleneck'))}"
    )

    st.markdown(
        f"**Blocked Downstream Task:** "
        f"{_norm(row.get('Blocked Downstream Task'))}"
    )

    st.markdown(
        f"**Next Action:** "
        f"{_norm(row.get('Next Action'))}"
    )

    st.markdown(
        f"**Responsible Party:** "
        f"{_norm(row.get('Responsible Party'))}"
    )

    explanation = _norm(
        row.get(
            "Control Tower Explanation"
        )
    )

    if explanation:

        st.info(
            explanation
        )

    st.markdown(
        "### Onboarding Checklist"
    )

    checklist_rows = []

    all_checklist_items = (
        CORE_CHECKLIST
        + ["SafeColleges"]
    )

    for item in all_checklist_items:

        status = _norm(
            row.get(item)
        )

        if not status:

            status = "Missing"

        checklist_rows.append({
            "Category": item,
            "Status": status
        })

    st.dataframe(
        pd.DataFrame(
            checklist_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Additional Onboarding System Status"
    )

    system_rows = [
        {
            "Category": "HRIS Employee ID",
            "Status": _norm(
                row.get(
                    "HRIS Employee ID"
                )
            )
        },
        {
            "Category": "Payroll Screens",
            "Status": _norm(
                row.get(
                    "Payroll Screens"
                )
            )
        },
        {
            "Category": "HR Notified IT",
            "Status": _norm(
                row.get(
                    "HR Notified IT"
                )
            )
        },
        {
            "Category": "IT Email/System Access",
            "Status": _norm(
                row.get(
                    "IT Email/System Access"
                )
            )
        },
    ]

    st.dataframe(
        pd.DataFrame(
            system_rows
        ),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------

if view == "Home":

    show_home()


# -----------------------------------------------------------------------------
# SEARCH EMPLOYEE
# -----------------------------------------------------------------------------

elif view == "Search Employee":

    st.subheader(
        "🔎 Search for New Employee Status"
    )

    st.write(
        "Search using Employee ID, Employee Name, or Department."
    )

    st.info(
        "You can use one search field at a time. "
        "Department search will show all employees in that department."
    )

    search_id = st.text_input(
        "Employee ID",
        placeholder="Example: NH001"
    )

    search_name = st.text_input(
        "Employee Name",
        placeholder="Example: Jane Smith"
    )

    departments = sorted(
        [
            d
            for d in
            analyzed["Department"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            if d
        ]
    )

    department_options = [
        "Select a department"
    ] + departments

    search_department = st.selectbox(
        "Department",
        department_options
    )

    filtered = analyzed.copy()

    search_performed = False

    if search_id.strip():

        search_performed = True

        query = (
            search_id
            .strip()
            .lower()
        )

        filtered = filtered[
            filtered["Employee ID"]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        ]

    elif search_name.strip():

        search_performed = True

        query = (
            search_name
            .strip()
            .lower()
        )

        filtered = filtered[
            filtered["Employee Name"]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        ]

    elif (
        search_department
        != "Select a department"
    ):

        search_performed = True

        filtered = filtered[
            filtered["Department"]
            .astype(str)
            .str.strip()
            == search_department
        ]

    if search_performed:

        if filtered.empty:

            st.warning(
                "No employees were found matching your search."
            )

        else:

            st.success(
                f"{len(filtered)} employee(s) found."
            )

            if (
                search_department
                != "Select a department"
                and not search_id.strip()
                and not search_name.strip()
            ):

                st.markdown(
                    f"### New Employees in "
                    f"{search_department}"
                )

                department_display = filtered[
                    [
                        "Employee ID",
                        "Employee Name",
                        "Department",
                        "Start Date",
                        "Core Checklist %",
                        "Core Checklist Ready?",
                        "Risk Level",
                    ]
                ].copy()

                department_display[
                    "Core Checklist %"
                ] = (
                    department_display[
                        "Core Checklist %"
                    ]
                    * 100
                ).round(0).astype(int).astype(str) + "%"

                department_display[
                    "Start Date"
                ] = (
                    department_display[
                        "Start Date"
                    ]
                    .apply(_fmt_start_date)
                )

                st.dataframe(
                    department_display,
                    use_container_width=True,
                    hide_index=True
                )

            employee_choices = (
                _employee_choices(
                    filtered
                )
            )

            if len(employee_choices) == 1:

                selected_choice = (
                    employee_choices[0]
                )

            else:

                selected_choice = st.selectbox(
                    "Select an employee to view details",
                    employee_choices
                )

            selected_id = (
                selected_choice
                .split(
                    " — ",
                    1
                )[0]
                .strip()
            )

            selected_rows = analyzed[
                analyzed["Employee ID"]
                .astype(str)
                == selected_id
            ]

            if not selected_rows.empty:

                selected_row = (
                    selected_rows.iloc[0]
                )

                st.divider()

                show_employee_detail(
                    selected_row
                )

                st.session_state[
                    "selected_employee_id"
                ] = selected_id

    else:

        st.markdown(
            """
            ### Search instructions

            Enter an **Employee ID**, **Employee Name**, or choose a
            **Department** above to view onboarding information.
            """
        )


# -----------------------------------------------------------------------------
# EMPLOYEE STATISTICS
# -----------------------------------------------------------------------------

elif view == "Employee Statistics":

    st.subheader(
        "📊 New Employees' Statistics"
    )

    st.write(
        "Employees are ordered by risk level, "
        "with higher-risk employees displayed first."
    )

    risk_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
        "Complete": 4
    }

    statistics = analyzed.copy()

    statistics["_risk_order"] = (
        statistics["Risk Level"]
        .map(risk_order)
        .fillna(99)
    )

    statistics = statistics.sort_values(
        by=[
            "_risk_order",
            "Employee Name"
        ],
        ascending=[
            True,
            True
        ]
    )

    statistics_display = statistics[
        [
            "Employee Name",
            "Risk Level",
            "Employee Email"
        ]
    ].copy()

    statistics_display[
        "Employee Email"
    ] = (
        statistics_display[
            "Employee Email"
        ].fillna("")
    )

    st.dataframe(
        statistics_display,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### Risk Summary"
    )

    critical_n = int(
        (
            analyzed["Risk Level"]
            == "Critical"
        ).sum()
    )

    high_n = int(
        (
            analyzed["Risk Level"]
            == "High"
        ).sum()
    )

    medium_n = int(
        (
            analyzed["Risk Level"]
            == "Medium"
        ).sum()
    )

    low_n = int(
        (
            analyzed["Risk Level"]
            == "Low"
        ).sum()
    )

    complete_n = int(
        (
            analyzed["Risk Level"]
            == "Complete"
        ).sum()
    )

    s1, s2, s3, s4, s5 = st.columns(5)

    s1.metric(
        "Critical",
        critical_n
    )

    s2.metric(
        "High",
        high_n
    )

    s3.metric(
        "Medium",
        medium_n
    )

    s4.metric(
        "Low",
        low_n
    )

    s5.metric(
        "Complete",
        complete_n
    )


# -----------------------------------------------------------------------------
# PRIORITY DASHBOARD
# -----------------------------------------------------------------------------

elif view == "Priority Dashboard":

    st.subheader(
        "Priority Control Tower"
    )

    st.caption(
        f"Source: {source_label} | "
        f"Days to Start recalculated as of "
        f"{as_of_text}"
    )

    total = len(analyzed)

    red_n = int(
        (
            analyzed["Priority"]
            == "Red"
        ).sum()
    )

    yellow_n = int(
        (
            analyzed["Priority"]
            == "Yellow"
        ).sum()
    )

    green_n = int(
        (
            analyzed["Priority"]
            == "Green"
        ).sum()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Employees",
        total
    )

    m2.metric(
        "🔴 Urgent",
        red_n
    )

    m3.metric(
        "🟡 Pending",
        yellow_n
    )

    m4.metric(
        "🟢 Complete",
        green_n
    )

    st.markdown(
        """
        **Priority logic:**

        🔴 Pending + start date within 7 days/overdue

        🟡 Pending + later start date

        🟢 Fully operational
        """
    )

    display_cols = [
        "Priority",
        "Employee ID",
        "Employee Name",
        "Department",
        "Start Date",
        "Days to Start",
        "Current Stage",
        "Primary Bottleneck",
        "Blocked Downstream Task",
        "Next Action",
        "Responsible Party",
        "Risk Level"
    ]

    disp = analyzed[
        display_cols
    ].copy()

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=560
    )

    # Keep the same chart logic
    stage_counts = (
        analyzed["Current Stage"]
        .value_counts()
        .reindex(
            STAGE_ORDER,
            fill_value=0
        )
        .reset_index()
    )

    stage_counts.columns = [
        "Stage",
        "Employees"
    ]

    fig_stage = px.bar(
        stage_counts,
        x="Stage",
        y="Employees",
        title="Employees by Current Stage"
    )

    fig_stage.update_layout(
        xaxis_tickangle=-30,
        height=430
    )

    risk_counts = (
        analyzed["Risk Level"]
        .value_counts()
        .reindex(
            RISK_ORDER,
            fill_value=0
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk",
        "Employees"
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk",
        y="Employees",
        title="Employees by Risk Level"
    )

    fig_risk.update_layout(
        height=380
    )

    p1, p2 = st.columns(2)

    with p1:

        st.plotly_chart(
            fig_stage,
            use_container_width=True
        )

    with p2:

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# EMPLOYEE DETAIL
# -----------------------------------------------------------------------------

elif view == "Employee Detail":

    st.subheader(
        "Employee Control Tower"
    )

    choices = _employee_choices(
        analyzed
    )

    detail_index = _choice_index(
        choices,
        st.session_state.get(
            "selected_employee_id"
        )
    )

    selected = st.selectbox(
        "Select employee",
        choices,
        index=detail_index,
        key="detail_emp",
        on_change=_sync_employee_from_widget,
        args=("detail_emp",)
    )

    emp_id = (
        selected
        .split(
            " — ",
            1
        )[0]
    )

    st.session_state[
        "selected_employee_id"
    ] = emp_id

    selected_rows = analyzed[
        analyzed["Employee ID"]
        .astype(str)
        == emp_id
    ]

    if not selected_rows.empty:

        row = selected_rows.iloc[0]

        show_employee_detail(
            row
        )

        st.divider()

        b1, b2 = st.columns(
            [1, 3]
        )

        with b1:

            if st.button(
                "✉️ Draft follow-up email",
                type="primary",
                use_container_width=True
            ):

                st.session_state[
                    "main_view"
                ] = "Communications"

                st.rerun()

        with b2:

            st.caption(
                "The Communications page will open with this employee already selected."
            )


# -----------------------------------------------------------------------------
# COMMUNICATIONS
# -----------------------------------------------------------------------------

elif view == "Communications":

    st.subheader(
        "Draft Follow-Up Email"
    )

    choices = _employee_choices(
        analyzed
    )

    comm_index = _choice_index(
        choices,
        st.session_state.get(
            "selected_employee_id"
        )
    )

    selected2 = st.selectbox(
        "Employee for communication",
        choices,
        index=comm_index,
        key="comm_emp",
        on_change=_sync_employee_from_widget,
        args=("comm_emp",)
    )

    emp_id2 = (
        selected2
        .split(
            " — ",
            1
        )[0]
    )

    st.session_state[
        "selected_employee_id"
    ] = emp_id2

    row2 = analyzed[
        analyzed["Employee ID"]
        .astype(str)
        == emp_id2
    ].iloc[0]

    route = resolve_recipient(
        row2,
        login_email=sender_email,
        default_it_email=default_it_email,
        contact_df=contact_df
    )

    st.caption(
        f"Suggested recipient role: {route['role']}. "
        "Recipient comes from workflow ownership + contact data, "
        "not AI guessing."
    )

    to_email = st.text_input(
        "To",
        value=route["to"],
        key=f"to_{emp_id2}"
    )

    cc_email = st.text_input(
        "CC",
        value=route["cc"],
        key=f"cc_{emp_id2}"
    )

    subject = st.text_input(
        "Subject",
        value=suggested_subject(row2),
        key=f"subject_{emp_id2}"
    )

    body_default = deterministic_reminder(
        row2,
        sender_name=sender_name,
        recipient_role=route["role"]
    )

    body = st.text_area(
        "Email body",
        value=body_default,
        height=260,
        key=f"body_{emp_id2}"
    )

    if to_email:

        params = {
            "view": "cm",
            "fs": "1",
            "to": to_email,
            "su": subject,
            "body": body
        }

        if cc_email:

            params["cc"] = cc_email

        gmail_url = (
            "https://mail.google.com/mail/?"
            +
            urllib.parse.urlencode(
                params,
                quote_via=urllib.parse.quote
            )
        )

        st.link_button(
            "Open Draft in Gmail",
            gmail_url
        )

    else:

        st.warning(
            "No email address could be resolved. "
            "Confirm or enter the recipient manually."
        )

    if st.button(
        "Send Email (Demo)"
    ):

        st.success(
            "Demo email logged as sent. "
            "No external email was transmitted."
        )


# -----------------------------------------------------------------------------
# RULE VALIDATION
# -----------------------------------------------------------------------------

elif view == "Rule Validation":

    st.subheader(
        "Rule Validation"
    )

    if (
        test_df is None
        or test_df.empty
    ):

        st.markdown(
            """
            ### Rule Validation

            No `Test_Cases` sheet was found.
            The app can still run normally.
            """
        )

    else:

        rows = []

        for _, t in test_df.iterrows():

            scenario = _norm(
                t.get("Scenario")
            )

            expected_stage = _norm(
                t.get("Expected Stage")
            )

            expected_b = _norm(
                t.get("Expected Bottleneck")
            )

            expected_a = _norm(
                t.get("Expected Next Action")
            )

            candidates = pd.DataFrame()

            if "Scenario Tag" in analyzed.columns:

                candidates = analyzed[
                    analyzed[
                        "Scenario Tag"
                    ]
                    .astype(str)
                    .str.contains(
                        scenario,
                        regex=False,
                        na=False
                    )
                ]

            if candidates.empty:

                use_rows = _norm(
                    t.get(
                        "Use These Employee Rows"
                    )
                )

                if "-" in use_rows:

                    start, end = (
                        use_rows.split(
                            "-",
                            1
                        )
                    )

                    try:

                        lo = int(
                            start.replace(
                                "NH",
                                ""
                            )
                        )

                        hi = int(
                            end.replace(
                                "NH",
                                ""
                            )
                        )

                        ids = (
                            analyzed[
                                "Employee ID"
                            ]
                            .astype(str)
                        )

                        nums = (
                            ids
                            .str.replace(
                                "NH",
                                "",
                                regex=False
                            )
                            .astype(int)
                        )

                        candidates = analyzed[
                            (
                                nums >= lo
                            )
                            & (
                                nums <= hi
                            )
                        ]

                    except Exception:

                        candidates = pd.DataFrame()

            if candidates.empty:

                rows.append([
                    scenario,
                    "NOT FOUND",
                    expected_stage,
                    "—",
                    "—",
                    "FAIL"
                ])

                continue

            actual_stage = (
                candidates[
                    "Current Stage"
                ]
                .mode()
                .iat[0]
            )

            actual_b = (
                candidates[
                    "Primary Bottleneck"
                ]
                .mode()
                .iat[0]
            )

            actual_a = (
                candidates[
                    "Next Action"
                ]
                .mode()
                .iat[0]
            )

            passed = (
                actual_stage
                == expected_stage
                and actual_b
                == expected_b
                and actual_a
                == expected_a
            )

            rows.append([
                scenario,
                actual_stage,
                expected_stage,
                actual_b,
                actual_a,
                (
                    "PASS"
                    if passed
                    else "FAIL"
                )
            ])

        result = pd.DataFrame(
            rows,
            columns=[
                "Scenario",
                "Actual Stage",
                "Expected Stage",
                "Actual Bottleneck",
                "Actual Next Action",
                "Result"
            ]
        )

        n_pass = int(
            (
                result["Result"]
                == "PASS"
            ).sum()
        )

        st.markdown(
            f"### Rule Validation\n"
            f"**{n_pass} of {len(result)} "
            f"supplied scenarios passed.**"
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )


# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------

elif view == "About":

    st.subheader(
        "About the Prototype"
    )

    st.markdown(
        """
        **AI Onboarding Control Tower**

        This classroom prototype helps HR monitor the employee
        onboarding process from accepted offer through operational readiness.

        ### Core Functions

        - Search employees by Employee ID
        - Search employees by Employee Name
        - Search employees by Department
        - View employee department and start date
        - Identify missing onboarding checklist items
        - Calculate Core Checklist %
        - Determine whether the Core Checklist is ready
        - Identify onboarding risk
        - Identify workflow bottlenecks
        - Identify the responsible party
        - Show the next required action
        - Prioritize employees by risk and urgency
        - Draft follow-up communication

        ### Data Sources

        The prototype supports:

        - Excel workbook upload
        - Live Google Sheet connection
        - Manual Google Sheet refresh

        HR can update the Google Sheet and then use
        **Refresh Data** to bring the latest information
        into the Control Tower.

        ### Prototype Boundary

        This application is designed for classroom demonstration
        using fictional/mock employee data.

        Do not publish real employee HR information in a public
        Streamlit application.
        """
    )
