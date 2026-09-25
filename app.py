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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------

DEFAULT_SESSION = {
    "authenticated": False,
    "main_view": "Overview",
    "selected_employee_id": None,
    "google_sheet_url": "",
    "google_sheet_loaded": False,
    "google_sheet_last_refresh": None,
    "login_attempted": False,
}

for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:

        st.session_state[key] = value


# -----------------------------------------------------------------------------
# DEMO LOGIN
# -----------------------------------------------------------------------------

DEMO_USERNAME = "HR.Admin"
DEMO_PASSWORD = "HR2026"


# -----------------------------------------------------------------------------
# PROFESSIONAL UI STYLING
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL
       ========================================================= */

    .stApp {
        background: #F6F7F9;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #111111;
        letter-spacing: -0.02em;
    }

    p {
        color: #555555;
    }


    /* =========================================================
       SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #242424;
    }

    section[data-testid="stSidebar"] * {
        color: #F5F5F5;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #D7D7D7;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #F4C542;
    }

    .sidebar-brand {
        padding: 10px 5px 25px 5px;
    }

    .sidebar-brand-title {
        font-size: 19px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.04em;
    }

    .sidebar-brand-accent {
        color: #F4C542;
    }

    .sidebar-brand-subtitle {
        font-size: 11px;
        color: #8E8E8E;
        margin-top: 5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-user {
        background: #1B1B1B;
        border: 1px solid #303030;
        border-radius: 12px;
        padding: 14px;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .sidebar-user-name {
        font-weight: 700;
        font-size: 14px;
        color: #FFFFFF;
    }

    .sidebar-user-role {
        font-size: 11px;
        color: #999999;
        margin-top: 3px;
    }


    /* =========================================================
       TOP HEADER
       ========================================================= */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }

    .top-header-title {
        font-size: 23px;
        font-weight: 800;
        color: #111111;
    }

    .top-header-subtitle {
        color: #777777;
        font-size: 13px;
        margin-top: 3px;
    }

    .top-header-user {
        text-align: right;
    }

    .top-header-user-name {
        font-size: 13px;
        font-weight: 700;
        color: #222222;
    }

    .top-header-user-role {
        font-size: 11px;
        color: #888888;
    }


    /* =========================================================
       LOGIN
       ========================================================= */

    .login-page {
        min-height: 82vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .login-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        padding: 42px;
        box-shadow: 0 18px 55px rgba(0,0,0,0.10);
    }

    .login-logo {
        width: 54px;
        height: 54px;
        border-radius: 14px;
        background: #111111;
        color: #F4C542;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .login-title {
        font-size: 31px;
        font-weight: 850;
        color: #111111;
        margin-bottom: 5px;
    }

    .login-subtitle {
        color: #777777;
        font-size: 14px;
        margin-bottom: 28px;
    }

    .login-footer {
        color: #999999;
        font-size: 11px;
        text-align: center;
        margin-top: 18px;
    }


    /* =========================================================
       HERO
       ========================================================= */

    .hero {
        background: #111111;
        border-radius: 20px;
        padding: 30px 34px;
        color: white;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: -60px;
        top: -100px;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        border: 45px solid #F4C542;
        opacity: 0.12;
    }

    .hero-eyebrow {
        color: #F4C542;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .hero-title {
        font-size: 31px;
        font-weight: 850;
        color: #FFFFFF;
        margin-bottom: 7px;
    }

    .hero-text {
        color: #BDBDBD;
        font-size: 14px;
        max-width: 680px;
    }


    /* =========================================================
       KPI CARDS
       ========================================================= */

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E4E7EB;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 2px 9px rgba(0,0,0,0.025);
    }

    .kpi-label {
        color: #777777;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .kpi-value {
        color: #111111;
        font-size: 32px;
        font-weight: 850;
        margin-top: 8px;
    }

    .kpi-description {
        color: #999999;
        font-size: 11px;
        margin-top: 3px;
    }

    .kpi-yellow {
        border-top: 4px solid #F4C542;
    }

    .kpi-red {
        border-top: 4px solid #D64545;
    }

    .kpi-green {
        border-top: 4px solid #3A9B6B;
    }

    .kpi-blue {
        border-top: 4px solid #4D73BE;
    }


    /* =========================================================
       SECTION HEADERS
       ========================================================= */

    .section-title {
        font-size: 19px;
        font-weight: 800;
        color: #111111;
        margin-top: 28px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #888888;
        margin-bottom: 15px;
    }


    /* =========================================================
       INSIGHT CARDS
       ========================================================= */

    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 15px;
        padding: 18px;
        min-height: 145px;
    }

    .insight-label {
        font-size: 11px;
        color: #8A8A8A;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.06em;
    }

    .insight-value {
        font-size: 17px;
        color: #111111;
        font-weight: 750;
        margin-top: 8px;
        line-height: 1.35;
    }

    .insight-small {
        font-size: 12px;
        color: #888888;
        margin-top: 7px;
    }


    /* =========================================================
       EMPLOYEE CARDS
       ========================================================= */

    .employee-card {
        background: #FFFFFF;
        border: 1px solid #E4E7EB;
        border-radius: 16px;
        padding: 19px;
        margin-bottom: 10px;
    }

    .employee-name {
        font-size: 17px;
        font-weight: 800;
        color: #111111;
    }

    .employee-meta {
        font-size: 12px;
        color: #888888;
        margin-top: 3px;
    }

    .status-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-green {
        background: #E8F5EE;
        color: #287A50;
    }

    .badge-yellow {
        background: #FFF5CF;
        color: #866A00;
    }

    .badge-red {
        background: #FBE9E9;
        color: #A62E2E;
    }


    /* =========================================================
       PROFILE
       ========================================================= */

    .profile-header {
        background: #FFFFFF;
        border: 1px solid #E4E7EB;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .profile-avatar {
        width: 58px;
        height: 58px;
        border-radius: 16px;
        background: #111111;
        color: #F4C542;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
        font-weight: 850;
    }

    .profile-name {
        font-size: 25px;
        font-weight: 850;
        color: #111111;
    }

    .profile-meta {
        color: #777777;
        font-size: 13px;
        margin-top: 3px;
    }


    /* =========================================================
       DATA SOURCE
       ========================================================= */

    .data-status {
        background: #FFFFFF;
        border: 1px solid #E4E7EB;
        border-radius: 14px;
        padding: 15px;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    .data-status-title {
        font-weight: 750;
        color: #222222;
        font-size: 13px;
    }

    .data-status-text {
        color: #777777;
        font-size: 11px;
        margin-top: 3px;
    }


    /* =========================================================
       NAV BUTTONS
       ========================================================= */

    .nav-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 15px;
        padding: 19px;
        min-height: 130px;
    }

    .nav-card-title {
        font-weight: 800;
        font-size: 16px;
        color: #111111;
    }

    .nav-card-text {
        color: #777777;
        font-size: 12px;
        margin-top: 6px;
        line-height: 1.45;
    }


    /* =========================================================
       TABLE
       ========================================================= */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }


    /* =========================================================
       BUTTONS
       ========================================================= */

    .stButton > button,
    .stLinkButton > a {
        border-radius: 9px !important;
        font-weight: 700 !important;
        border: 1px solid #D8D8D8 !important;
        min-height: 40px !important;
    }

    .stButton > button[kind="primary"] {
        background: #111111 !important;
        color: #FFFFFF !important;
        border-color: #111111 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #F4C542 !important;
        color: #111111 !important;
        border-color: #F4C542 !important;
    }


    /* =========================================================
       PROGRESS
       ========================================================= */

    .progress-container {
        background: #E9EAEC;
        height: 9px;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 9px;
        margin-bottom: 7px;
    }

    .progress-bar {
        background: #F4C542;
        height: 100%;
        border-radius: 20px;
    }

    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #777777;
    }


    /* =========================================================
       FOOTER
       ========================================================= */

    .app-footer {
        border-top: 1px solid #E4E7EB;
        margin-top: 45px;
        padding-top: 15px;
        color: #999999;
        font-size: 10px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# LOGIN PAGE
# -----------------------------------------------------------------------------

def show_login():

    st.markdown(
        "<div class='login-page'>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns(
        [1, 1.15, 1]
    )

    with center:

        st.markdown(
            """
            <div class="login-card">

                <div class="login-logo">◈</div>

                <div class="login-title">
                    Welcome back
                </div>

                <div class="login-subtitle">
                    Sign in to the AI Onboarding Control Tower
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        username = st.text_input(
            "Username",
            placeholder="Enter your HR username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        remember = st.checkbox(
            "Remember this session"
        )

        if st.button(
            "Sign in",
            type="primary",
            use_container_width=True
        ):

            st.session_state[
                "login_attempted"
            ] = True

            if (
                username.strip()
                == DEMO_USERNAME
                and password
                == DEMO_PASSWORD
            ):

                st.session_state[
                    "authenticated"
                ] = True

                st.session_state[
                    "main_view"
                ] = "Overview"

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

        st.markdown(
            """
            <div class="login-footer">
                Internal HR workspace · Classroom prototype
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# STOP HERE IF NOT AUTHENTICATED
# -----------------------------------------------------------------------------

if not st.session_state["authenticated"]:

    show_login()
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-title">
                AI ONBOARDING
                <span class="sidebar-brand-accent">
                    CONTROL TOWER
                </span>
            </div>

            <div class="sidebar-brand-subtitle">
                HR Operations Platform
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#777777;
            font-size:10px;
            font-weight:800;
            letter-spacing:.10em;
            margin-bottom:8px;
        ">
            WORKSPACE
        </div>
        """,
        unsafe_allow_html=True
    )

    NAV_OPTIONS = [
        "Overview",
        "Employee Directory",
        "Risk Monitor",
        "Analytics",
        "Communications",
        "Rule Validation",
        "Data Management",
        "About",
    ]

    nav_icons = {
        "Overview": "⌂",
        "Employee Directory": "♙",
        "Risk Monitor": "!",
        "Analytics": "◫",
        "Communications": "✉",
        "Rule Validation": "✓",
        "Data Management": "↻",
        "About": "i",
    }

    current_view = st.session_state["main_view"]

    if current_view not in NAV_OPTIONS:

        current_view = "Overview"

    selected_view = st.radio(
        "Navigation",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(current_view),
        format_func=lambda x: (
            f"{nav_icons.get(x, '')}   {x}"
        ),
        key="sidebar_navigation",
        label_visibility="collapsed"
    )

    st.session_state[
        "main_view"
    ] = selected_view

    st.divider()

    st.markdown(
        """
        <div style="
            color:#777777;
            font-size:10px;
            font-weight:800;
            letter-spacing:.10em;
            margin-bottom:8px;
        ">
            ACCOUNT
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-user">

            <div class="sidebar-user-name">
                HR Admin
            </div>

            <div class="sidebar-user-role">
                HR Administrator
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Sign out",
        use_container_width=True
    ):

        st.session_state[
            "authenticated"
        ] = False

        st.session_state[
            "main_view"
        ] = "Overview"

        st.rerun()


# -----------------------------------------------------------------------------
# TOP HEADER
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="top-header">

        <div>
            <div class="top-header-title">
                AI Onboarding Control Tower
            </div>

            <div class="top-header-subtitle">
                Employee lifecycle visibility and onboarding workflow intelligence
            </div>
        </div>

        <div class="top-header-user">

            <div class="top-header-user-name">
                HR Admin
            </div>

            <div class="top-header-user-role">
                HR Administrator
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# DATA VARIABLES
# -----------------------------------------------------------------------------

raw_df = None
test_df = pd.DataFrame()
contact_df = pd.DataFrame()
source_label = ""


# -----------------------------------------------------------------------------
# DATA SOURCE PAGE / LOADER
# -----------------------------------------------------------------------------

def load_excel(uploaded_file):

    xls = pd.ExcelFile(
        uploaded_file
    )

    emp_sheet = (
        "Employee_Data"
        if "Employee_Data"
        in xls.sheet_names
        else xls.sheet_names[0]
    )

    employees = pd.read_excel(
        uploaded_file,
        sheet_name=emp_sheet
    )

    tests = pd.DataFrame()
    contacts = pd.DataFrame()

    if "Test_Cases" in xls.sheet_names:

        tests = pd.read_excel(
            uploaded_file,
            sheet_name="Test_Cases"
        )

    if "Contact_Directory" in xls.sheet_names:

        contacts = pd.read_excel(
            uploaded_file,
            sheet_name="Contact_Directory"
        )

    return employees, tests, contacts


def load_google_data(sheet_url):

    employees = read_public_google_sheet(
        sheet_url,
        "Employee_Data"
    )

    try:

        tests = read_public_google_sheet(
            sheet_url,
            "Test_Cases"
        )

    except Exception:

        tests = pd.DataFrame()

    try:

        contacts = read_public_google_sheet(
            sheet_url,
            "Contact_Directory"
        )

    except Exception:

        contacts = pd.DataFrame()

    return employees, tests, contacts


# -----------------------------------------------------------------------------
# DATA MANAGEMENT PAGE
# -----------------------------------------------------------------------------

def show_data_management():

    st.markdown(
        """
        <div class="section-title">
            Data Management
        </div>

        <div class="section-subtitle">
            Manage the employee data that powers the Control Tower.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="data-status">

            <div class="data-status-title">
                ◉ Data connection
            </div>

            <div class="data-status-text">
                Connect an Excel workbook for a snapshot or a Google Sheet
                for continuously updateable demo data.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    source = st.radio(
        "Choose data source",
        [
            "Excel upload",
            "Live Google Sheet"
        ],
        horizontal=True
    )

    if source == "Excel upload":

        st.markdown(
            "### Upload employee data"
        )

        uploaded = st.file_uploader(
            "Choose Excel workbook",
            type=["xlsx", "xls"],
            key="professional_excel_upload"
        )

        st.caption(
            "Expected worksheets: Employee_Data, Test_Cases, Contact_Directory."
        )

        if uploaded is not None:

            try:

                employees, tests, contacts = load_excel(
                    uploaded
                )

                st.session_state[
                    "uploaded_employees"
                ] = employees

                st.session_state[
                    "uploaded_tests"
                ] = tests

                st.session_state[
                    "uploaded_contacts"
                ] = contacts

                st.success(
                    f"{len(employees)} employee records loaded from {uploaded.name}."
                )

            except Exception as e:

                st.error(
                    f"Could not read workbook: {e}"
                )

    else:

        st.markdown(
            "### Connect a live Google Sheet"
        )

        st.info(
            "For the classroom prototype, the Google Sheet must be shared "
            "as 'Anyone with the link — Viewer'."
        )

        sheet_url = st.text_input(
            "Google Sheet URL",
            value=st.session_state[
                "google_sheet_url"
            ],
            placeholder="Paste your Google Sheets URL here"
        )

        if sheet_url:

            st.session_state[
                "google_sheet_url"
            ] = sheet_url

        if st.button(
            "↻ Refresh live data",
            type="primary",
            use_container_width=False
        ):

            if not sheet_url.strip():

                st.warning(
                    "Please enter a Google Sheet URL first."
                )

            else:

                try:

                    with st.spinner(
                        "Refreshing employee data..."
                    ):

                        employees, tests, contacts = (
                            load_google_data(
                                sheet_url
                            )
                        )

                    st.session_state[
                        "google_employees"
                    ] = employees

                    st.session_state[
                        "google_tests"
                    ] = tests

                    st.session_state[
                        "google_contacts"
                    ] = contacts

                    st.session_state[
                        "google_sheet_loaded"
                    ] = True

                    st.session_state[
                        "google_sheet_last_refresh"
                    ] = datetime.now().strftime(
                        "%m/%d/%Y %I:%M:%S %p"
                    )

                    st.success(
                        f"Live data refreshed — {len(employees)} employees loaded."
                    )

                except Exception as e:

                    st.error(
                        f"Could not load Google Sheet: {e}"
                    )

        if st.session_state.get(
            "google_sheet_last_refresh"
        ):

            st.caption(
                "Last refreshed: "
                + st.session_state[
                    "google_sheet_last_refresh"
                ]
            )

    st.divider()

    st.markdown(
        "### Data source guide"
    )

    g1, g2 = st.columns(2)

    with g1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-label">
                    Excel
                </div>

                <div class="insight-value">
                    Snapshot upload
                </div>

                <div class="insight-small">
                    Upload a new workbook whenever HR wants
                    to replace the current data.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with g2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-label">
                    Google Sheets
                </div>

                <div class="insight-value">
                    Updateable source
                </div>

                <div class="insight-small">
                    HR can edit the connected Sheet and refresh
                    the Control Tower to pull the latest values.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# LOAD STORED DATA
# -----------------------------------------------------------------------------

if "google_employees" in st.session_state:

    raw_df = st.session_state[
        "google_employees"
    ]

    test_df = st.session_state.get(
        "google_tests",
        pd.DataFrame()
    )

    contact_df = st.session_state.get(
        "google_contacts",
        pd.DataFrame()
    )

    source_label = "Live Google Sheet"

elif "uploaded_employees" in st.session_state:

    raw_df = st.session_state[
        "uploaded_employees"
    ]

    test_df = st.session_state.get(
        "uploaded_tests",
        pd.DataFrame()
    )

    contact_df = st.session_state.get(
        "uploaded_contacts",
        pd.DataFrame()
    )

    source_label = "Excel Upload"


# -----------------------------------------------------------------------------
# DATA MANAGEMENT CAN BE USED WITHOUT DATA
# -----------------------------------------------------------------------------

if st.session_state["main_view"] == "Data Management":

    show_data_management()


# -----------------------------------------------------------------------------
# IF NO DATA
# -----------------------------------------------------------------------------

if raw_df is None:

    if st.session_state["main_view"] != "Data Management":

        st.markdown(
            """
            <div class="hero">

                <div class="hero-eyebrow">
                    Control Tower
                </div>

                <div class="hero-title">
                    Connect your onboarding data
                </div>

                <div class="hero-text">
                    Upload the HR onboarding workbook or connect a
                    Google Sheet to activate employee tracking,
                    risk monitoring, and workflow analytics.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Open Data Management from the sidebar to connect your employee data."
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
# AS-OF DATE
# -----------------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="
            color:#777777;
            font-size:10px;
            font-weight:800;
            letter-spacing:.10em;
            margin-top:10px;
            margin-bottom:7px;
        ">
            CONTROL DATE
        </div>
        """,
        unsafe_allow_html=True
    )

    as_of_date = st.date_input(
        "As-of date",
        value=date.today(),
        label_visibility="collapsed"
    )

    as_of_text = as_of_date.isoformat()


# -----------------------------------------------------------------------------
# ANALYZE DATA
# -----------------------------------------------------------------------------

analyzed = prioritize_df(
    analyze_rows(
        raw_df,
        as_of_text
    )
)

analyzed["Employee ID"] = (
    analyzed["Employee ID"]
    .astype(str)
    .str.strip()
)


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
# GLOBAL METRICS
# -----------------------------------------------------------------------------

total_employees = len(analyzed)

ready_count = int(
    (
        analyzed["Risk Level"]
        == "Complete"
    ).sum()
)

critical_count = int(
    (
        analyzed["Risk Level"]
        == "Critical"
    ).sum()
)

high_count = int(
    (
        analyzed["Risk Level"]
        == "High"
    ).sum()
)

pending_count = int(
    (
        analyzed["Risk Level"]
        .isin(["Medium", "Low"])
    ).sum()
)

avg_checklist = (
    analyzed["Core Checklist %"]
    .mean()
    if len(analyzed)
    else 0
)

if pd.isna(avg_checklist):

    avg_checklist = 0


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def employee_choices(df):

    return [
        f"{r['Employee ID']} — {r['Employee Name']}"
        for _, r in df.iterrows()
    ]


def get_employee(employee_id):

    rows = analyzed[
        analyzed["Employee ID"]
        .astype(str)
        == str(employee_id)
    ]

    if rows.empty:

        return None

    return rows.iloc[0]


def risk_badge(risk):

    risk = _norm(risk)

    if risk in ["Critical", "High"]:

        return "badge-red"

    if risk in ["Medium", "Low"]:

        return "badge-yellow"

    return "badge-green"


def priority_badge(priority):

    if priority == "Red":

        return "badge-red"

    if priority == "Yellow":

        return "badge-yellow"

    return "badge-green"


def show_progress(percent):

    try:

        pct = float(percent) * 100

    except Exception:

        pct = 0

    pct = max(
        0,
        min(
            100,
            pct
        )
    )

    st.markdown(
        f"""
        <div class="progress-label">

            <span>Core onboarding readiness</span>
            <strong>{pct:.0f}%</strong>

        </div>

        <div class="progress-container">

            <div class="progress-bar"
                 style="width:{pct:.0f}%;">
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------

def show_overview():

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-eyebrow">
                HR Operations Intelligence
            </div>

            <div class="hero-title">
                Welcome back, HR Admin
            </div>

            <div class="hero-text">
                Monitor onboarding readiness, identify workflow
                bottlenecks, and focus attention where action is needed.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Onboarding Overview"
    )

    st.caption(
        f"Live control tower status · {source_label}"
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(
            f"""
            <div class="kpi-card kpi-yellow">

                <div class="kpi-label">
                    Total Employees
                </div>

                <div class="kpi-value">
                    {total_employees}
                </div>

                <div class="kpi-description">
                    Active onboarding records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:

        st.markdown(
            f"""
            <div class="kpi-card kpi-green">

                <div class="kpi-label">
                    Ready
                </div>

                <div class="kpi-value">
                    {ready_count}
                </div>

                <div class="kpi-description">
                    Fully operational
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:

        st.markdown(
            f"""
            <div class="kpi-card kpi-red">

                <div class="kpi-label">
                    Critical
                </div>

                <div class="kpi-value">
                    {critical_count}
                </div>

                <div class="kpi-description">
                    Requires immediate attention
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:

        st.markdown(
            f"""
            <div class="kpi-card kpi-blue">

                <div class="kpi-label">
                    Avg. Readiness
                </div>

                <div class="kpi-value">
                    {avg_checklist * 100:.0f}%
                </div>

                <div class="kpi-description">
                    Across core checklist
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='section-title'>AI Onboarding Insights</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-subtitle'>Signals generated from the current onboarding workflow data.</div>",
        unsafe_allow_html=True
    )

    missing_items = []

    for _, row in analyzed.iterrows():

        text = _norm(
            row.get(
                "Missing Core Items"
            )
        )

        if text and text != "None":

            missing_items.extend(
                [
                    x.strip()
                    for x in text.split(",")
                    if x.strip()
                ]
            )

    if missing_items:

        most_common = (
            pd.Series(
                missing_items
            )
            .value_counts()
            .index[0]
        )

    else:

        most_common = "None"

    pending_rows = analyzed[
        analyzed["Risk Level"]
        .isin(
            [
                "Critical",
                "High"
            ]
        )
    ]

    if len(pending_rows):

        top_department = (
            pending_rows[
                "Department"
            ]
            .astype(str)
            .value_counts()
            .index[0]
        )

    else:

        top_department = "No high-risk department"

    i1, i2, i3 = st.columns(3)

    with i1:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-label">
                    Most common missing item
                </div>

                <div class="insight-value">
                    {html.escape(most_common)}
                </div>

                <div class="insight-small">
                    Based on incomplete core checklist items
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-label">
                    Highest attention area
                </div>

                <div class="insight-value">
                    {html.escape(str(top_department))}
                </div>

                <div class="insight-small">
                    Department with the most Critical / High records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-label">
                    Employees requiring attention
                </div>

                <div class="insight-value">
                    {critical_count + high_count}
                </div>

                <div class="insight-small">
                    Critical + High risk onboarding records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='section-title'>Priority Queue</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-subtitle'>Employees currently requiring the most attention.</div>",
        unsafe_allow_html=True
    )

    priority_preview = analyzed[
        analyzed["Priority"]
        .isin(
            [
                "Red",
                "Yellow"
            ]
        )
    ].head(5)

    if priority_preview.empty:

        st.success(
            "No pending employees are currently in the priority queue."
        )

    else:

        for _, row in priority_preview.iterrows():

            name = _norm(
                row.get("Employee Name")
            )

            department = _norm(
                row.get("Department")
            )

            risk = _norm(
                row.get("Risk Level")
            )

            stage = _norm(
                row.get("Current Stage")
            )

            priority = _norm(
                row.get("Priority")
            )

            st.markdown(
                f"""
                <div class="employee-card">

                    <div class="employee-name">
                        {html.escape(name)}
                    </div>

                    <div class="employee-meta">
                        {html.escape(department)}
                        &nbsp; · &nbsp;
                        {html.escape(stage)}
                    </div>

                    <div style="margin-top:10px;">

                        <span class="status-badge {priority_badge(priority)}">
                            {html.escape(priority)}
                        </span>

                        <span style="margin-left:7px;"
                              class="status-badge {risk_badge(risk)}">
                            {html.escape(risk)}
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="app-footer">
            AI Onboarding Control Tower · Classroom prototype · Fictional/mock employee data
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# EMPLOYEE DIRECTORY
# -----------------------------------------------------------------------------

def show_employee_directory():

    st.markdown(
        """
        <div class="section-title">
            Employee Directory
        </div>

        <div class="section-subtitle">
            Search and explore onboarding status across the employee population.
        </div>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Search employees",
        placeholder="Search by employee name, ID, or department..."
    )

    filtered = analyzed.copy()

    if search.strip():

        q = search.strip().lower()

        mask = (
            filtered["Employee ID"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)
            |
            filtered["Employee Name"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)
            |
            filtered["Department"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)
        )

        filtered = filtered[mask]

    st.caption(
        f"{len(filtered)} employee record(s)"
    )

    if filtered.empty:

        st.warning(
            "No employees match your search."
        )

        return

    for _, row in filtered.iterrows():

        employee_id = _norm(
            row.get("Employee ID")
        )

        name = _norm(
            row.get("Employee Name")
        )

        department = _norm(
            row.get("Department")
        )

        risk = _norm(
            row.get("Risk Level")
        )

        stage = _norm(
            row.get("Current Stage")
        )

        pct = row.get(
            "Core Checklist %",
            0
        )

        try:

            pct_display = (
                float(pct) * 100
            )

        except Exception:

            pct_display = 0

        c1, c2, c3 = st.columns(
            [4, 2, 1]
        )

        with c1:

            st.markdown(
                f"""
                <div class="employee-card">

                    <div class="employee-name">
                        {html.escape(name)}
                    </div>

                    <div class="employee-meta">
                        ID: {html.escape(employee_id)}
                        &nbsp; · &nbsp;
                        {html.escape(department)}
                    </div>

                    <div style="margin-top:9px;">
                        <span class="status-badge {risk_badge(risk)}">
                            {html.escape(risk)}
                        </span>
                    </div>

                    <div style="margin-top:10px;">

                        <div class="progress-label">

                            <span>
                                Onboarding readiness
                            </span>

                            <strong>
                                {pct_display:.0f}%
                            </strong>

                        </div>

                        <div class="progress-container">

                            <div class="progress-bar"
                                 style="width:{pct_display:.0f}%;">
                            </div>

                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div style="
                    background:#FFFFFF;
                    border:1px solid #E4E7EB;
                    border-radius:16px;
                    padding:20px;
                    min-height:100px;
                ">

                    <div class="insight-label">
                        Current stage
                    </div>

                    <div style="
                        margin-top:8px;
                        font-size:13px;
                        font-weight:700;
                        color:#222222;
                    ">
                        {html.escape(stage)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            if st.button(
                "View",
                key=f"view_{employee_id}",
                use_container_width=True
            ):

                st.session_state[
                    "selected_employee_id"
                ] = employee_id

                st.session_state[
                    "main_view"
                ] = "Employee Profile"

                st.rerun()

        st.markdown(
            "<div style='height:5px'></div>",
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# EMPLOYEE PROFILE
# -----------------------------------------------------------------------------

def show_employee_profile():

    employee_id = st.session_state.get(
        "selected_employee_id"
    )

    row = get_employee(
        employee_id
    )

    if row is None:

        st.warning(
            "Select an employee from the Employee Directory first."
        )

        return

    name = _norm(
        row.get("Employee Name")
    )

    department = _norm(
        row.get("Department")
    )

    employee_email = _norm(
        row.get("Employee Email")
    )

    risk = _norm(
        row.get("Risk Level")
    )

    stage = _norm(
        row.get("Current Stage")
    )

    core_pct = row.get(
        "Core Checklist %",
        0
    )

    try:

        pct_display = float(
            core_pct
        ) * 100

    except Exception:

        pct_display = 0

    initials = "".join(
        [
            x[0].upper()
            for x in name.split()
            if x
        ]
    )[:2]

    st.markdown(
        f"""
        <div class="profile-header">

            <div style="
                display:flex;
                align-items:center;
                gap:17px;
            ">

                <div class="profile-avatar">
                    {html.escape(initials)}
                </div>

                <div>

                    <div class="profile-name">
                        {html.escape(name)}
                    </div>

                    <div class="profile-meta">
                        {html.escape(department)}
                        &nbsp; · &nbsp;
                        Employee ID {_norm(row.get("Employee ID"))}
                    </div>

                    <div style="margin-top:8px;">

                        <span class="status-badge {risk_badge(risk)}">
                            {html.escape(risk)} Risk
                        </span>

                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Start Date",
        _fmt_start_date(
            row.get("Start Date")
        )
    )

    p2.metric(
        "Readiness",
        f"{pct_display:.0f}%"
    )

    p3.metric(
        "Current Stage",
        stage
    )

    p4.metric(
        "Email",
        employee_email or "N/A"
    )

    st.markdown(
        "### Onboarding readiness"
    )

    show_progress(
        core_pct
    )

    st.markdown(
        "### Workflow intelligence"
    )

    w1, w2 = st.columns(2)

    with w1:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-label">
                    Primary bottleneck
                </div>

                <div class="insight-value">
                    {html.escape(_norm(row.get("Primary Bottleneck")))}
                </div>

                <div class="insight-small">
                    Current process constraint
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with w2:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-label">
                    Next action
                </div>

                <div class="insight-value">
                    {html.escape(_norm(row.get("Next Action")))}
                </div>

                <div class="insight-small">
                    Responsible party:
                    {html.escape(_norm(row.get("Responsible Party")))}
                </div>

            </div>
            """,
            unsafe_allow_html=True
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
        "### Onboarding checklist"
    )

    checklist_items = (
        CORE_CHECKLIST
        + ["SafeColleges"]
    )

    checklist_rows = []

    for item in checklist_items:

        status = _norm(
            row.get(item)
        )

        if not status:

            status = "Missing"

        checklist_rows.append(
            {
                "Requirement": item,
                "Status": status
            }
        )

    checklist_df = pd.DataFrame(
        checklist_rows
    )

    st.dataframe(
        checklist_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### System readiness"
    )

    system_df = pd.DataFrame(
        [
            {
                "System Gate": "HRIS Employee ID",
                "Status": _norm(
                    row.get(
                        "HRIS Employee ID"
                    )
                )
            },
            {
                "System Gate": "Payroll Screens",
                "Status": _norm(
                    row.get(
                        "Payroll Screens"
                    )
                )
            },
            {
                "System Gate": "HR Notified IT",
                "Status": _norm(
                    row.get(
                        "HR Notified IT"
                    )
                )
            },
            {
                "System Gate": "IT Email/System Access",
                "Status": _norm(
                    row.get(
                        "IT Email/System Access"
                    )
                )
            }
        ]
    )

    st.dataframe(
        system_df,
        use_container_width=True,
        hide_index=True
    )

    b1, b2 = st.columns(2)

    with b1:

        if st.button(
            "← Back to Employee Directory",
            use_container_width=True
        ):

            st.session_state[
                "main_view"
            ] = "Employee Directory"

            st.rerun()

    with b2:

        if st.button(
            "Draft Follow-Up Email →",
            type="primary",
            use_container_width=True
        ):

            st.session_state[
                "main_view"
            ] = "Communications"

            st.rerun()


# -----------------------------------------------------------------------------
# RISK MONITOR
# -----------------------------------------------------------------------------

def show_risk_monitor():

    st.markdown(
        """
        <div class="section-title">
            Risk Monitor
        </div>

        <div class="section-subtitle">
            Prioritized view of employees requiring onboarding attention.
        </div>
        """,
        unsafe_allow_html=True
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Critical",
        critical_count
    )

    r2.metric(
        "High",
        high_count
    )

    r3.metric(
        "Pending",
        pending_count
    )

    r4.metric(
        "Complete",
        ready_count
    )

    risk_df = analyzed[
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
            "Risk Level"
        ]
    ].copy()

    st.markdown(
        "### Priority queue"
    )

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True,
        height=500
    )


# -----------------------------------------------------------------------------
# ANALYTICS
# -----------------------------------------------------------------------------

def show_analytics():

    st.markdown(
        """
        <div class="section-title">
            Analytics
        </div>

        <div class="section-subtitle">
            Understand how employees are moving through the onboarding workflow.
        </div>
        """,
        unsafe_allow_html=True
    )

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
        title="Employees by onboarding stage"
    )

    fig_stage.update_layout(
        height=420,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=90
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
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
        title="Employees by risk level"
    )

    fig_risk.update_layout(
        height=420,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=40
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    a1, a2 = st.columns(2)

    with a1:

        st.plotly_chart(
            fig_stage,
            use_container_width=True
        )

    with a2:

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    st.markdown(
        "### Department overview"
    )

    department_df = (
        analyzed.groupby(
            "Department",
            dropna=False
        )
        .agg(
            Employees=(
                "Employee ID",
                "count"
            ),
            Avg_Readiness=(
                "Core Checklist %",
                "mean"
            )
        )
        .reset_index()
    )

    department_df[
        "Avg_Readiness"
    ] = (
        department_df[
            "Avg_Readiness"
        ] * 100
    ).round(0)

    department_df = department_df.rename(
        columns={
            "Avg_Readiness":
            "Average Readiness %"
        }
    )

    st.dataframe(
        department_df,
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------------------------------------------------
# COMMUNICATIONS
# -----------------------------------------------------------------------------

def show_communications():

    st.markdown(
        """
        <div class="section-title">
            Communications
        </div>

        <div class="section-subtitle">
            Create a workflow-based follow-up message for an employee or stakeholder.
        </div>
        """,
        unsafe_allow_html=True
    )

    choices = employee_choices(
        analyzed
    )

    if not choices:

        st.warning(
            "No employees are available."
        )

        return

    selected = st.selectbox(
        "Employee",
        choices
    )

    employee_id = (
        selected
        .split(
            " — ",
            1
        )[0]
    )

    st.session_state[
        "selected_employee_id"
    ] = employee_id

    row = get_employee(
        employee_id
    )

    route = resolve_recipient(
        row,
        login_email="hr.onboarding@example.com",
        default_it_email="it.access@example.com",
        contact_df=contact_df
    )

    to_email = st.text_input(
        "To",
        value=route["to"]
    )

    cc_email = st.text_input(
        "CC",
        value=route["cc"]
    )

    subject = st.text_input(
        "Subject",
        value=suggested_subject(
            row
        )
    )

    body = st.text_area(
        "Message",
        value=deterministic_reminder(
            row,
            sender_name="HR Admin",
            recipient_role=route["role"]
        ),
        height=260
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

    if st.button(
        "Log Demo Communication",
        type="primary"
    ):

        st.success(
            "Demo communication logged. No external email was transmitted."
        )


# -----------------------------------------------------------------------------
# RULE VALIDATION
# -----------------------------------------------------------------------------

def show_rule_validation():

    st.markdown(
        """
        <div class="section-title">
            Rule Validation
        </div>

        <div class="section-subtitle">
            Test whether the workflow logic produces the expected stage,
            bottleneck, and next action.
        </div>
        """,
        unsafe_allow_html=True
    )

    if test_df.empty:

        st.info(
            "No Test_Cases worksheet was found."
        )

        return

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

        use_rows = _norm(
            t.get(
                "Use These Employee Rows"
            )
        )

        if "-" in use_rows:

            try:

                start, end = (
                    use_rows.split(
                        "-",
                        1
                    )
                )

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
                    &
                    (
                        nums <= hi
                    )
                ]

            except Exception:

                candidates = pd.DataFrame()

        if candidates.empty:

            rows.append(
                [
                    scenario,
                    "NOT FOUND",
                    expected_stage,
                    "—",
                    "—",
                    "FAIL"
                ]
            )

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

        rows.append(
            [
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
            ]
        )

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

    passed_count = int(
        (
            result["Result"]
            == "PASS"
        ).sum()
    )

    st.metric(
        "Validation pass rate",
        f"{passed_count}/{len(result)}"
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------

def show_about():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                About the prototype
            </div>

            <div class="hero-title">
                AI Onboarding Control Tower
            </div>

            <div class="hero-text">
                A classroom prototype designed to improve visibility
                across the employee onboarding lifecycle.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-label">
                    Visibility
                </div>

                <div class="insight-value">
                    One view of onboarding status
                </div>

                <div class="insight-small">
                    Track readiness, workflow stage, and missing requirements.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with a2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-label">
                    Risk
                </div>

                <div class="insight-value">
                    Surface employees needing attention
                </div>

                <div class="insight-small">
                    Prioritize based on onboarding status and timing.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with a3:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-label">
                    Action
                </div>

                <div class="insight-value">
                    Connect blockers to next steps
                </div>

                <div class="insight-small">
                    Identify the responsible stakeholder and next action.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "### Prototype boundaries"
    )

    st.write(
        """
        This application is designed for classroom demonstration
        using fictional/mock employee data. It is not intended to
        replace a production HR information system or serve as a
        secure authentication platform.
        """
    )


# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------

view = st.session_state[
    "main_view"
]


if view == "Overview":

    show_overview()

elif view == "Employee Directory":

    show_employee_directory()

elif view == "Employee Profile":

    show_employee_profile()

elif view == "Risk Monitor":

    show_risk_monitor()

elif view == "Analytics":

    show_analytics()

elif view == "Communications":

    show_communications()

elif view == "Rule Validation":

    show_rule_validation()

elif view == "Data Management":

    show_data_management()

elif view == "About":

    show_about()
