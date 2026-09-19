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

# -----------------------------------------------------------------------------
# PROCESS CONFIGURATION — grounded in the supplied project rules
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

PAYROLL_PREREQS = ["Federal W-4", "Iowa W-4", "Direct Deposit"]

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

RISK_ORDER = ["Critical", "High", "Medium", "Low", "Complete"]

DEMO_ACTIONS = [
    "Complete all core checklist items",
    "Complete payroll documents only",
    "Create HRIS employee ID + payroll screens",
    "Complete SafeColleges",
    "HR notifies IT",
    "IT completes email/system access",
]


def _norm(v):
    """Normalize values for robust business-rule comparisons."""
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
    """
    Robustly parse Start Date values from Excel.
    Handles:
      - true Excel date cells / pandas timestamps
      - ISO or US-style date strings
      - raw Excel serial numbers such as 46307
      - serial numbers stored as text
    """
    s = series.copy()

    # Start with ordinary datetime parsing.
    out = pd.to_datetime(s, errors="coerce")

    # Then explicitly handle Excel serial numbers.
    numeric = pd.to_numeric(s, errors="coerce")
    serial_mask = numeric.notna() & numeric.between(20000, 80000)

    if serial_mask.any():
        serial_dates = pd.to_datetime(
            numeric.loc[serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        out.loc[serial_mask] = serial_dates

    return pd.to_datetime(out, errors="coerce").dt.normalize()


def _calculate_days_to_start(start_dates, as_of):
    """
    Always derive Days to Start from Start Date.
    The workbook's existing Days to Start column is treated as display/reference
    data only and is never trusted for the workflow logic.
    """
    parsed = _coerce_date(start_dates)
    days = (parsed - as_of).dt.days
    return parsed, days.astype("Int64")


def validate_schema(df):
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    return missing


def analyze_rows(df, as_of_date=None):
    """
    Recalculate process stage and controls from the raw onboarding fields.
    Existing derived spreadsheet columns are intentionally overwritten so the app
    can demonstrate the logic instead of simply displaying precomputed answers.
    """
    df = df.copy()
    if as_of_date is None:
        as_of = pd.Timestamp.today().normalize()
    else:
        as_of = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(as_of):
            as_of = pd.Timestamp.today().normalize()
        as_of = as_of.normalize()

    # Preserve the source value for audit/debugging if the workbook already
    # contains a Days to Start field, then recalculate it from Start Date.
    if "Days to Start" in df.columns:
        df["Imported Days to Start"] = pd.to_numeric(df["Days to Start"], errors="coerce").astype("Int64")

    df["Start Date"], df["Days to Start"] = _calculate_days_to_start(df["Start Date"], as_of)

    # Date parse visibility: if this appears, the source Start Date could not be parsed.
    df["Start Date Parse Status"] = np.where(
        df["Start Date"].isna(),
        "Could not parse Start Date",
        "OK"
    )

    outputs = []
    for _, row in df.iterrows():
        core_complete = sum(_is_complete(row.get(c)) for c in CORE_CHECKLIST)
        core_pct = core_complete / len(CORE_CHECKLIST)
        core_ready = core_complete == len(CORE_CHECKLIST)
        missing_core = [c for c in CORE_CHECKLIST if not _is_complete(row.get(c))]
        payroll_missing = [c for c in PAYROLL_PREREQS if not _is_complete(row.get(c))]

        hris_created = _created(row.get("HRIS Employee ID"))
        safe_complete = _is_complete(row.get("SafeColleges"))
        hr_notified_it = _yes(row.get("HR Notified IT"))
        it_complete = _is_complete(row.get("IT Email/System Access"))

        # Rule 3: early HRIS creation with missing payroll prerequisites is an exception.
        exception = hris_created and len(payroll_missing) > 0

        # Stage logic mirrors the supplied workbook/test cases.
        if it_complete:
            stage = "Fully Operational"
        elif exception:
            stage = "Exception / Payroll Risk"
        elif hr_notified_it:
            stage = "IT Provisioning"
        elif hris_created and core_ready and safe_complete:
            stage = "Ready for IT Notification"
        elif hris_created:
            stage = "Compliance Pending"
        elif core_ready:
            stage = "Ready for HRIS Setup"
        elif core_pct == 0 and _norm(row.get("SafeColleges")).lower() in ("", "not started"):
            stage = "Not Started"
        else:
            stage = "Checklist In Progress"

        # Rule-based blocker, dependent work, next action, and owner.
        if stage == "Fully Operational":
            bottleneck = "None"
            blocked = "None"
            next_action = "No action needed"
            owner = "None"
        elif stage == "Not Started":
            bottleneck = "Checklist not begun"
            blocked = "Onboarding workflow"
            next_action = "New hire begins required checklist items"
            owner = "New Hire"
        elif stage == "Exception / Payroll Risk":
            bottleneck = "Missing payroll information"
            blocked = "Payroll setup / pay accuracy"
            next_action = "Submit missing tax/direct-deposit information; HR verifies payroll screens"
            owner = "New Hire / HR"
        elif stage == "IT Provisioning":
            bottleneck = "IT access setup"
            blocked = "Operational readiness"
            next_action = "IT completes email/system access"
            owner = "IT"
        elif stage == "Ready for IT Notification":
            bottleneck = "HR-to-IT handoff"
            blocked = "IT email/system access"
            next_action = "HR notifies IT"
            owner = "HR"
        elif stage == "Compliance Pending":
            bottleneck = "SafeColleges"
            blocked = "IT email/system access"
            next_action = "New hire completes SafeColleges"
            owner = "New Hire"
        elif stage == "Ready for HRIS Setup":
            bottleneck = "HRIS setup"
            blocked = "HRIS/payroll setup"
            next_action = "HR creates HRIS employee ID and payroll screens"
            owner = "HR"
        else:
            if payroll_missing:
                bottleneck = "Payroll prerequisite"
                blocked = "HRIS/payroll setup"
                next_action = "New hire submits missing payroll documents"
                owner = "New Hire"
            else:
                bottleneck = "Outstanding checklist items"
                blocked = "Remaining onboarding work"
                next_action = "New hire completes outstanding checklist items"
                owner = "New Hire"

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

        exception_flag = "Payroll Risk Exception" if exception else ""
        missing_text = ", ".join(missing_core) if missing_core else "None"

        if stage == "Fully Operational":
            explanation = "All modeled onboarding gates are complete. The employee is fully operational."
        else:
            explanation = (
                f"{stage}: the primary blocker is {bottleneck}. "
                f"This is preventing {blocked} from moving forward. "
                f"Next action: {next_action}. Owner: {owner}."
            )

        outputs.append({
            "Days to Start": days,
            "Core Checklist %": core_pct,
            "Core Checklist Ready?": "Ready" if core_ready else "Not Ready",
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

    derived = pd.DataFrame(outputs, index=df.index)
    for c in derived.columns:
        df[c] = derived[c]
    return df


# -----------------------------------------------------------------------------
# CONTROL-TOWER PRIORITIZATION
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

PRIORITY_ORDER = {"Red": 0, "Yellow": 1, "Green": 2}

def prioritize_df(df):
    """
    Sort unresolved employees first by start-date urgency, then by how far
    downstream the current blocker occurs. Fully operational rows are placed last.
    """
    df = df.copy()
    depth = df["Current Stage"].map(WORKFLOW_DEPTH).fillna(0).astype(int)
    df["Workflow Depth"] = depth

    priorities = []
    reasons = []
    for _, row in df.iterrows():
        stage = _norm(row.get("Current Stage"))
        days = row.get("Days to Start")
        if stage == "Fully Operational":
            priorities.append("Green")
            reasons.append("All modeled onboarding requirements are complete.")
        elif not pd.isna(days) and days <= 7:
            priorities.append("Red")
            if days < 0:
                reasons.append("Start date has passed and onboarding work is still pending.")
            else:
                reasons.append(f"Starts in {int(days)} day(s) and onboarding work is still pending.")
        else:
            priorities.append("Yellow")
            if pd.isna(days):
                reasons.append("Pending onboarding work; start date is unavailable.")
            else:
                reasons.append(f"Pending onboarding work; start date is in {int(days)} day(s).")

    df["Priority"] = priorities
    df["Priority Reason"] = reasons

    df["_priority_sort"] = df["Priority"].map(PRIORITY_ORDER).fillna(9)
    df["_days_sort"] = pd.to_numeric(df["Days to Start"], errors="coerce").fillna(99999)
    # Red first, then Yellow, then Green; within each group earlier start dates first.
    # For equal urgency/date, a more downstream blocker is surfaced first.
    df = df.sort_values(
        by=["_priority_sort", "_days_sort", "Workflow Depth", "Employee Name"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return df.drop(columns=["_priority_sort", "_days_sort"], errors="ignore")


def _fmt_start_date(v):
    if pd.isna(v):
        return "Unknown"
    try:
        return pd.to_datetime(v).strftime("%m/%d/%Y")
    except Exception:
        return str(v)


def priority_table_html(df):
    """Render a compact color-coded control-tower list for the Gradio dashboard."""
    if df is None or len(df) == 0:
        return "<p>No employees to display.</p>"

    colors = {
        "Red": ("#FDE2E2", "#9B1C1C", "🔴"),
        "Yellow": ("#FFF4CC", "#7A5A00", "🟡"),
        "Green": ("#E3F6E8", "#146C2E", "🟢"),
    }
    cols = [
        "Priority", "Employee ID", "Employee Name", "Department", "Start Date",
        "Days to Start", "Current Stage", "Primary Bottleneck",
        "Blocked Downstream Task", "Next Action", "Responsible Party"
    ]

    out = ["""
    <style>
      .ct-wrap {overflow-x:auto; border:1px solid #ddd; border-radius:8px;}
      table.ct {border-collapse:collapse; width:100%; font-size:13px;}
      table.ct th {background:#1F4E78; color:white; padding:8px; text-align:left; position:sticky; top:0;}
      table.ct td {padding:8px; border-top:1px solid rgba(0,0,0,.08); vertical-align:top;}
      .legend {margin:6px 0 12px 0; font-size:13px;}
      .badge {font-weight:700; white-space:nowrap;}
    </style>
    <div class='legend'>
      🔴 Urgent: pending + starts within 7 days &nbsp;&nbsp;
      🟡 Pending: later start date &nbsp;&nbsp;
      🟢 Complete: fully operational
    </div>
    <div class='ct-wrap'><table class='ct'><thead><tr>
    """]
    for c in cols:
        out.append(f"<th>{html.escape(c)}</th>")
    out.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        p = _norm(row.get("Priority")) or "Yellow"
        bg, fg, icon = colors.get(p, colors["Yellow"])
        out.append(f"<tr style='background:{bg}; color:{fg};'>")
        for c in cols:
            if c == "Priority":
                val = f"{icon} {p}"
                out.append(f"<td><span class='badge'>{html.escape(val)}</span></td>")
            elif c == "Start Date":
                out.append(f"<td>{html.escape(_fmt_start_date(row.get(c)))}</td>")
            else:
                v = row.get(c)
                if pd.isna(v):
                    v = ""
                elif c == "Days to Start":
                    v = int(v)
                out.append(f"<td>{html.escape(str(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _directory_email(contact_df, role):
    if contact_df is None or len(contact_df) == 0:
        return ""
    try:
        hit = contact_df[contact_df["Role"].astype(str).str.strip().str.lower() == role.lower()]
        return _norm(hit.iloc[0]["Email"]) if len(hit) else ""
    except Exception:
        return ""


def resolve_recipient(row, login_email="", default_it_email="", contact_df=None):
    """
    Routing is deterministic: workflow owner -> contact source.
    AI drafts wording but never invents an email address.
    """
    owner = _norm(row.get("Responsible Party"))
    employee_email = _norm(row.get("Employee Email"))
    supervisor_email = _norm(row.get("Supervisor Email"))
    directory_hr = _directory_email(contact_df, "HR")
    directory_it = _directory_email(contact_df, "IT")

    hr_email = _norm(login_email) or directory_hr
    it_email = _norm(default_it_email) or directory_it

    to_email, cc_email, role = "", "", owner or "Responsible stakeholder"

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
    name = _norm(row.get("Employee Name")) or "New Hire"
    stage = _norm(row.get("Current Stage"))
    blocker = _norm(row.get("Primary Bottleneck"))
    if stage == "IT Provisioning":
        return f"Onboarding access follow-up — {name}"
    if stage == "Ready for IT Notification":
        return f"Action needed: IT onboarding handoff — {name}"
    if stage == "Exception / Payroll Risk":
        return f"Action needed: payroll onboarding information — {name}"
    return f"Onboarding action needed — {name}: {blocker}"


def gmail_compose_html(to_email, cc_email, subject, body):
    if not _norm(to_email):
        return "<div style='padding:10px;background:#FFF4CC;border-radius:6px;'>Recipient email is not available. Enter/confirm an address before opening Gmail.</div>"
    params = {
        "view": "cm",
        "fs": "1",
        "to": to_email,
        "su": subject,
        "body": body,
    }
    if _norm(cc_email):
        params["cc"] = cc_email
    url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return (
        "<a href='" + html.escape(url, quote=True) + "' target='_blank' "
        "style='display:inline-block;padding:10px 16px;background:#1F6FEB;color:white;"
        "text-decoration:none;border-radius:6px;font-weight:700;'>Open Draft in Gmail</a>"
        "<div style='margin-top:6px;font-size:12px;color:#555;'>"
        "Gmail will send from whichever Google account is currently signed in. Review the draft before sending."
        "</div>"
    )

def make_summary(df):
    total = len(df)
    operational = int((df["Current Stage"] == "Fully Operational").sum())
    high_critical = int(df["Risk Level"].isin(["High", "Critical"]).sum())
    exceptions = int((df["Exception Flag"] == "Payroll Risk Exception").sum())
    compliance = int((df["Current Stage"] == "Compliance Pending").sum())
    ready_it = int((df["Current Stage"] == "Ready for IT Notification").sum())

    unresolved = df[df["Primary Bottleneck"] != "None"]
    if len(unresolved):
        top_b = unresolved["Primary Bottleneck"].value_counts().idxmax()
        top_n = int(unresolved["Primary Bottleneck"].value_counts().iloc[0])
        top_text = f"**Most common current blocker:** {top_b} ({top_n} employees)"
    else:
        top_text = "**Most common current blocker:** None"

    return f"""
### Control Tower Summary
| Metric | Count |
|---|---:|
| Total employees | **{total}** |
| Fully operational | **{operational}** |
| High/Critical risk | **{high_critical}** |
| Compliance pending | **{compliance}** |
| Ready for IT notification | **{ready_it}** |
| Payroll-risk exceptions | **{exceptions}** |

{top_text}
"""


def make_plots(df):
    stage_counts = (
        df["Current Stage"].value_counts().reindex(STAGE_ORDER, fill_value=0).reset_index()
    )
    stage_counts.columns = ["Stage", "Employees"]
    fig_stage = px.bar(stage_counts, x="Stage", y="Employees", title="Employees by Current Stage")
    fig_stage.update_layout(xaxis_tickangle=-30, height=430)

    risk_counts = (
        df["Risk Level"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    )
    risk_counts.columns = ["Risk", "Employees"]
    fig_risk = px.bar(risk_counts, x="Risk", y="Employees", title="Employees by Risk Level")
    fig_risk.update_layout(height=380)
    return fig_stage, fig_risk


def export_analysis(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()
    dashboard = pd.DataFrame({
        "Metric": [
            "Total Employees",
            "Fully Operational",
            "High/Critical Risk",
            "Compliance Pending",
            "Ready for IT Notification",
            "Payroll Risk Exceptions",
        ],
        "Value": [
            len(df),
            int((df["Current Stage"] == "Fully Operational").sum()),
            int(df["Risk Level"].isin(["High", "Critical"]).sum()),
            int((df["Current Stage"] == "Compliance Pending").sum()),
            int((df["Current Stage"] == "Ready for IT Notification").sum()),
            int((df["Exception Flag"] == "Payroll Risk Exception").sum()),
        ],
    })
    with pd.ExcelWriter(tmp.name, engine="xlsxwriter", datetime_format="mm/dd/yyyy") as writer:
        df.to_excel(writer, sheet_name="Employee_Data_Analyzed", index=False)
        dashboard.to_excel(writer, sheet_name="Dashboard_Summary", index=False)
    return tmp.name


def validate_test_cases(analyzed_df, test_df):
    if test_df is None or test_df.empty:
        return "### Rule Validation\nNo `Test_Cases` sheet was found. The app can still run normally.", pd.DataFrame()

    rows = []
    for _, t in test_df.iterrows():
        scenario = _norm(t.get("Scenario"))
        expected_stage = _norm(t.get("Expected Stage"))
        expected_b = _norm(t.get("Expected Bottleneck"))
        expected_a = _norm(t.get("Expected Next Action"))

        # Prefer Scenario Tag matching because it remains valid even if row order changes.
        candidates = analyzed_df[analyzed_df.get("Scenario Tag", "").astype(str).str.contains(scenario, regex=False, na=False)] if "Scenario Tag" in analyzed_df.columns else pd.DataFrame()
        if candidates.empty:
            # Fall back to the textual employee range if the scenario tag is unavailable.
            use_rows = _norm(t.get("Use These Employee Rows"))
            if "-" in use_rows:
                start, end = use_rows.split("-", 1)
                ids = analyzed_df["Employee ID"].astype(str)
                # Mock IDs are NH###, so numeric comparison is safe for the supplied data.
                try:
                    lo, hi = int(start.replace("NH", "")), int(end.replace("NH", ""))
                    nums = ids.str.replace("NH", "", regex=False).astype(int)
                    candidates = analyzed_df[(nums >= lo) & (nums <= hi)]
                except Exception:
                    candidates = pd.DataFrame()

        if candidates.empty:
            rows.append([scenario, "NOT FOUND", expected_stage, "—", "—", "FAIL"])
            continue

        actual_stage = candidates["Current Stage"].mode().iat[0]
        actual_b = candidates["Primary Bottleneck"].mode().iat[0]
        actual_a = candidates["Next Action"].mode().iat[0]
        passed = actual_stage == expected_stage and actual_b == expected_b and actual_a == expected_a
        rows.append([scenario, actual_stage, expected_stage, actual_b, actual_a, "PASS" if passed else "FAIL"])

    result = pd.DataFrame(rows, columns=[
        "Scenario", "Actual Stage", "Expected Stage", "Actual Bottleneck", "Actual Next Action", "Result"
    ])
    n_pass = int((result["Result"] == "PASS").sum())
    return f"### Rule Validation\n**{n_pass} of {len(result)} supplied scenarios passed.**", result

# -----------------------------------------------------------------------------
# COMMUNICATION + OPTIONAL LLM HELPERS
# -----------------------------------------------------------------------------

def deterministic_reminder(row, tone="Friendly", sender_name="", recipient_role=""):
    name = _norm(row.get("Employee Name")) or "there"
    owner = _norm(row.get("Responsible Party"))
    action = _norm(row.get("Next Action"))
    stage = _norm(row.get("Current Stage"))
    blocked = _norm(row.get("Blocked Downstream Task"))
    bottleneck = _norm(row.get("Primary Bottleneck"))
    days = row.get("Days to Start")

    if stage == "Fully Operational":
        return "No reminder is needed. This employee is fully operational."

    urgency = ""
    if not pd.isna(days):
        if days < 0:
            urgency = " The employee's start date has already passed, so this requires immediate follow-up."
        elif days <= 7:
            urgency = f" The employee starts in {int(days)} day(s), so timely follow-up is important."

    role = recipient_role or owner
    sign = f"\n\nThank you,\n{sender_name}" if _norm(sender_name) else "\n\nThank you"

    if role.startswith("New Hire"):
        opening = f"Hi {name},"
        body = (
            f"{opening}\n\n"
            f"We're following up on your onboarding. The next required step is: {action}."
            f"{urgency}\n\n"
            f"Until this is completed, {blocked} is blocked.\n\n"
            "Please complete the item when you can so the onboarding process can continue."
            f"{sign}"
        )
    elif role == "IT":
        body = (
            "Hello,\n\n"
            f"Please follow up on onboarding access for {name}. The current next action is: {action}."
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
            f"This is an onboarding follow-up for {name}. The next action is: {action}."
            f"{urgency}\n\n"
            f"Primary blocker: {bottleneck}.\n"
            f"Blocked downstream work: {blocked}."
            f"{sign}"
        )
    return body


def build_llm_prompt(row, purpose="explanation", sender_name="", recipient_role=""):
    payload = {
        "employee_id": _norm(row.get("Employee ID")),
        "employee_name": _norm(row.get("Employee Name")),
        "department": _norm(row.get("Department")),
        "days_to_start": row.get("Days to Start"),
        "current_stage": _norm(row.get("Current Stage")),
        "primary_bottleneck": _norm(row.get("Primary Bottleneck")),
        "blocked_downstream_task": _norm(row.get("Blocked Downstream Task")),
        "next_action": _norm(row.get("Next Action")),
        "responsible_party": _norm(row.get("Responsible Party")),
        "risk_level": _norm(row.get("Risk Level")),
        "missing_core_items": _norm(row.get("Missing Core Items")),
        "exception_flag": _norm(row.get("Exception Flag")),
        "recipient_role": recipient_role,
        "sender_name": sender_name,
    }
    if purpose == "email":
        task = "Draft a short, professional reminder email to the responsible stakeholder. Do not invent a due date, requirement, or email address."
    else:
        task = "Explain in 2-4 sentences why the employee is blocked or at risk and what should happen next."

    return f"""
You are an HR onboarding workflow assistant for a classroom prototype.
Use ONLY the supplied employee status and these business rules:
1. All core checklist items except SafeColleges should be complete before HR creates the HRIS employee ID.
2. Federal W-4, Iowa W-4, and Direct Deposit should be complete when the HRIS ID is created so payroll can be set up at the same time.
3. An HRIS ID may be created early, but missing payroll prerequisites must be flagged as payroll risk.
4. SafeColleges plus the full checklist are required before HR clears the employee for IT email/system access.
5. If an upstream prerequisite is incomplete, describe the downstream task as BLOCKED rather than merely incomplete.
6. Identify the responsible human stakeholder and next action. Do not make employment decisions and do not invent requirements.

Employee status JSON:
{json.dumps(payload, default=str, indent=2)}

Task: {task}
""".strip()


def optional_llm_text(row, purpose, api_key, model_name, sender_name="", recipient_role=""):
    """Use an OpenAI-compatible API only if the presenter supplies a key/model."""
    if not api_key or not model_name:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model_name,
            input=build_llm_prompt(row, purpose=purpose, sender_name=sender_name, recipient_role=recipient_role),
        )
        return response.output_text.strip()
    except Exception as e:
        return f"Optional LLM call was unavailable ({type(e).__name__}). The deterministic prototype still works."


# -----------------------------------------------------------------------------
# LIVE GOOGLE SHEETS HELPERS
# -----------------------------------------------------------------------------

def _extract_sheet_id(url_or_id):
    text = str(url_or_id or "").strip()
    if not text:
        return ""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    return m.group(1) if m else text

def read_public_google_sheet(sheet_url_or_id, worksheet_name="Employee_Data"):
    sheet_id = _extract_sheet_id(sheet_url_or_id)
    if not sheet_id:
        raise ValueError("Enter a Google Sheet URL or Sheet ID.")
    params = urllib.parse.urlencode({"tqx":"out:csv", "sheet":worksheet_name})
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if "<html" in r.text[:300].lower():
        raise RuntimeError(
            "Google returned a sign-in page. For this demo, share the Sheet as "
            "'Anyone with the link — Viewer'."
        )
    return pd.read_csv(io.StringIO(r.text))



# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="AI Onboarding Control Tower",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 AI-Enabled Employee Onboarding Control Tower")
st.caption("Group 6 classroom prototype — fictional/mock employee data only")

st.info(
    "The workflow engine determines status, blockers, ownership, and recipient role. "
    "AI may help draft communication language, but it does not invent HR requirements "
    "or email addresses."
)

# -------------------------
# Sidebar / session settings
# -------------------------
with st.sidebar:
    st.header("Session Settings")
    sender_name = st.text_input("Current HR user name", "Demo HR Coordinator")
    sender_email = st.text_input("Current HR user email", "hr.onboarding@example.com")
    default_it_email = st.text_input("Default IT contact email", "it.access@example.com")
    as_of_text = st.date_input("As-of date", value=date.today()).isoformat()

    st.divider()
    st.subheader("Data Source")
    source_type = st.radio("Choose source", ["Excel upload", "Live Google Sheet"])

# -------------------------
# Load data
# -------------------------
raw_df = None
test_df = pd.DataFrame()
contact_df = pd.DataFrame()
source_label = ""

if source_type == "Excel upload":
    uploaded = st.file_uploader("Upload onboarding Excel workbook", type=["xlsx","xls"])
    if uploaded is not None:
        try:
            xls = pd.ExcelFile(uploaded)
            emp_sheet = "Employee_Data" if "Employee_Data" in xls.sheet_names else xls.sheet_names[0]
            raw_df = pd.read_excel(uploaded, sheet_name=emp_sheet)
            if "Test_Cases" in xls.sheet_names:
                test_df = pd.read_excel(uploaded, sheet_name="Test_Cases")
            if "Contact_Directory" in xls.sheet_names:
                contact_df = pd.read_excel(uploaded, sheet_name="Contact_Directory")
            source_label = f"Excel: {uploaded.name}"
        except Exception as e:
            st.error(f"Could not read workbook: {e}")
else:
    sheet_url = st.text_input(
        "Google Sheet URL or ID",
        placeholder="https://docs.google.com/spreadsheets/d/..."
    )
    c1, c2 = st.columns([1,3])
    with c1:
        refresh = st.button("Load / Refresh Sheet", type="primary", use_container_width=True)
    with c2:
        st.caption(
            "For the classroom demo, share the fictional Sheet as "
            "'Anyone with the link — Viewer'."
        )

    if sheet_url:
        # Streamlit reruns on every interaction, so the sheet is effectively refreshed
        # whenever the page reruns or the button is clicked.
        try:
            raw_df = read_public_google_sheet(sheet_url, "Employee_Data")
            try:
                test_df = read_public_google_sheet(sheet_url, "Test_Cases")
            except Exception:
                test_df = pd.DataFrame()
            try:
                contact_df = read_public_google_sheet(sheet_url, "Contact_Directory")
            except Exception:
                contact_df = pd.DataFrame()
            source_label = "Live Google Sheet"
        except Exception as e:
            st.error(str(e))

if raw_df is None:
    st.warning("Load an Excel workbook or a live Google Sheet to begin.")
    st.stop()

missing = validate_schema(raw_df)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

analyzed = prioritize_df(analyze_rows(raw_df, as_of_text))

# -------------------------
# Dashboard
# -------------------------
tabs = st.tabs([
    "Priority Dashboard",
    "Employee Detail",
    "Communications",
    "Rule Validation",
    "About"
])

with tabs[0]:
    st.subheader("Priority Control Tower")
    st.caption(f"Source: {source_label} | Days to Start recalculated as of {as_of_text}")

    total = len(analyzed)
    red_n = int((analyzed["Priority"]=="Red").sum())
    yellow_n = int((analyzed["Priority"]=="Yellow").sum())
    green_n = int((analyzed["Priority"]=="Green").sum())

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Employees", total)
    m2.metric("🔴 Urgent", red_n)
    m3.metric("🟡 Pending", yellow_n)
    m4.metric("🟢 Complete", green_n)

    st.markdown(
        """
        **Priority logic:** 🔴 pending + start date within 7 days/overdue;
        🟡 pending + later start date; 🟢 fully operational.
        Within the same urgency/date, farther-downstream blockers are surfaced first.
        """
    )

    display_cols = [
        "Priority","Employee ID","Employee Name","Department","Start Date","Days to Start",
        "Current Stage","Primary Bottleneck","Blocked Downstream Task","Next Action",
        "Responsible Party","Risk Level"
    ]
    disp = analyzed[display_cols].copy()

    def color_priority(row):
        p = row["Priority"]
        if p == "Red":
            return ["background-color:#f8d7da"] * len(row)
        if p == "Yellow":
            return ["background-color:#fff3cd"] * len(row)
        return ["background-color:#d1e7dd"] * len(row)

    st.dataframe(
        disp.style.apply(color_priority, axis=1),
        use_container_width=True,
        hide_index=True,
        height=560
    )

    fig_stage, fig_risk = make_plots(analyzed)
    p1,p2 = st.columns(2)
    with p1:
        st.plotly_chart(fig_stage, use_container_width=True)
    with p2:
        st.plotly_chart(fig_risk, use_container_width=True)

    # Download analyzed workbook.
    export_path = export_analysis(analyzed)
    with open(export_path, "rb") as f:
        st.download_button(
            "Download analyzed workbook",
            f.read(),
            file_name="Onboarding_Control_Tower_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tabs[1]:
    st.subheader("Employee Control Tower")
    choices = [f"{r['Employee ID']} — {r['Employee Name']}" for _,r in analyzed.iterrows()]
    selected = st.selectbox("Select employee", choices)
    emp_id = selected.split(" — ",1)[0]
    row = analyzed[analyzed["Employee ID"].astype(str)==emp_id].iloc[0]

    p = row["Priority"]
    icon = {"Red":"🔴","Yellow":"🟡","Green":"🟢"}.get(p,"")
    st.markdown(f"### {row['Employee Name']} — {row['Employee ID']}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Priority", f"{icon} {p}")
    c2.metric("Days to Start", row["Days to Start"])
    c3.metric("Stage", row["Current Stage"])
    c4.metric("Risk", row["Risk Level"])

    st.markdown(f"**Primary blocker:** {row['Primary Bottleneck']}")
    st.markdown(f"**Blocked downstream task:** {row['Blocked Downstream Task']}")
    st.markdown(f"**Next action:** {row['Next Action']}")
    st.markdown(f"**Owner:** {row['Responsible Party']}")
    st.info(row["Control Tower Explanation"])

    checklist = pd.DataFrame(
        [[c, _norm(row.get(c))] for c in CORE_CHECKLIST + ["SafeColleges"]] +
        [
            ["HRIS Employee ID", _norm(row.get("HRIS Employee ID"))],
            ["Payroll Screens", _norm(row.get("Payroll Screens"))],
            ["HR Notified IT", _norm(row.get("HR Notified IT"))],
            ["IT Email/System Access", _norm(row.get("IT Email/System Access"))],
        ],
        columns=["Onboarding Item","Status"]
    )
    st.dataframe(checklist, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Draft Follow-Up Email")
    selected2 = st.selectbox(
        "Employee for communication",
        [f"{r['Employee ID']} — {r['Employee Name']}" for _,r in analyzed.iterrows()],
        key="comm_emp"
    )
    emp_id2 = selected2.split(" — ",1)[0]
    row2 = analyzed[analyzed["Employee ID"].astype(str)==emp_id2].iloc[0]

    route = resolve_recipient(
        row2,
        login_email=sender_email,
        default_it_email=default_it_email,
        contact_df=contact_df
    )

    st.caption(
        f"Suggested recipient role: {route['role']}. "
        "Recipient comes from workflow ownership + contact data, not AI guessing."
    )

    to_email = st.text_input("To", value=route["to"])
    cc_email = st.text_input("CC", value=route["cc"])
    subject = st.text_input("Subject", value=suggested_subject(row2))
    body_default = deterministic_reminder(
        row2,
        sender_name=sender_name,
        recipient_role=route["role"]
    )
    body = st.text_area("Email body", value=body_default, height=260)

    if to_email:
        params = {"view":"cm","fs":"1","to":to_email,"su":subject,"body":body}
        if cc_email:
            params["cc"] = cc_email
        gmail_url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(
            params, quote_via=urllib.parse.quote
        )
        st.link_button("Open Draft in Gmail", gmail_url)
    else:
        st.warning("No email address could be resolved. Confirm or enter the recipient manually.")

    if st.button("Send Email (Demo)"):
        st.success("Demo email logged as sent. No external email was transmitted.")

with tabs[3]:
    st.subheader("Rule Validation")
    md, val = validate_test_cases(analyzed, test_df)
    st.markdown(md)
    if len(val):
        st.dataframe(val, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("About the Prototype")
    st.markdown("""
**Core design**
- HRIS readiness gate
- payroll prerequisite gate
- SafeColleges / IT access gate
- downstream blocking
- human ownership / next action
- urgency prioritization
- role-based reminder drafting

**Live Google Sheet**
If the fictional Sheet is updated, Streamlit reloads the data on page rerun.
No Excel re-upload is required.

**Prototype boundary**
Use fictional/demo data in the classroom app. Do not publish real employee HR data
in a public Streamlit app.
""")
