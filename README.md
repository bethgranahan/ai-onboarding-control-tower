# AI Onboarding Control Tower

Streamlit deployment package for the Group 6 MBA:8240 prototype.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and the optional demo workbook.
3. Go to https://share.streamlit.io/
4. Create an app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

Every time you commit/push a code change to GitHub, Streamlit Community Cloud
will redeploy the app from the updated repository.

## Live Google Sheet
For the fictional classroom demo:
1. Import the mock workbook into Google Sheets.
2. Keep the `Employee_Data` worksheet name.
3. Share the Google Sheet as `Anyone with the link — Viewer`.
4. Paste the Sheet URL into the app.

Optional worksheets:
- `Contact_Directory`
- `Test_Cases`

Use fictional data only in a public demo.
