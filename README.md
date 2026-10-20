# ai-assistant-2# ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get:

- an **ATS score out of 100** with a per-category breakdown,
- **prioritised improvements**, each with the problem, a specific fix and an example rewrite,
- **missing keywords**, optionally measured against a job description you paste in,
- the **plain text an ATS would extract** from your file, so you can spot parsing problems,
- a downloadable Markdown report.

Built with [Streamlit](https://streamlit.io) and the Google Gemini API (Flash model).

## How the score works

The score is hybrid, so it is explainable and does not depend on the model doing arithmetic.
Python computes the final number from fixed weights.

| Category | Weight | Computed by |
|---|---|---|
| Keywords & relevance | 25% | Gemini |
| Impact & achievements | 20% | Gemini |
| Clarity & language | 10% | Gemini |
| ATS-friendly content | 10% | Gemini |
| Contact details (email, phone, LinkedIn/GitHub) | 10% | Python |
| Standard sections (summary, experience, education, skills) | 10% | Python |
| Quantified results (share of bullets containing numbers) | 10% | Python |
| Resume length (best at roughly 400-800 words) | 5% | Python |

The score is a guide, not what any real employer's ATS will output. Every ATS is different.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. You can either paste it into the
app's sidebar or configure it once (see below).

### Configuration

The app reads these settings from Streamlit secrets first, then from environment variables:

| Name | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No | If set, visitors do not need their own key. |
| `GEMINI_MODEL` | No | Model name. Defaults to `gemini-2.5-flash`. |

For local use, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-2.5-flash"
```

Never commit this file (see `.gitignore` below).

> **Model names change.** If you see "could not find that model", look up the current Flash model
> name in the [Gemini models list](https://ai.google.dev/gemini-api/docs/models) and set it in the
> sidebar or in `GEMINI_MODEL`.

## Push to GitHub

```bash
# inside the project folder
printf ".venv/\n__pycache__/\n.streamlit/secrets.toml\n.env\n" > .gitignore
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: ATS resume checker"
git branch -M main
```

Create an empty repository on GitHub (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

GitHub does not accept account passwords for pushes. When prompted, use a
[personal access token](https://github.com/settings/tokens) as the password, or sign in with
GitHub CLI (`gh auth login`).

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Pick your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** and paste your secrets in TOML format:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes.

Every `git push` to `main` redeploys automatically.

### Who pays for the AI calls?

- **Set `GEMINI_API_KEY` in secrets:** visitors use your key and your quota. The app limits each
  browser session to 10 analyses, but a determined visitor can refresh to reset that. Fine for a
  portfolio demo; keep an eye on your usage.
- **Leave it unset:** each visitor pastes their own key in the sidebar. Nothing is billed to you.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Almost no text could be read" | The PDF is a scanned image or heavily designed. Export a text-based PDF or DOCX. |
| Extracted text looks scrambled | Multi-column layouts, tables and text boxes confuse parsers. Use a single-column layout. |
| "Gemini rejected the API key" | Check the key and that the Gemini API is enabled for it. |
| "Rate limit or quota reached" | Wait a minute, or use a key with a higher quota. |
| Build fails on Streamlit Cloud | Check the logs; confirm `requirements.txt` is in the repo root. |

## Privacy

Resume text is sent to Google's Gemini API for analysis. The app does not save uploaded files or
text; results live only in the visitor's session. Review Google's terms for the API tier you use,
since free-tier data may be used to improve Google's products.

## Project files

- `app.py`: the whole app (extraction, checks, Gemini call, UI)
- `requirements.txt`: Python dependencies
- `README.md`: this file
