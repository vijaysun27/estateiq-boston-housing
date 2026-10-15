# EstateIQ — Boston Housing Price Predictor (Phase 5 Production Ready)

**EstateIQ** is a production-hardened machine learning web application built with Flask, Scikit-Learn, TensorFlow/Keras, and modern vanilla frontend technologies. The system predicts historical median housing valuations for Boston census tracts based on 12 validated property and neighborhood characteristics, completely excluding the historical discriminatory variable `B`.

---

## 1. Project Objectives & Features

- **Interactive Valuation Engine**: SaaS-grade analytics dashboard with grouped input cards, bidirectional range sliders, and dynamic prediction chips.
- **Trained Scikit-Learn Pipeline**: Evaluated `GradientBoostingRegressor` with `StandardScaler` (`models/boston_housing_pipeline.joblib`) achieving an $R^2$ of **0.8310** and MAE of **$2,381.50** on held-out test data.
- **Deep ANN Multi-Engine Support**: Forward and backward compatibility for the 4-layer Keras Sequential ANN (`models/boston_housing_model.keras`).
- **Strict Schema & Range Validation**: All 12 inputs are thoroughly validated against empirical dataset bounds, discrete accessibility levels (RAD), and non-negativity constraints. Unexpected fields are rejected.
- **Ethical Feature Policy**: Historical discriminatory variable `B` is strictly excluded from training, UI, and API endpoints.
- **Historical Benchmark Clarity**: Clearly labels outputs as 1970s benchmark valuations ($1,000s) to prevent confusion with modern appraisals.
- **Production Hardened**: Includes security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options), payload size limits (16 KB), graceful 503 error handling without tracebacks, and race-condition prevention via `AbortController`.
- **Automated Testing Suite**: **50 automated unit, integration, and functional acceptance tests** covering User Journeys A through G.
- **Production WSGI & PaaS Ready**: Configured for Gunicorn on Linux hosts and automated deployment to Render via Infrastructure-as-Code (`render.yaml`).

---

## 2. Directory Structure

```text
estateiq-boston-housing/
├── app/
│   ├── __init__.py            # Hardened Flask application factory & security headers
│   ├── config.py              # Environment configuration & 16 KB MAX_CONTENT_LENGTH
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── main.py            # Web routes: /, /insights, /about
│   │   └── prediction.py      # REST API: /api/predict, /api/health, /api/sample
│   ├── services/
│   │   ├── __init__.py
│   │   └── predictor.py       # Decoupled inference service (dict & DataFrame support)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── validation.py      # Input validation, bounds, RAD checks & 12-feature schema
│   ├── templates/
│   │   ├── base.html          # SaaS navigation, status badge, and provenance footer
│   │   ├── index.html         # 2-column prediction dashboard with dual controls
│   │   ├── insights.html      # Model insights, CV table, and feature importance chart
│   │   ├── about.html         # Ethical policy, provenance, and dataset limitations
│   │   └── errors/
│   │       ├── 404.html       # 404 Not Found error template
│   │       └── 500.html       # 500 Server Error error template
│   └── static/
│       ├── css/
│       │   └── styles.css     # Premium SaaS dark design system
│       └── js/
│           └── app.js         # Slider sync, AbortController, and decoupled clear/reset
├── data/
│   ├── README.md              # Dataset provenance, license, and schema documentation
│   └── boston_housing.csv     # Verified 506-observation Harrison & Rubinfeld dataset
├── models/
│   ├── README.md              # Model artifact loading and registry notes
│   ├── boston_housing_pipeline.joblib # Trained StandardScaler + GradientBoosting pipeline (~304 KB)
│   ├── model_metadata.json    # Phase 2 metrics, CV results, hyperparameters, provenance
│   ├── boston_housing_model.keras     # Pre-trained Keras Sequential ANN
│   └── scaler.joblib          # Pre-fitted StandardScaler for continuous features
├── scripts/
│   └── train_model.py         # Reproducible scikit-learn training & CV pipeline
├── tests/
│   ├── conftest.py            # Pytest application fixtures and sample payloads
│   ├── test_acceptance_journeys.py # Functional acceptance tests (Journeys A through G)
│   ├── test_routes.py         # HTTP endpoints & error handling tests
│   ├── test_validation.py     # Schema validation and bounds unit tests
│   ├── test_predictor.py      # Predictor service unit tests
│   └── test_training_pipeline.py # Dataset verification & training pipeline unit tests
├── docs/
│   ├── architecture.md        # Architectural flow and design decisions
│   └── model_card.md          # Model card, ethical policy, CV comparison & test metrics
├── .env.example               # Example non-secret configuration variables
├── .gitignore                 # Standard Python/IDE/Environment ignore patterns
├── .python-version            # Python runtime specification (3.12.3)
├── Procfile                   # Gunicorn production process definition
├── render.yaml                # Render Infrastructure-as-Code Blueprint specification
├── pytest.ini                 # Pytest configuration with pythonpath set
├── requirements.txt           # Production dependencies (Flask, Gunicorn, scikit-learn, etc.)
├── requirements-dev.txt       # Testing and development dependencies (pytest, flake8)
├── wsgi.py                    # Production WSGI application entrypoint
└── README.md
```

---

## 3. Quick Start & Local Execution

### 1. Setup Environment
```bash
cd estateiq-boston-housing
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 2. Run Model Training (Optional — Pre-trained Artifacts Included)
```bash
python scripts/train_model.py
```
Outputs `models/boston_housing_pipeline.joblib` and `models/model_metadata.json`.

### 3. Run Automated Tests
```bash
pytest
```
Executes all **50 unit, integration, and functional acceptance tests** with zero failures.

### 4. Start Local Development Server
```bash
python wsgi.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 4. Production WSGI Server Configuration

In production environments (e.g. Linux cloud containers), run the application using **Gunicorn**:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 2 --access-logfile - --error-logfile - wsgi:app
```

- **Entrypoint**: `wsgi:app` (loads `create_app("production")` via [wsgi.py](file:///c:/Users/Vijay/Documents/ProITBridge/Deep_Learning/Chapter8_ANN_for_Regression/estateiq-boston-housing/wsgi.py)).
- **Binding**: Binds to all network interfaces on the dynamically assigned platform port (`$PORT`).
- **Workers**: 2 worker processes (optimal for memory usage and concurrency on Render free/starter tiers).
- **Logging**: Directs access and error logs to standard output/error (`-`) for platform observability.

---

## 5. Production Hosting Deployment (Render)

EstateIQ is fully configured for deployment on [Render](https://render.com).

### Method A: Declarative Blueprint Deployment (`render.yaml`) — Recommended

1. Push your project to GitHub or GitLab:
   ```bash
   git init
   git add .
   git commit -m "feat: complete EstateIQ production web application"
   git remote add origin https://github.com/<your-username>/estateiq-boston-housing.git
   git push -u origin main
   ```
2. Log in to your [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** and select **Blueprint**.
4. Connect your `estateiq-boston-housing` repository.
5. Render detects [render.yaml](file:///c:/Users/Vijay/Documents/ProITBridge/Deep_Learning/Chapter8_ANN_for_Regression/estateiq-boston-housing/render.yaml) and automatically configures:
   - **Service Name**: `estateiq-boston-housing`
   - **Runtime**: `Python 3.12.3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --access-logfile - --error-logfile - wsgi:app`
   - **Health Check Path**: `/api/health`
   - **Environment Variables**: `FLASK_ENV=production`, `LOG_LEVEL=INFO`, `SECRET_KEY` (auto-generated)
6. Click **Apply**. Render will build and deploy the application.

### Method B: Manual Web Service Creation via Render Dashboard

If you prefer configuring via the web UI without blueprints:
1. On the Render Dashboard, click **New +** > **Web Service**.
2. Select your repository.
3. Configure the following parameters:
   - **Name**: `estateiq-boston-housing`
   - **Region**: `Oregon (US West)` or preferred region
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --access-logfile - --error-logfile - wsgi:app`
   - **Instance Type**: `Free`
4. Under **Advanced**:
   - **Health Check Path**: `/api/health`
   - **Auto-Deploy**: `Yes`
5. Under **Environment Variables**, add:
   - `FLASK_ENV` = `production`
   - `FLASK_DEBUG` = `0`
   - `LOG_LEVEL` = `INFO`
   - `PYTHON_VERSION` = `3.12.3`
   - `SECRET_KEY` = *(click "Generate" to generate a secure random secret)*
6. Click **Create Web Service**.

### Continuous Deployment
Every subsequent `git push origin main` triggers an automatic zero-downtime rebuild and redeployment on Render.

---

## 6. Model Artifact Strategy

1. **Committed Artifact in Git**:
   - The primary trained model pipeline (`models/boston_housing_pipeline.joblib`) is approximately **304 KB**.
   - Model metadata (`models/model_metadata.json`) is **2.9 KB**.
   - Because these artifacts are well under Git's 100 MB file limit, they are committed directly to version control.
2. **Zero-Retraining Startup**:
   - The application **never retrains the model on deployment or container startup**.
   - `PredictorService.load_model()` deserializes the pipeline into memory once per worker process during container boot.
   - Startup latency is under 500 ms with zero external S3/GCS download dependencies or risk of third-party network failure.
3. **Storage Hygiene**:
   - The raw dataset is ~35 KB (506 rows) and no extraneous large binary files or unneeded checkpoints are bundled into the container image.

---

## 7. Environment & Security Configuration

Configure application runtime behavior using environment variables:

| Variable | Default Value | Production Value | Description |
| :--- | :--- | :--- | :--- |
| `FLASK_ENV` | `development` | `production` | Selects configuration class (`ProductionConfig`). |
| `FLASK_DEBUG` | `1` | `0` | Disables debug mode and interactive traceback debugger. |
| `SECRET_KEY` | *(dev fallback)* | *(Render auto-generated)* | Flask session and CSRF signature key. |
| `PORT` | `5000` | Assigned by `$PORT` | Port for the HTTP/WSGI server. |
| `HOST` | `127.0.0.1` | `0.0.0.0` | Network binding interface. |
| `MODEL_PIPELINE_PATH`| `models/boston_housing_pipeline.joblib` | `models/boston_housing_pipeline.joblib` | Filesystem path to primary trained Scikit-Learn pipeline. |
| `LOG_LEVEL` | `INFO` | `INFO` | Logging threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PYTHON_VERSION` | `3.12.3` | `3.12.3` | Target Python runtime. |

### Security Protections
- **No Committed Secrets**: `.env` is explicitly ignored by `.gitignore`. Only `.env.example` is committed.
- **Payload Clamping**: Request bodies exceeding `16 KB` receive `HTTP 413 Payload Too Large`.
- **Traceback Shielding**: All 400, 404, 405, 413, and 500 error handlers return generic JSON responses or styled error templates without leaking directory paths or Python stack traces.
- **Standard Output Logging**: Production logs are piped to `stdout` and `stderr` for ingestion by cloud logging agents.

---

## 8. Health Checks & Observability

EstateIQ provides a dedicated health check endpoint at `/api/health`.

### `GET /api/health`
Differentiates between **application process availability**, **model readiness**, and **prediction service availability**:

```json
{
  "status": "healthy",
  "application": "running",
  "model_ready": true,
  "prediction_service": "available",
  "model_info": {
    "status": "ready",
    "model_type": "Scikit-Learn Pipeline (GradientBoostingRegressor)",
    "model_identifier": "boston_housing_pipeline",
    "supported_features": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"],
    "target": "MEDV ($1,000s historical benchmark)",
    "test_metrics": {
      "mae": 2.3815,
      "rmse": 3.9177,
      "r2": 0.831
    }
  }
}
```

- When the model artifact is loaded and verified: Returns **HTTP 200 OK**.
- If the artifact cannot be loaded: Returns **HTTP 503 Service Unavailable** with `"model_ready": false` and `"prediction_service": "unavailable"`.

---

## 9. Model Evaluation & Benchmark Results

Model selection was conducted via 5-fold cross-validation on 404 training samples (20% held out under random seed 42):

| Estimator Candidate | Preprocessing | 5-Fold CV RMSE | 5-Fold CV MAE | 5-Fold CV $R^2$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyRegressor (Mean)** | *None* | 9.0813 | 6.6402 | -0.0103 | Baseline |
| **Linear Regression** | `StandardScaler` | 4.3547 | 3.1822 | 0.7595 | Evaluated |
| **Ridge Regression** | `StandardScaler` | 4.3501 | 3.1737 | 0.7600 | Evaluated |
| **Random Forest** | `StandardScaler` | 3.5476 | 2.4351 | 0.8286 | Evaluated |
| **Gradient Boosting** | `StandardScaler` | **3.3571** | **2.2536** | **0.8460** | **Selected** |

### Held-Out Test Set Performance (102 Observations)

| Metric | Dummy Baseline | Selected GradientBoosting Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** ($1,000s) | 6.7727 ($6,772.70) | **2.3815 ($2,381.50)** | **64.8% error reduction** |
| **RMSE** ($1,000s) | 9.5319 ($9,531.90) | **3.9177 ($3,917.70)** | **58.9% error reduction** |
| **$R^2$ Score** | -0.0003 | **0.8310** | **Explains 83.1% of variance** |

---

## 10. Post-Deployment Verification Checklist

Once deployed to Render (or your chosen hosting provider), verify the live service using this checklist:

1. **HTTPS & Security**: Navigate to `https://<service-name>.onrender.com`. Verify valid SSL certificate and that security headers (`Content-Security-Policy`, `X-Content-Type-Options: nosniff`) are present in response headers.
2. **Asset Loading**: Confirm CSS (`/static/css/styles.css`), JavaScript (`/static/js/app.js`), and Google Fonts load cleanly with `200 OK`.
3. **Health Check Probe**: Verify `https://<service-name>.onrender.com/api/health` returns HTTP 200 with `"status": "healthy"`, `"application": "running"`, and `"model_ready": true`.
4. **Sample Prediction**: Click **Load Sample** on the dashboard; verify input sliders populate. Click **Calculate Valuation** and confirm a valid prediction appears (e.g., ~$24,316) with benchmark disclaimer.
5. **Decoupled Clear & Reset**:
   - Click **Clear Prediction**: The output card resets to standby while preserving form values.
   - Click **Reset Form**: All sliders and numeric inputs return to initial defaults.
6. **Navigation**: Visit `/insights` (Feature importances & CV metrics) and `/about` (Provenance & ethical policy); confirm zero layout breaking.
7. **Input Validation**: Test invalid input (e.g. submitting out-of-range values or empty fields) and ensure graceful client-side and server-side validation error displays without crashing.
8. **Mobile Responsiveness**: Resize viewport to 375px width; verify controls, cards, and navigation stack cleanly.

> [!NOTE]
> **Live Deployment Status**: The codebase is pre-configured and deployment-ready for Render hosting. Because direct Render account credentials and an external Git remote are not connected to this local environment, the live deployment will activate as soon as the repository is pushed to GitHub/GitLab and connected to Render using the steps above.

---

## 11. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `HTTP 503 Model Unavailable` | Artifact missing in `models/` | Ensure `models/boston_housing_pipeline.joblib` is committed to Git or run `python scripts/train_model.py`. |
| `HTTP 400 Validation Error` | Missing features or unexpected keys | Verify all 12 canonical keys are present. Do not include key `B`. |
| `HTTP 413 Payload Too Large` | Payload > 16 KB | Ensure JSON body is under 16 KB limit. |
| `Address already in use` | Another process occupies port 5000 | Set custom port: `$env:PORT="5050"; python wsgi.py`. |
| `ImportError: No module named 'app'` | Running pytest without pythonpath | Ensure `pytest.ini` is present or invoke with `python -m pytest`. |
| `Gunicorn worker timeout` | Heavy startup or cold boot | Check container memory limits and ensure pre-trained artifact is loaded from disk (not retrained). |

---

## 12. Dataset Limitations & Ethical Disclaimer

1. **Academic Historical Benchmark**: Data represents 1970 Boston tract metrics. Real estate values and neighborhood demographics have changed fundamentally over the last 50+ years.
2. **Not Financial or Real Estate Advice**: The predicted prices are for instructional demonstrations of regression algorithms only.
3. **Fair Housing & Variable `B` Exclusion**: Feature `B` is permanently dropped from modern training workflows to adhere to fair housing principles and avoid algorithmic perpetuation of historical segregation.
