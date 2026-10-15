/**
 * EstateIQ — Interactive Client Script (Phase 3)
 * Handles dual-control slider synchronization, request identity, AbortController cancellation,
 * empirical range warnings, client-side validation, and decoupled clear/reset lifecycles.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------------------
    const form = document.getElementById("prediction-form");
    const btnSubmit = document.getElementById("btn-submit-prediction");
    const btnSubmitText = document.getElementById("btn-submit-text");
    const btnSubmitIcon = document.getElementById("btn-submit-icon");
    const btnLoadSample = document.getElementById("btn-load-sample");
    const btnResetForm = document.getElementById("btn-reset-form");
    const btnResetSecondary = document.getElementById("btn-reset-secondary");
    const btnClearResult = document.getElementById("btn-clear-result");
    const alertBanner = document.getElementById("alert-banner");
    const alertMessage = document.getElementById("alert-message");
    const alertIcon = document.getElementById("alert-icon");

    // Result panel state elements
    const stateStandby = document.getElementById("state-standby");
    const stateLoading = document.getElementById("state-loading");
    const stateResult = document.getElementById("state-result");

    // Result display elements
    const displayDollar = document.getElementById("display-dollar-value");
    const displayRaw = document.getElementById("display-raw-value");
    const displayModelType = document.getElementById("display-model-type");
    const displayModelId = document.getElementById("display-model-id");
    const displayModelR2 = document.getElementById("display-model-r2");
    const displayModelMae = document.getElementById("display-model-mae");
    const chipsContainer = document.getElementById("summary-chips-container");

    // Header badge
    const headerStatusBadge = document.getElementById("header-status-badge");
    const headerStatusText = document.getElementById("header-status-text");

    // Mobile nav toggle
    const mobileNavToggle = document.getElementById("mobile-nav-toggle");
    const mainNavLinks = document.getElementById("main-nav-links");

    // -------------------------------------------------------------------------
    // Constants & Empirical Boundaries
    // -------------------------------------------------------------------------
    const FEATURE_KEYS = [
        "CRIM", "ZN", "INDUS", "CHAS", "RAD", "TAX", "RM", "AGE", "NOX", "DIS", "PTRATIO", "LSTAT"
    ];

    const SLIDER_KEYS = ["TAX", "RM", "AGE", "NOX", "DIS", "PTRATIO", "LSTAT"];

    const EMPIRICAL_BOUNDS = {
        CRIM: { min: 0.00632, max: 88.976, unit: "ratio" },
        ZN: { min: 0.0, max: 100.0, unit: "%" },
        INDUS: { min: 0.46, max: 27.74, unit: "%" },
        CHAS: { allowed: [0, 1] },
        NOX: { min: 0.385, max: 0.871, unit: "p10m" },
        RM: { min: 3.561, max: 8.780, unit: "rooms" },
        AGE: { min: 2.9, max: 100.0, unit: "%" },
        DIS: { min: 1.1296, max: 12.1265, unit: "dist" },
        RAD: { allowed: [1, 2, 3, 4, 5, 6, 7, 8, 24] },
        TAX: { min: 187.0, max: 711.0, unit: "$/10k" },
        PTRATIO: { min: 12.6, max: 22.0, unit: "ratio" },
        LSTAT: { min: 1.73, max: 37.97, unit: "%" },
    };

    // Typical authentic suburban Boston census tract benchmark sample
    const SAMPLE_PAYLOAD = {
        CRIM: 0.02731,
        ZN: 0.0,
        INDUS: 7.07,
        CHAS: 0,
        NOX: 0.469,
        RM: 6.421,
        AGE: 78.9,
        DIS: 4.9671,
        RAD: 2,
        TAX: 242.0,
        PTRATIO: 17.8,
        LSTAT: 9.14,
    };

    // -------------------------------------------------------------------------
    // Request Identity & Concurrency Management
    // -------------------------------------------------------------------------
    let currentRequestId = 0;
    let activeAbortController = null;

    // -------------------------------------------------------------------------
    // Mobile Nav Toggle
    // -------------------------------------------------------------------------
    if (mobileNavToggle && mainNavLinks) {
        mobileNavToggle.addEventListener("click", () => {
            mainNavLinks.classList.toggle("mobile-open");
        });
    }

    // -------------------------------------------------------------------------
    // Synchronize Sliders and Numeric Inputs
    // -------------------------------------------------------------------------
    SLIDER_KEYS.forEach((key) => {
        const inputEl = document.getElementById(key);
        const sliderEl = document.getElementById(`slider-${key}`);

        if (!inputEl || !sliderEl) return;

        // Numeric input -> Slider
        inputEl.addEventListener("input", () => {
            const val = parseFloat(inputEl.value);
            if (!isNaN(val)) {
                if (val >= parseFloat(sliderEl.min) && val <= parseFloat(sliderEl.max)) {
                    sliderEl.value = val;
                }
            }
            checkEmpiricalWarning(key, val);
        });

        // Slider -> Numeric input
        sliderEl.addEventListener("input", () => {
            inputEl.value = sliderEl.value;
            checkEmpiricalWarning(key, parseFloat(sliderEl.value));
            clearFieldError(key);
        });
    });

    // -------------------------------------------------------------------------
    // Binary CHAS Selector
    // -------------------------------------------------------------------------
    const chasHidden = document.getElementById("CHAS");
    const btnChas0 = document.getElementById("btn-chas-0");
    const btnChas1 = document.getElementById("btn-chas-1");

    function setChasValue(val) {
        if (!chasHidden) return;
        chasHidden.value = val;
        if (btnChas0) btnChas0.classList.toggle("active", val === 0);
        if (btnChas1) btnChas1.classList.toggle("active", val === 1);
        clearFieldError("CHAS");
    }

    if (btnChas0) {
        btnChas0.addEventListener("click", () => setChasValue(0));
    }
    if (btnChas1) {
        btnChas1.addEventListener("click", () => setChasValue(1));
    }

    // -------------------------------------------------------------------------
    // Empirical Range Warning Helper
    // -------------------------------------------------------------------------
    function checkEmpiricalWarning(key, val) {
        const warnEl = document.getElementById(`warning-${key}`);
        if (!warnEl) return;

        const bounds = EMPIRICAL_BOUNDS[key];
        if (!bounds || bounds.min === undefined) return;

        if (isNaN(val) || val === null) {
            warnEl.textContent = "";
            return;
        }

        if (val < bounds.min || val > bounds.max) {
            warnEl.textContent = `Notice: Value is outside 1970 empirical range [${bounds.min}, ${bounds.max}].`;
        } else {
            warnEl.textContent = "";
        }
    }

    // Check empirical bounds on change for non-slider inputs (CRIM, ZN, INDUS)
    ["CRIM", "ZN", "INDUS"].forEach((key) => {
        const inputEl = document.getElementById(key);
        if (inputEl) {
            inputEl.addEventListener("input", () => {
                const val = parseFloat(inputEl.value);
                checkEmpiricalWarning(key, val);
            });
        }
    });

    // -------------------------------------------------------------------------
    // System Health & Model Status Verification
    // -------------------------------------------------------------------------
    async function verifyModelReadiness() {
        try {
            const resp = await fetch("/api/health");
            const data = await resp.json();

            if (resp.ok && data.model_ready) {
                if (headerStatusBadge) {
                    headerStatusBadge.className = "status-indicator-badge ready";
                    if (headerStatusText) headerStatusText.textContent = "Model Ready";
                }
            } else {
                if (headerStatusBadge) {
                    headerStatusBadge.className = "status-indicator-badge unavailable";
                    if (headerStatusText) headerStatusText.textContent = "Model Unavailable";
                }
            }
        } catch (e) {
            if (headerStatusBadge) {
                headerStatusBadge.className = "status-indicator-badge unavailable";
                if (headerStatusText) headerStatusText.textContent = "Offline";
            }
        }
    }

    verifyModelReadiness();

    // -------------------------------------------------------------------------
    // Form Submission: Predict
    // -------------------------------------------------------------------------
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearErrors();
            hideAlert();

            const payload = extractAndValidateForm();
            if (!payload) return;

            // Increment request ID & abort any existing pending request
            const thisRequestId = ++currentRequestId;
            if (activeAbortController) {
                activeAbortController.abort();
            }
            activeAbortController = new AbortController();

            // Set UI to loading state
            setSubmissionLoading(true);
            setResultsState("loading");

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    body: JSON.stringify(payload),
                    signal: activeAbortController.signal,
                });

                const data = await response.json();

                // Disregard if a newer request was dispatched or form was reset
                if (thisRequestId !== currentRequestId) {
                    return;
                }

                if (response.ok && data.success) {
                    renderPredictionResult(data.data, payload);
                } else {
                    handlePredictionError(response.status, data);
                    setResultsState("standby");
                }
            } catch (err) {
                // If user aborted via reset or new request, quietly ignore
                if (err.name === "AbortError") {
                    return;
                }
                if (thisRequestId !== currentRequestId) {
                    return;
                }
                showAlert("A network error occurred while communicating with the prediction engine.", "danger");
                setResultsState("standby");
            } finally {
                if (thisRequestId === currentRequestId) {
                    setSubmissionLoading(false);
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // Form Extraction & Client-Side Validation
    // -------------------------------------------------------------------------
    function extractAndValidateForm() {
        const payload = {};
        let hasErrors = false;

        FEATURE_KEYS.forEach((key) => {
            const inputEl = document.getElementById(key);
            if (!inputEl) return;

            const rawVal = inputEl.value.trim();

            if (rawVal === "") {
                setFieldError(key, "This field is required.");
                hasErrors = true;
                return;
            }

            const numVal = parseFloat(rawVal);
            if (isNaN(numVal) || !isFinite(numVal)) {
                setFieldError(key, "Please enter a valid numeric value.");
                hasErrors = true;
                return;
            }

            // Key-specific constraints
            if (key === "CHAS") {
                if (numVal !== 0 && numVal !== 1) {
                    setFieldError(key, "CHAS must be 0 or 1.");
                    hasErrors = true;
                    return;
                }
                payload[key] = parseInt(numVal, 10);
                return;
            }

            if (key === "RAD") {
                const intRad = parseInt(numVal, 10);
                const allowed = EMPIRICAL_BOUNDS.RAD.allowed;
                if (!allowed.includes(intRad)) {
                    setFieldError(key, `Select a valid accessibility level (${allowed.join(", ")}).`);
                    hasErrors = true;
                    return;
                }
                payload[key] = intRad;
                return;
            }

            // Non-negativity checks
            if (numVal < 0 && key !== "ZN") {
                setFieldError(key, "Value cannot be negative.");
                hasErrors = true;
                return;
            }

            payload[key] = numVal;
        });

        if (hasErrors) {
            showAlert("Please correct the highlighted inputs before generating an estimate.", "danger");
            return null;
        }

        return payload;
    }

    // -------------------------------------------------------------------------
    // Render Prediction Output & Feature Chips
    // -------------------------------------------------------------------------
    function renderPredictionResult(result, submittedPayload) {
        if (displayDollar) displayDollar.textContent = result.prediction_display;
        if (displayRaw) displayRaw.textContent = `${result.prediction_raw}`;
        if (displayModelType) displayModelType.textContent = result.model_type || "GradientBoostingRegressor";
        if (displayModelId) displayModelId.textContent = result.model_identifier || "boston_housing_pipeline";

        if (result.metrics) {
            if (displayModelR2 && result.metrics.r2 !== undefined) {
                displayModelR2.textContent = result.metrics.r2.toFixed(4);
            }
            if (displayModelMae && result.metrics.mae !== undefined) {
                const dollarMae = result.metrics.mae * 1000.0;
                displayModelMae.textContent = `$${dollarMae.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            }
        }

        // Populate submitted feature chips
        if (chipsContainer && submittedPayload) {
            chipsContainer.innerHTML = "";
            Object.entries(submittedPayload).forEach(([k, v]) => {
                const chip = document.createElement("div");
                chip.className = "summary-chip";
                chip.innerHTML = `${k}: <strong>${v}</strong>`;
                chipsContainer.appendChild(chip);
            });
        }

        setResultsState("result");
    }

    // -------------------------------------------------------------------------
    // Separate Actions: Clear Prediction vs Reset Form
    // -------------------------------------------------------------------------

    // Action A: Clear Prediction
    // Removes the prediction result, leaves user's form inputs intact, allows immediate re-prediction
    function clearPrediction() {
        // Cancel pending requests to prevent race condition
        currentRequestId++;
        if (activeAbortController) {
            activeAbortController.abort();
        }
        setSubmissionLoading(false);
        setResultsState("standby");
        hideAlert();
    }

    if (btnClearResult) {
        btnClearResult.addEventListener("click", clearPrediction);
    }

    // Action B: Reset Form
    // Clears prediction, resets form controls, sliders, clears errors, cancels pending requests
    function resetForm() {
        currentRequestId++;
        if (activeAbortController) {
            activeAbortController.abort();
        }

        if (form) form.reset();

        // Reset CHAS to default 0
        setChasValue(0);

        // Reset RAD to default unselected
        const radEl = document.getElementById("RAD");
        if (radEl) radEl.selectedIndex = 0;

        // Reset sliders to initial default values
        SLIDER_KEYS.forEach((key) => {
            const sliderEl = document.getElementById(`slider-${key}`);
            const inputEl = document.getElementById(key);
            if (sliderEl && inputEl) {
                sliderEl.value = sliderEl.min;
                inputEl.value = "";
            }
            const warnEl = document.getElementById(`warning-${key}`);
            if (warnEl) warnEl.textContent = "";
        });

        // Reset non-slider warnings
        ["CRIM", "ZN", "INDUS"].forEach((key) => {
            const warnEl = document.getElementById(`warning-${key}`);
            if (warnEl) warnEl.textContent = "";
        });

        clearErrors();
        hideAlert();
        setSubmissionLoading(false);
        setResultsState("standby");
    }

    if (btnResetForm) {
        btnResetForm.addEventListener("click", resetForm);
    }
    if (btnResetSecondary) {
        btnResetSecondary.addEventListener("click", resetForm);
    }

    // -------------------------------------------------------------------------
    // Load Sample Data Action
    // -------------------------------------------------------------------------
    function loadSampleData() {
        clearErrors();
        hideAlert();

        Object.entries(SAMPLE_PAYLOAD).forEach(([k, val]) => {
            const inputEl = document.getElementById(k);
            if (!inputEl) return;

            if (k === "CHAS") {
                setChasValue(val);
                return;
            }

            inputEl.value = val;

            // Update associated slider if present
            const sliderEl = document.getElementById(`slider-${k}`);
            if (sliderEl) {
                sliderEl.value = val;
            }

            checkEmpiricalWarning(k, val);
        });

        showAlert("Loaded representative 1970 Boston tract sample. Select 'Estimate House Price' to compute valuation.", "info");
    }

    if (btnLoadSample) {
        btnLoadSample.addEventListener("click", loadSampleData);
    }

    // -------------------------------------------------------------------------
    // Error Handling & State UI Helpers
    // -------------------------------------------------------------------------
    function handlePredictionError(statusCode, data) {
        if (statusCode === 503) {
            showAlert(data.error || "Prediction engine is currently unavailable. No trained artifact loaded.", "danger");
            return;
        }

        if (statusCode === 400 || statusCode === 422) {
            if (data.details && typeof data.details === "object") {
                Object.entries(data.details).forEach(([f, msg]) => {
                    setFieldError(f, msg);
                });
                showAlert("Validation error: Some inputs did not satisfy schema constraints.", "danger");
            } else {
                showAlert(data.error || "Input validation failed. Please review your entries.", "danger");
            }
            return;
        }

        showAlert(data.error || "An unexpected server error occurred during prediction.", "danger");
    }

    function setSubmissionLoading(isLoading) {
        if (!btnSubmit) return;
        btnSubmit.disabled = isLoading;
        if (btnSubmitIcon) {
            btnSubmitIcon.className = isLoading ? "spinner-icon" : "";
            btnSubmitIcon.textContent = isLoading ? "" : "⚡";
        }
        if (btnSubmitText) {
            btnSubmitText.textContent = isLoading ? "Computing Estimate..." : "Estimate House Price";
        }
    }

    function setResultsState(state) {
        if (stateStandby) stateStandby.classList.toggle("hidden", state !== "standby");
        if (stateLoading) stateLoading.classList.toggle("hidden", state !== "loading");
        if (stateResult) stateResult.classList.toggle("hidden", state !== "result");
    }

    function clearErrors() {
        document.querySelectorAll(".field-error-msg").forEach((el) => {
            el.textContent = "";
        });
        document.querySelectorAll(".form-input, .form-select").forEach((el) => {
            el.classList.remove("input-error");
        });
    }

    function setFieldError(key, msg) {
        const errorEl = document.getElementById(`error-${key}`);
        if (errorEl) errorEl.textContent = msg;

        const inputEl = document.getElementById(key);
        if (inputEl) inputEl.classList.add("input-error");
    }

    function clearFieldError(key) {
        const errorEl = document.getElementById(`error-${key}`);
        if (errorEl) errorEl.textContent = "";

        const inputEl = document.getElementById(key);
        if (inputEl) inputEl.classList.remove("input-error");
    }

    function showAlert(msg, type = "info") {
        if (!alertBanner || !alertMessage) return;
        alertMessage.textContent = msg;
        alertBanner.className = `alert alert-${type}`;

        if (alertIcon) {
            alertIcon.textContent = type === "danger" ? "⚠️" : type === "success" ? "✓" : "ℹ️";
        }
        alertBanner.classList.remove("hidden");
    }

    function hideAlert() {
        if (alertBanner) alertBanner.classList.add("hidden");
    }
});
