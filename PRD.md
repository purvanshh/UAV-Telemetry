# PRD: UAV Telemetry Anomaly Detection & AI Diagnostic Assistant

Product Requirements Document · Version 1.0 · October 2026

## 1. Product overview

Product name: UAV Sentinel — Telemetry Intelligence & Diagnostics

UAV Sentinel is a web application that enables users to upload UAV flight telemetry, detect anomalous behavior using a pretrained LSTM Autoencoder, inspect anomaly events through visualizations and structured JSON reports, and investigate potential causes through an AI-powered conversational assistant using OpenAI GPT-4o mini.

The product builds on the existing preprocessing, training, inference and validation artifacts developed in the Kaggle notebook. The goal is to turn the current ML pipeline into a usable software application without retraining the model as part of the normal upload workflow.

## 2. Problem statement

UAV flight telemetry is distributed across multiple sensor streams, including IMU, ATT and BARO. Analyzing these files manually makes it difficult to identify unusual behavior, understand when it occurred, and investigate possible causes.

The existing anomaly detection pipeline can score flight telemetry, but it lacks a complete user-facing workflow for uploading data, reviewing detected events, generating reports and asking follow-up questions.

UAV Sentinel will bridge that gap by combining the existing anomaly detector with structured reporting, visual analysis and an AI assistant.

## 3. Product goals

- Allow users to upload IMU, ATT and BARO CSV files and process them as one flight.
- Reuse the existing trained LSTM Autoencoder and its associated preprocessing artifacts.
- Generate anomaly scores and group flagged windows into meaningful events.
- Produce downloadable, machine-readable JSON reports.
- Use GPT-4o mini to interpret anomaly evidence and propose plausible explanations.
- Provide a conversational interface for follow-up questions about the analyzed flight.
- Present results in a clean, minimal, productivity-focused interface.

### Non-goals

The initial version will not:

- Train or fine-tune models through the UI.
- Guarantee detection of every anomaly or diagnose every fault correctly.
- Control a UAV or trigger autonomous flight actions.
- Require cloud deployment, multi-user authentication or a production database.
- Replace specialist inspection or validated flight-safety systems.

## 4. Target users

[Growth in engineering sector calls for more engineers to deliver secure future](https://images.openai.com/static-rsc-4/BavVkOEwKHLZPy8lKINsIfwcb_DiYScdP1C4SB73j9-ZjZREuZGo8W6NuPcGwp1vz0WYhuO-_u4pQl1e8J0VPSADYvZhDlN5iQjw_jt2e2Zj3xl1wY_FTXEX1NE_Q-eL-abAz-r7wbqaU3apy3SBOaALJdgfQCJwEmanJhwyAH8?purpose=inline)

UAV / ML researchers

Upload flight logs, inspect anomalies and evaluate model behavior across different flights.

UAV engineers

Investigate unusual sensor behavior and explore possible causes using telemetry evidence.

[Applied Artificial Intelligence and Data Science, Bachelor of Science (B.S.) | Programs of Study | Miner School of Computer and Information Sciences | Kennedy College of Sciences | UMass Lowell](https://images.openai.com/static-rsc-4/XBMENb6bWl9nBFmegPuALrtIb0_T9slSu35gptFfEMSLJhrG_XsrxS2NPnTMEiPsSQ0IufjRBrZv9xKUtRr7VsMrqxbsGj5n5-bMk7Kc59H6ahPeSpueGxHUHBjVLNZrI873XCJHqrRigppQWrjbyQhHUYAZXbAAVJV5gvz4wJc?purpose=inline)

Project evaluators and demonstrators

Review a complete anomaly detection and AI-assisted diagnostic workflow.

## 5. Product scope and core workflow

### 01

### Upload

Select the three sensor files or upload a supported ZIP archive.

### 02

### Validate

Check file structure, columns, timestamps and data quality.

### 03

### Analyze

Align telemetry, generate model-compatible windows and run inference.

### 04

### Review

Inspect anomaly scores, events, affected signals and warnings.

### 05

### Diagnose

Send structured evidence to GPT-4o mini for interpretation.

### 06

### Chat & export

Ask follow-up questions and download the JSON report.

## 6. Functional requirements

### FR-01: Telemetry upload

The application must support:

- Three individual CSV uploads: IMU, ATT and BARO.
- A ZIP archive containing the three CSV files, with recognized filenames or folders.
- A user-provided flight identifier, or a generated identifier when none is provided.
- File-size and row-count validation, with clear error messages.
- A preview of filenames, detected columns, timestamps and row counts before analysis.

The initial version will support CSV input matching the schema understood by the existing preprocessing pipeline. Other telemetry formats are outside the initial scope.

### FR-02: Data validation and preprocessing

The system must:

- Validate required timestamp and sensor columns.
- Verify compatible timestamps and convert `TimeUS` using the verified unit convention.
- Align IMU, ATT and BARO streams to the model's expected time grid.
- Preserve session boundaries and avoid creating windows across large source-data gaps.
- Reproduce the deployed model's exact feature order, transformations and scaling.
- Identify missing values, invalid values, extreme sensor readings and insufficient valid windows.
- Show warnings when data quality may affect the analysis.

Important: The deployed model uses 12 conservative features. The upload pipeline must reproduce the preprocessing expected by that model, rather than passing the original 17-feature set directly into inference.

The current pipeline's handling of extreme BARO values must remain transparent. Since BARO-derived features are excluded from the deployed model, the system may report BARO quality issues but must not claim to detect pressure or altitude anomalies through the LSTM.

### FR-03: Anomaly detection

The system must load the existing artifacts from the `uav_model/` directory, including:

- `lstm_autoencoder.pth`
- `inference.py`
- `preprocessing.py`
- `model_config.json`
- `feature_schema.json`
- `scaler.json`
- `threshold_config.json`

The detector must:

- Generate windows of 30 timesteps with a stride of 15 timesteps.
- Compute a reconstruction-based anomaly score for each valid window.
- Apply the saved conservative-model threshold from `uav_model/threshold_config.json` (currently `4.034964084625244`; an earlier draft of this PRD cited `4.42494`, which does not match the repository artifacts — see `docs/DEPLOYED_MODEL.md`), provided it matches the loaded model and configuration.
- Record timestamps and whether each score exceeds the threshold.
- Preserve raw scores for further inspection.
- Return a clear error if the artifacts are missing or incompatible.

The threshold must not be recalibrated on the uploaded flight. A flight with insufficient valid windows must return an insufficient-data status instead of a misleading normal-flight verdict.

### FR-04: Anomaly event generation

The system must group overlapping or consecutive flagged windows into event intervals using a documented, configurable grouping rule.

Each event should contain:

- Event ID and flight ID.
- Start and end timestamps.
- Number of flagged windows.
- Peak, mean and threshold-relative anomaly scores.
- Per-feature reconstruction errors or residuals, where supported by the inference implementation.
- Relevant signal summaries.
- Data-quality warnings and nearby unflagged context.

The system must distinguish between an individual flagged window and a grouped anomaly event. Event scores must be aggregated using an explicitly defined method.

### FR-05: Results dashboard

The results page must include:

- Flight summary and processing status.
- Total valid windows, flagged windows and event count.
- An interactive anomaly-score timeline with the threshold.
- Event list with start time, end time and peak score.
- Sensor telemetry plots for the available IMU, ATT and BARO streams.
- A detailed view of each event, including its relevant telemetry and reconstruction errors.
- Visible data-quality warnings.

Users must be able to select an event and inspect the evidence associated with it.

### FR-06: JSON report generation

The application must produce a downloadable JSON report containing the detector configuration, flight summary, window-level results, grouped events and data-quality warnings.

The report should follow a versioned schema. A representative structure is:

```
{
  "schema_version": "1.0",
  "flight_id": "flight-001",
  "analysis_status": "completed",
  "detector": {
    "model": "LSTMAE",
    "variant": "conservative",
    "feature_count": 12,
    "window_length": 30,
    "stride": 15,
    "threshold": 4.034964084625244
  },
  "summary": {
    "total_windows": 0,
    "flagged_windows": 0,
    "anomaly_events": 0
  },
  "events": [],
  "data_quality_warnings": [],
  "ai_analysis": null
}
```

The zero counts and empty arrays are placeholders for the schema, not predicted model output. The final report must populate these fields with actual inference results.

The report should also provide enough provenance to identify the model version, feature schema and threshold used. Large raw telemetry arrays should not be embedded in the default report; the UI can retain the telemetry for event inspection.

### FR-07: GPT-4o mini diagnostic analysis

The application must allow a user to request an AI interpretation of a completed detector report.

The LLM input must include:

- Flight metadata and analysis summary.
- Grouped anomaly events.
- Actual scores and threshold.
- Relevant sensor values and residuals.
- Data-quality warnings.
- Relevant baseline comparisons, when available.

The LLM response should use a structured format containing:

- Executive summary.
- Possible anomaly explanations.
- Ranked fault hypotheses, where evidence supports them.
- Supporting and contradictory evidence.
- Alternative explanations.
- Confidence or uncertainty, clearly described as a qualitative assessment unless calibrated.
- Suggested inspection or further analysis.
- An explicit insufficient-evidence response when appropriate.

The AI must not fabricate measurements, change the detector's outputs, or assert a specific fault solely because an anomaly was detected.

### FR-08: Conversational assistant

The application must provide a chat panel associated with the currently selected flight.

Users should be able to ask questions such as:

- "Why was this event flagged?"
- "Which signals changed the most?"
- "Compare the two largest anomaly events."
- "Could this be a sensor issue rather than a motor issue?"
- "What additional evidence would help distinguish these explanations?"
- "Summarize this flight for an engineering report."

The assistant must:

- Use the current flight's report and available telemetry evidence as context.
- Retain relevant conversation history for follow-up questions within the active session.
- Distinguish measured facts from hypotheses.
- Say when the available data cannot support an answer.
- Avoid treating earlier AI-generated claims as verified sensor evidence.

The initial version does not require persistent accounts or cross-session chat history.

### FR-09: Export and session management

Users must be able to:

- Download the detector JSON report.
- Download the AI diagnostic report separately.
- Revisit results while the current application session remains active.
- Start a fresh analysis without mixing data from another flight.

### FR-10: Error handling and operational feedback

The UI must clearly handle:

- Missing or malformed CSV files.
- Missing sensor columns.
- Invalid or incompatible timestamps.
- No overlapping telemetry intervals.
- Insufficient valid windows.
- Incompatible model artifacts.
- OpenAI API errors, timeouts or rate limits.
- Invalid LLM responses.
- Failed JSON exports.

If the LLM API fails, the detector report must remain available. The system must never require a successful AI call to complete anomaly scoring.

## 7. User interface requirements

The interface should feel like a modern engineering analysis workspace: minimal, data-focused and easy to navigate.

Proposed navigation

New analysis

Upload files and inspect validation status.

Flight analysis

Telemetry charts, score timeline and events.

AI diagnostics

Explanations and flight-specific conversation.

Reports

JSON downloads and diagnostic summaries.

Design principles:

- Minimal visual clutter and consistent spacing.
- Clear hierarchy between flight status, anomalies and supporting details.
- Charts with readable axes, timestamps and thresholds.
- Expandable event details instead of showing every raw value at once.
- Explicit loading, success, warning and error states.
- Desktop-first layout, with basic responsive behavior for smaller screens.

## 8. Non-functional requirements

| Category        | Requirement                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Reproducibility | Same input and artifact versions should produce the same detector output, subject to deterministic inference settings. |
| Performance     | Processing time must be measured on representative flight files; no unsupported latency guarantee is assumed.          |
| Reliability     | Detector results and JSON export remain accessible if AI analysis fails.                                               |
| Security        | API keys remain server-side and are excluded from source control and exports.                                          |
| Data isolation  | A flight's telemetry and chat context must not leak into another flight's session.                                     |
| Data integrity  | Raw uploaded files remain unchanged; preprocessing operates on working copies.                                         |
| Maintainability | Preprocessing, inference, event generation, LLM integration and UI are separate modules.                               |
| Traceability    | Reports identify the model configuration, feature schema and threshold used.                                           |
| Safety          | AI explanations are advisory and must not directly control UAV behavior.                                               |

## 9. Proposed technical architecture

\#chatgpt-mermaid-\_r_1sd\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_1sd\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_1sd\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_1sd\_ .error-icon{fill:rgb(48, 48, 48);}#chatgpt-mermaid-\_r_1sd\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_1sd\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_1sd\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_1sd\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_1sd\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_1sd\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1sd\_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1sd\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_1sd\_ p{margin:0;}#chatgpt-mermaid-\_r_1sd\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_1sd\_ .label text,#chatgpt-mermaid-\_r_1sd\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .node rect,#chatgpt-mermaid-\_r_1sd\_ .node circle,#chatgpt-mermaid-\_r_1sd\_ .node ellipse,#chatgpt-mermaid-\_r_1sd\_ .node polygon,#chatgpt-mermaid-\_r_1sd\_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .rough-node .label text,#chatgpt-mermaid-\_r_1sd\_ .node .label text,#chatgpt-mermaid-\_r_1sd\_ .image-shape .label,#chatgpt-mermaid-\_r_1sd\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_1sd\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .rough-node .label,#chatgpt-mermaid-\_r_1sd\_ .node .label,#chatgpt-mermaid-\_r_1sd\_ .image-shape .label,#chatgpt-mermaid-\_r_1sd\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_1sd\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_1sd\_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1sd\_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1sd\_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-\_r_1sd\_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_1sd\_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1sd\_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1sd\_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-\_r_1sd\_ .cluster rect{fill:rgb(48, 48, 48);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(48, 48, 48);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_1sd\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1sd\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_1sd\_ .icon-shape,#chatgpt-mermaid-\_r_1sd\_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_1sd\_ .icon-shape p,#chatgpt-mermaid-\_r_1sd\_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-\_r_1sd\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_1sd\_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1sd\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_1sd\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_1sd\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_1sd\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_1sd\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_1sd\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_1sd\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1sd\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_1sd\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1sd\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_1sd\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_1sd\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_1sd\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_1sd\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .node rect,#chatgpt-mermaid-\_r_1sd\_ .node circle,#chatgpt-mermaid-\_r_1sd\_ .node ellipse,#chatgpt-mermaid-\_r_1sd\_ .node polygon,#chatgpt-mermaid-\_r_1sd\_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_1sd\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_1sd\_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_1sd\_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_1sd\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1sd\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Streamlit UIUpload & ValidationTelemetry PreprocessorLSTM Autoencoder InferenceAnomaly Event BuilderVersioned JSON ReportResults DashboardOpenAI GPT-4o miniStructured DiagnosticAnalysisFlight-specific ChatJSON ExportDiagnostic Export

### Suggested project structure

```
uav-sentinel/
├── app.py
├── pages/
│   ├── 1_Flight_Analysis.py
│   ├── 2_AI_Diagnostics.py
│   └── 3_Reports.py
├── src/
│   ├── upload.py
│   ├── validation.py
│   ├── preprocessing_adapter.py
│   ├── detector.py
│   ├── event_builder.py
│   ├── report_schema.py
│   ├── llm_service.py
│   └── chat_service.py
├── uav_model/
│   ├── lstm_autoencoder.pth
│   ├── inference.py
│   ├── preprocessing.py
│   ├── model_config.json
│   ├── feature_schema.json
│   ├── scaler.json
│   └── threshold_config.json
├── tests/
├── outputs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

This is the proposed application layout. Existing model files should be reused rather than regenerated; adapt the directory structure to their actual dependencies.

## 10. Acceptance criteria

The MVP is considered complete when all of the following pass:

- A user can upload three compatible telemetry files or a supported ZIP.
- Invalid files produce actionable validation messages.
- The existing 12-feature model loads with its matching scaler and threshold.
- The pipeline respects sessions, gaps, feature order and the expected window shape.
- Inference results are reproducible for a fixed input and configuration.
- Flagged windows are grouped into documented anomaly events.
- The dashboard displays the anomaly timeline and event details.
- The JSON report is valid, versioned and downloadable.
- GPT-4o mini receives actual detector evidence and returns a structured interpretation.
- Users can ask follow-up questions about the selected flight.
- AI failure does not destroy or block detector results.
- The application handles the tested normal, multi-session, short-flight and poor-quality-data scenarios.
- API keys and sensitive configuration are not exposed in the UI, source code or reports.

## 11. Delivery plan and scope boundaries

The implementation will be divided into five phases, to be planned separately:

1. Application foundation and telemetry upload.
2. Existing preprocessing and LSTM inference integration.
3. Event generation, dashboard and JSON exports.
4. GPT-4o mini integration and conversational chat.
5. End-to-end testing, hardening and documentation.

The first release should run locally and use the existing trained model. Cloud deployment, authentication, multi-user access, real-time telemetry streaming and automated UAV control are explicitly deferred.

## 12. Success metrics

The first version will track measurable software and workflow outcomes:

- Successful processing rate for compatible uploaded flights.
- Number of valid windows processed and anomaly events generated.
- Consistency of detector output across repeated runs.
- Time required to complete analysis on representative flight files.
- Percentage of reports passing JSON schema validation.
- LLM response validity and the presence of evidence-backed explanations.
- User ability to inspect an event, understand its supporting evidence and export the result.

No target for fault-detection accuracy or diagnostic precision is claimed until suitable ground-truth evaluation is available.

Product decision: Build a local-first Streamlit application around the existing conservative LSTM Autoencoder, then add the event layer, structured JSON, GPT-4o mini analysis and chat. The detector remains the source of anomaly scores; the LLM is the evidence-based interpretation layer.

This PRD defines the product scope and acceptance criteria. The five-phase implementation plan can now be developed one phase at a time, starting with Phase 1 when you ask.