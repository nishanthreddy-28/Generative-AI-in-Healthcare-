"""Medical report assistant: local extraction, reviewed facts and sourced explanations."""

import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import hmac
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from healthcare_app.analysis import build_analysis, export_markdown, input_fingerprint, answer_question, attach_simulation
from healthcare_app.documents import extract_document, DocumentError, ocr_capabilities
from healthcare_app.knowledge import KnowledgeBase
from healthcare_app.records import extract_findings
from healthcare_app.history import extract_events
from healthcare_app.summary import render_summary, synthesize_summary

st.set_page_config(page_title="ClearCare | Medical reports", page_icon="🩺", layout="wide")
st.markdown("""<style>
.stApp { background: #f5f7f8; }
h1, h2, h3 { color: #123b40; }
[data-testid="stSidebar"] { background: #e8f0f0; }
[data-testid="stMetric"] { background: white; padding: 16px; border-radius: 12px; border: 1px solid #dce7e7; }
.stButton > button[kind="primary"] { background: #176f73; border-color: #176f73; }
</style>""", unsafe_allow_html=True)


def authenticate():
    token = os.environ.get("APP_ACCESS_TOKEN", "")
    public = os.environ.get("APP_PUBLIC_DEPLOYMENT", "0") == "1"
    if public and not token:
        st.error("Public mode requires APP_ACCESS_TOKEN. Configure authentication before starting this deployment.")
        st.stop()
    if token and not st.session_state.get("authenticated"):
        st.title("ClearCare")
        with st.form("sign_in"):
            entered = st.text_input("Access token", type="password", max_chars=256)
            if st.form_submit_button("Sign in"):
                if hmac.compare_digest(entered, token):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access token was not accepted.")
        st.stop()


authenticate()


@st.cache_resource(max_entries=2)
def load_knowledge(semantic_model=None):
    # Public knowledge only: patient uploads and findings are never globally cached.
    return KnowledgeBase(semantic_model=semantic_model)


def clear_reports():
    for key in list(st.session_state):
        if key.startswith(("report_", "review_", "simulation_")) or key in {"analysis", "chat_messages", "llm_plan", "llm_summary", "api_key", "llm_consent"}:
            del st.session_state[key]
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1


def clear_key():
    st.session_state.api_key = ""
    st.session_state.llm_consent = False
    st.session_state.pop("llm_plan", None)
    st.session_state.pop("llm_summary", None)


try:
    kb = load_knowledge(os.environ.get("APP_SEMANTIC_MODEL") or None)
except Exception:
    st.warning("Configured offline embeddings could not load. Using the explicitly labelled lexical hybrid fallback; no model was downloaded.")
    kb = load_knowledge()
st.sidebar.title("🩺 ClearCare")
st.sidebar.caption("Understand the report. Keep the evidence.")
public = os.environ.get("APP_PUBLIC_DEPLOYMENT", "0") == "1"
research_enabled = os.environ.get("APP_ENABLE_RESEARCH", "0" if public else "1") == "1"
workspaces = ["My medical reports"] + (["My glucose simulation", "Diabetes research"] if research_enabled else [])
workspace = st.sidebar.radio("Workspace", workspaces, key="workspace")
st.sidebar.caption(kb.retrieval_method)
st.sidebar.button("Clear reports and API key", on_click=clear_reports, width="stretch")
st.sidebar.caption("Uploads remain in this session. Local OCR uses temporary files that are removed after reading. No API calls occur unless you explicitly enable one.")
if st.session_state.get("authenticated"):
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()


def show_reports():
    st.title("Understand your medical report")
    st.write("Upload a report, review what was read, and explore explanations backed by medical sources. One report is enough to get started.")
    st.info("Educational research application. Review findings with your clinician. It does not diagnose conditions, prescribe treatment or assess emergencies.")
    with st.form("report_input"):
        uploads = st.file_uploader("Upload PDF, image or text reports", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True, key=f"report_upload_{st.session_state.get('upload_generation', 0)}")
        pasted = st.text_area("Or paste report text", height=140, max_chars=100_000, key="report_pasted", placeholder="Report date: 2026-10-01\nHemoglobin 13.5 g/dL 12.0-16.0\nHbA1c 6.2 % 4.0-5.6")
        submitted = st.form_submit_button("Read report", type="primary")
    if st.button("Try a fictional example"):
        clear_reports()
        example = "Report date: 2026-10-01\nHbA1c 8.2 % 4.0-5.6\nFasting glucose 146 mg/dL 70-99\nHemoglobin 13.5 g/dL 12.0-16.0\nCreatinine 0.9 mg/dL 0.6-1.2\nTSH 2.1 mIU/L 0.4-4.0\nDiagnosis: Type 1 diabetes; no nephropathy\nMedication: Insulin glargine 12 units nightly"
        st.session_state.report_documents = [extract_document(example.encode(), "fictional_example.txt").to_dict()]
        st.session_state.report_originals = {st.session_state.report_documents[0]["digest"]: example.encode()}
        st.rerun()
    if submitted:
        st.session_state.pop("analysis", None)
        st.session_state.pop("report_documents", None)
        st.session_state.pop("report_originals", None)
        st.session_state.pop("llm_plan", None)
        st.session_state.pop("llm_summary", None)
        st.session_state.chat_messages = []
        try:
            if uploads and pasted.strip():
                raise DocumentError("Choose uploaded files or pasted text for this reading. Add history as separate uploaded reports.")
            if uploads:
                if len(uploads) > 5 or sum(f.size for f in uploads) > 20 * 1024 * 1024:
                    raise DocumentError("Use at most five files and 20 MB in total; each file must be at most 10 MB.")
                with st.spinner("Reading reports locally…"):
                    documents = [extract_document(f.getvalue(), f.name, f"Report {i + 1}").to_dict() for i, f in enumerate(uploads)]
                if len({d["digest"] for d in documents}) != len(documents):
                    raise DocumentError("The same report was uploaded more than once. Remove duplicates.")
            elif pasted.strip():
                documents = [extract_document(pasted.encode("utf-8"), "pasted_report.txt").to_dict()]
            else:
                raise DocumentError("Upload a report or paste its text first.")
            st.session_state.report_documents = documents
            st.session_state.report_originals = {d["digest"]: (f.getvalue() if uploads else pasted.encode("utf-8")) for d, f in zip(documents, uploads or [None])}
        except DocumentError as exc:
            st.error(str(exc))
    documents = st.session_state.get("report_documents")
    if not documents:
        st.caption("Supported: common numeric labs and explicit diagnosis/medication statements in English. Documented mentions require review; medical images and unrestricted narrative require clinical interpretation.")
        with st.expander("Reading capabilities on this computer"):
            st.json(ocr_capabilities())
        return
    st.subheader("1 · Review the extraction")
    st.write("Compare every result with the original. Correct the text below, including units and ranges. Use YYYY-MM-DD for dates. A suggested format is: `Test name  value  unit  reference range`.")
    reviewed = []
    for document in documents:
        new_document = {**document, "pages": []}
        with st.expander(f"{document['label']} · {document['name']} · {document['method']}", expanded=True):
            original = st.session_state.get("report_originals", {}).get(document["digest"])
            if original and Path(document["name"]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(original, caption="Original upload — compare it with the transcription", width="stretch")
            for warning in document["warnings"]:
                st.warning(warning)
            for page in document["pages"]:
                key = f"review_text_{document['label']}_{document['digest']}_{page['page']}"
                text = st.text_area(f"Page {page['page']} — editable transcription", value=page["text"], height=180, max_chars=100_000, key=key)
                new_document["pages"].append({"page": page["page"], "text": text})
        reviewed.append(new_document)
    fingerprint = input_fingerprint(reviewed)
    preview, parser_warnings = extract_findings(reviewed)
    event_preview, event_warnings = extract_events(reviewed)
    st.caption(f"Recognized {len(preview)} supported numeric results. Unrecognized text is not interpreted; review the complete report with your clinician.")
    if preview:
        st.dataframe(pd.DataFrame(preview)[["id", "name", "value", "unit", "ref_low", "ref_high", "date", "status", "source", "page"]], hide_index=True, width="stretch")
    if event_preview:
        st.write("Documented clinical mentions — verify names, assertion status and recorded doses against the original. Document date does not establish onset or current use.")
        st.dataframe(pd.DataFrame(event_preview)[["id", "kind", "name", "status", "documented_dose", "date", "source", "page"]], hide_index=True, width="stretch")
    for event in event_preview:
        for issue in event["issues"]:
            st.warning(f"{event['id']}: {issue}")
    for warning in parser_warnings:
        st.warning(warning)
    confirmed = st.checkbox("I checked the values, units, reference ranges, dates, diagnosis assertions and medication names/dose text against the original reports.", key=f"review_confirm_{fingerprint}")
    same_patient = True
    if len(reviewed) > 1:
        same_patient = st.checkbox("All reports belong to the same patient.", key=f"review_same_patient_{fingerprint}")
    if st.button("Explain reviewed findings", type="primary", disabled=not (confirmed and same_patient)):
        try:
            st.session_state.analysis = build_analysis(reviewed, kb, user_reviewed=True)
            st.session_state.chat_messages = []
            st.session_state.pop("llm_plan", None)
            st.session_state.pop("llm_summary", None)
        except ValueError as exc:
            st.error(str(exc))
    analysis = st.session_state.get("analysis")
    if not analysis or analysis["fingerprint"] != fingerprint:
        if analysis:
            st.info("The extraction changed. Review it again and regenerate the explanation.")
        return
    simulation = st.session_state.get("simulation_v2_result")
    simulation_settings = st.session_state.get("simulation_v2_settings")
    if simulation and simulation_settings and simulation_settings[2] == fingerprint and simulation_settings[1] == st.session_state.get("simulation_v2_count"):
        analysis = attach_simulation(analysis, simulation)
    st.subheader("2 · Your findings and the evidence")
    columns = st.columns(3)
    columns[0].metric("Reviewed results", len(analysis["findings"]))
    columns[1].metric("Outside report range", sum(f["status"] in {"Above report range", "Below report range"} for f in analysis["findings"]))
    columns[2].metric("Dated comparisons", len(analysis["trends"]))
    if len({f["date"] for f in analysis["findings"] if f["date"]}) < 2:
        st.info("This is a snapshot. No trend or future forecast was inferred from this report.")
    for warning in analysis["warnings"]:
        st.warning(warning)
    by_id = {f["id"]: f for f in analysis["findings"]}
    order = st.session_state.get("llm_plan") or [{"finding_id": e["finding_id"], "evidence_id": (e["evidence_ids"][0] if e["evidence_ids"] else None)} for e in analysis["explanations"]]
    evidence_by_id = {d["id"]: d for d in analysis["evidence"]}
    tabs = st.tabs(["Explanation", "Timeline", "Questions", "Sources & downloads"])
    with tabs[0]:
        with st.expander("Patient summary — findings, attention, uncertainty and questions", expanded=True):
            st.markdown(render_summary(analysis, st.session_state.get("llm_summary")))
        if analysis["clinical_events"]:
            st.subheader("Documented diagnoses and medications")
        event_refs = {e["event_id"]: e["evidence_ids"] for e in analysis["event_explanations"]}
        for event in analysis["clinical_events"]:
            with st.expander(f"[{event['id']}] {event['name']} · {event['status']}"):
                st.code(event["quote"], language=None)
                st.write(f"{event['source']}, page {event['page']}, line {event['line']}. Document date: {event['date'] or 'unknown'}. {event['date_basis']}.")
                st.write(f"Dose text recorded: {event['documented_dose'] or 'not recorded'}. This is not a dose recommendation.")
                for eid in event_refs[event["id"]]:
                    source = evidence_by_id[eid]
                    st.write(source["content"])
                    st.markdown(f"[{eid} · {source['title']}]({source['url']})")
        for item in order:
            f = by_id[item["finding_id"]]
            with st.expander(f"{f['name']} · {f['comparator']}{f['value']:g} {f['unit']} · {f['status']}", expanded=True):
                st.write(f"[{f['id']}] {f['source']}, page {f['page']}, line {f['line']}. Date: {f['date'] or 'unknown'}.")
                st.code(f["quote"], language=None)
                if item["evidence_id"]:
                    d = evidence_by_id[item["evidence_id"]]
                    st.write(d["content"])
                    st.markdown(f"[{d['id']} · {d['title']}]({d['url']})")
        st.write(evidence_by_id["NLM-LAB"]["content"])
        st.markdown(f"[How to understand lab results]({evidence_by_id['NLM-LAB']['url']})")
        with st.expander("Optional LLM-assisted organization"):
            st.write("Optional paid calls compose the patient summary or organize lab findings. The model selects complete approved claims; the server preserves numbers, negation, uncertainty and citations. Free medical claims are rejected. Raw documents, quotations, names and exact dates are withheld; clinical facts remain sensitive.")
            from healthcare_app.llm import SUPPORTED_MODELS
            selected_model = st.selectbox("Optional LLM model", SUPPORTED_MODELS, key="llm_model")
            api_key = st.text_input("Your OpenAI API key", type="password", key="api_key", max_chars=256)
            consent = st.checkbox("I consent to sending reviewed numeric findings, recognized clinical mentions, linked CGM summary values and approved evidence to OpenAI. This may incur API charges.", key="llm_consent")
            st.button("Clear API key", on_click=clear_key)
            if st.button("Organize with LLM", disabled=not (api_key.strip() and consent and analysis["findings"])):
                from healthcare_app.llm import select_explanations, LLMError
                try:
                    with st.spinner("Requesting constrained organization…"):
                        st.session_state.llm_plan = select_explanations(analysis, api_key, consent, model=selected_model)
                    st.rerun()
                except LLMError as exc:
                    st.error(str(exc))
            if st.button("Compose patient summary with LLM", disabled=not (api_key.strip() and consent)):
                from healthcare_app.llm import LLMError
                try:
                    with st.spinner("Composing a summary from approved claims…"):
                        st.session_state.llm_summary = synthesize_summary(analysis, api_key, consent, model=selected_model)
                    st.rerun()
                except LLMError as exc:
                    st.error(str(exc))
    with tabs[1]:
        st.caption("Clinical events use document dates, not confirmed onset/start dates. Negated, family-history and uncertain mentions retain their status. All source entries remain visible.")
        if analysis["timeline"]:
            st.dataframe(pd.DataFrame(analysis["timeline"]), hide_index=True, width="stretch")
        if not analysis["trends"]:
            st.write("No reliable dated comparison is available. Add same-patient reports with unambiguous dates and matching units.")
        for trend in analysis["trends"]:
            st.write(trend["text"])
            points = [by_id[fid] for fid in trend["finding_ids"]]
            st.line_chart(pd.DataFrame({"date": pd.to_datetime([p["date"] for p in points]), trend["name"]: [p["value"] for p in points]}).set_index("date"))
    with tabs[2]:
        for q in analysis["questions"]:
            st.write("• " + q)
        for message in st.session_state.get("chat_messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
        question = st.chat_input("Ask about a test in your reports", max_chars=2000)
        if question:
            answer = answer_question(question, analysis, kb)
            st.session_state.setdefault("chat_messages", []).extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            st.session_state.chat_messages = st.session_state.chat_messages[-20:]
            st.rerun()
    with tabs[3]:
        st.caption("Reviewed NLM/NIDDK educational passages and a bounded ADA recommendation extract. " + kb.retrieval_method + ". Corpus coverage is limited; patient-similarity retrieval is not implemented.")
        for d in analysis["evidence"]:
            st.markdown(f"[{d['id']} · {d['title']}]({d['url']}) · Source checked {d['reviewed']}")
            if d.get("passage"):
                st.text(f"{d.get('locator', '')}: {d['passage']}")
        for limit in analysis["limits"]:
            st.write("• " + limit)
        st.download_button("Download explanation", render_summary(analysis, st.session_state.get("llm_summary")) + "\n\n" + export_markdown(analysis), "medical_report_explanation.md", "text/markdown")
        st.download_button("Download structured findings", json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), "medical_report_findings.json", "application/json")
    if research_enabled:
        st.info("Future glucose simulation needs separate measured CGM history with sufficient consecutive coverage. The legacy model additionally requires complete multimodal features. A single report remains useful for explanation.")
        def open_simulation():
            st.session_state.workspace = "My glucose simulation"
        st.button("Check my history for glucose simulation", on_click=open_simulation)


@st.cache_resource(max_entries=1)
def load_upload_models():
    from healthcare_app.research import ResearchPipeline
    return ResearchPipeline(load_cohort=False)


@st.cache_resource(max_entries=1)
def load_corrected_system():
    from healthcare_app.trajectory_v2 import CorrectedSimulator
    return CorrectedSimulator()


def show_measured_cgm():
    from healthcare_app.cgm import read_cgm_upload, CORE
    import hashlib
    st.write("Upload measured CGM data for the corrected CGM-only research release. Reports provide the separate clinical history and evidence explanation; medicines and lab values are not invented as forecasting inputs.")
    st.download_button("Download CGM template", "timestamp,glucose_mg_dl\n", "cgm_template.csv", "text/csv")
    with st.form("simulation_cgm_input"):
        upload = st.file_uploader("Measured single-patient CGM CSV", type=["csv"], key=f"simulation_cgm_upload_{st.session_state.get('upload_generation', 0)}")
        pasted = st.text_area("Or paste measured CGM CSV", max_chars=1_000_000, key="simulation_cgm_pasted", placeholder="timestamp,glucose_mg_dl\n2026-09-01 00:00:00,146")
        read = st.form_submit_button("Validate measured CGM")
    if read:
        st.session_state.pop("simulation_v2_result", None)
        st.session_state.pop("simulation_cgm_daily", None)
        try:
            if upload and pasted.strip():
                raise ValueError("Choose uploaded or pasted CGM, rather than both.")
            data = upload.getvalue() if upload else pasted.encode()
            daily, quality = read_cgm_upload(data)
            st.session_state.simulation_cgm_daily = daily
            st.session_state.simulation_cgm_quality = quality
            st.session_state.simulation_cgm_digest = hashlib.sha256(data).hexdigest()
        except ValueError as exc:
            st.error(str(exc))
    daily = st.session_state.get("simulation_cgm_daily")
    if daily is None:
        st.caption("Required: at least 30 consecutive days with 64 occupied 15-minute bins per day. Bins average actual observations; missing bins are never interpolated. Resolve conflicting timestamps. Use measured mg/dL and ISO local timestamps.")
        return
    digest = st.session_state.simulation_cgm_digest
    analysis = st.session_state.get("analysis")
    report_fingerprint = analysis["fingerprint"] if analysis else None
    st.dataframe(daily, hide_index=True, width="stretch")
    st.json(st.session_state.simulation_cgm_quality)
    confirm = st.checkbox("These are reviewed measurements from one person with type 1 diabetes in mg/dL. I understand this is an internally evaluated research simulation, not clinical guidance.", key=f"simulation_cgm_confirm_{digest}")
    same_patient = True
    if analysis:
        same_patient = st.checkbox("This CGM history belongs to the same person as my reviewed reports.", key=f"simulation_cgm_same_{digest}_{report_fingerprint}")
    count = st.slider("Corrected-release sensitivity paths", 10, 100, 20, 10, key="simulation_v2_count")
    if st.button("Run corrected glucose simulation", disabled=not (confirm and same_patient), type="primary"):
        st.session_state.pop("simulation_v2_result", None)
        st.session_state.pop("llm_summary", None)
        try:
            with st.spinner("Encoding measured glucose history and simulating locally…"):
                result = load_corrected_system().run(daily, count)
            st.session_state.simulation_v2_result = result
            st.session_state.simulation_v2_settings = (digest, count, report_fingerprint)
        except (OSError, ValueError, ImportError, RuntimeError, KeyError):
            st.error("The corrected release is unavailable, failed validation or could not accept this history. No legacy/cohort result was substituted. Check the release and source measurements.")
    result = st.session_state.get("simulation_v2_result")
    if not result or st.session_state.get("simulation_v2_settings") != (digest, count, report_fingerprint):
        return
    st.subheader("Measured history and possible future glucose patterns")
    st.write(f"Observed window: {result['window_dates'][0]} to {result['window_dates'][1]}. Future curves are research simulations; the flat line is a simple historical-mean comparator.")
    st.caption(f"Latest qualifying uploaded daily observation: {result['latest_available_daily_date']}. Simulation dates begin after the selected history window; they may include dates already in the past.")
    st.warning("Internal patient-held-out evaluation does not establish accuracy for you. No medicine, meal, illness or treatment-response model is included. Simulations do not estimate your clinical risk.")
    for column in CORE:
        observed = result["observed_window"][column]
        frame = pd.DataFrame({"Measured daily history": observed, "Research central path": result["point"][column], "Past-30-day mean comparator": result["baseline_point"][column]})
        st.subheader(column)
        st.line_chart(frame)
    with st.expander("Calibration, sensitivity paths, benchmarks and provenance"):
        st.write(f"Validation-selected algorithm: {result['selected_model']}. Live VAE glucose-pattern vector: {len(result['latent_z'])} dimensions.")
        st.write(result["metadata"]["interval"])
        st.write("Marginal bounds are separate from simulated paths. They do not form a joint feasible trajectory or individual clinical probabilities.")
        intervals = pd.concat([result["core_interval_low"].add_suffix("_low"), result["core_interval_high"].add_suffix("_high")], axis=1)
        st.dataframe(intervals, width="stretch")
        st.write(result["metadata"]["generation"])
        sample_frame = pd.DataFrame(result["simulations"][0], columns=result["columns"], index=result["point"].index)
        st.line_chart(sample_frame[["glucose_mean"]])
        st.json({k: result["metadata"][k] for k in ("selected_model", "validation_scores", "test_metrics", "patient_counts", "window_counts", "vae_test", "status")})
        st.caption("Artifact manifest fingerprint: " + result["artifact_fingerprint"])
    st.text(result["note"])
    st.download_button("Download corrected trajectory", result["point"].to_csv(), "corrected_research_trajectory.csv", "text/csv")
    st.download_button("Download corrected research note", result["note"], "corrected_research_note.txt", "text/plain")


def show_patient_simulation():
    from healthcare_app.model_inputs import daily_template, read_daily_upload, model_schema
    import hashlib
    st.title("My glucose simulation")
    input_format = st.radio("Input format", ["Measured CGM (corrected release)", "Legacy multimodal daily features"], key="simulation_format")
    if input_format == "Measured CGM (corrected release)":
        show_measured_cgm()
        return
    st.warning("Experimental Granada-trained models retain preprocessing leakage and poor calibration. This feature runs a technical simulation, not a validated forecast, disease risk estimate or treatment recommendation.")
    st.write("Provide one patient's complete daily model-feature history. At least 30 consecutive days are required. Report text and CGM-only measurements cannot fill the missing lab/profile inputs.")
    st.download_button("Download empty daily-feature template", daily_template(), "daily_feature_template.csv", "text/csv")
    with st.expander("Input contract and why a report alone is insufficient"):
        st.write("The template requires 50 numeric features in the existing model representation, plus date. Do not guess unknown diagnosis flags, use zeros for missing labs, or substitute a cohort patient's profile. Glucose is mg/dL; flags use the original binary convention. Lab features must already be generated by a reviewed compatible preprocessing pipeline; their original source-unit conventions are not independently validated here. This application does not auto-create these vectors from sparse reports.")
        st.json(model_schema()["input_cols"])
    with st.form("simulation_input"):
        upload = st.file_uploader("Single-patient daily-feature CSV", type=["csv"], key=f"simulation_upload_{st.session_state.get('upload_generation', 0)}")
        pasted = st.text_area("Or paste the daily-feature CSV", max_chars=1_000_000, key="simulation_pasted")
        read = st.form_submit_button("Validate daily history")
    if read:
        st.session_state.pop("simulation_result", None)
        st.session_state.pop("simulation_frame", None)
        try:
            if upload and pasted.strip():
                raise ValueError("Choose a CSV file or pasted CSV, rather than both.")
            data = upload.getvalue() if upload else pasted.encode()
            frame = read_daily_upload(data)
            st.session_state.simulation_frame = frame
            st.session_state.simulation_digest = hashlib.sha256(data).hexdigest()
        except ValueError as exc:
            st.error(str(exc))
    frame = st.session_state.get("simulation_frame")
    if frame is None:
        return
    digest = st.session_state.simulation_digest
    st.success(f"Input checks passed for {len(frame)} daily rows. This confirms format consistency, not clinical validity or source correctness.")
    st.dataframe(frame, hide_index=True, width="stretch")
    confirmed = st.checkbox("I verified this single patient's features, dates, missingness flags and training-unit conventions. I understand the simulations are experimental.", key=f"simulation_confirm_{digest}")
    count = st.slider("Sensitivity paths", 10, 100, 20, 10, key="simulation_count")
    if st.button("Simulate my glucose history", disabled=not confirmed, type="primary"):
        st.session_state.pop("simulation_result", None)
        try:
            with st.spinner("Loading trusted models and encoding your uploaded history locally…"):
                result = load_upload_models().run_uploaded_history(frame, count)
            st.session_state.simulation_result = result
            st.session_state.simulation_settings = (digest, count)
        except Exception as exc:
            if isinstance(exc, ValueError):
                st.error(str(exc))
            else:
                st.error("Local model inference failed. Check installed research dependencies and trusted artifacts; no prediction was substituted.")
    result = st.session_state.get("simulation_result")
    if result and st.session_state.get("simulation_settings") == (digest, count):
        st.write(result["input_source"])
        st.write(f"Observation window: {result['window_dates'][0]} to {result['window_dates'][1]}; latest available: {result['latest_observation']}.")
        st.caption("Neural trajectories and conditional synthetic states are separate model outputs. The flat persistence comparator simply repeats the final observation and is not a calibrated prediction.")
        trajectory = result["trajectory_simulation"]
        rows = pd.DataFrame(trajectory["p50"], columns=trajectory["target_cols"], index=pd.RangeIndex(1, len(trajectory["p50"]) + 1, name="Future day"))
        from healthcare_app.research import latest_contiguous_window
        observed = latest_contiguous_window(frame).iloc[-1]
        for column in rows:
            plot = rows[[column]].rename(columns={column: "Experimental neural central path"})
            plot["Last-observation comparator"] = float(observed[column])
            st.subheader(column)
            st.line_chart(plot)
        st.caption(trajectory["band_type"] + ". Spread is not displayed as a clinical probability or confidence interval.")
        with st.expander("Synthetic states and grounded research note"):
            st.dataframe(result["ctgan_samples"], hide_index=True, width="stretch")
            st.text(result["doctor_note"])
        st.download_button("Download my simulated trajectory", rows.to_csv(), "uploaded_research_trajectory.csv", "text/csv")
        st.download_button("Download my research note", result["doctor_note"], "uploaded_research_note.txt", "text/plain")


def show_research():
    st.title("Diabetes research workspace")
    st.warning("Experimental cohort models: preprocessing leakage and poorly calibrated forecast spread remain in the supplied weights. Outputs are for technical research demonstration, not treatment decisions. Uploaded lab reports are never substituted into these models.")
    consent = st.checkbox("I understand the model limitations and am authorized to access the supplied research cohort.")
    if not consent:
        st.write("This workspace needs the existing local datasets and model artifacts. It loads them only after you opt in.")
        return
    @st.cache_resource(max_entries=1)
    def load_research_system():
        from healthcare_app.research import ResearchPipeline
        return ResearchPipeline()
    try:
        with st.spinner("Loading trusted research artifacts…"):
            pipeline = load_research_system()
    except Exception:
        st.error("Research artifacts or dependencies are unavailable or failed validation. The report workspace remains available. Check the installation guide and model manifest.")
        return
    patient = st.selectbox("Research cohort patient", sorted(pipeline.df.Patient_ID.unique()))
    count = st.slider("Synthetic samples / sensitivity paths", 10, 100, 20, 10)
    if st.button("Run research simulation", type="primary"):
        try:
            with st.spinner("Encoding the latest consecutive window and running models…"):
                st.session_state.research_result = pipeline.run_inference_for_patient(patient, count)
                st.session_state.research_settings = (patient, count)
        except ValueError as exc:
            st.session_state.pop("research_result", None)
            st.error(str(exc))
    result = st.session_state.get("research_result")
    if not result or st.session_state.get("research_settings") != (patient, count):
        return
    st.write(f"Model observation window: {result['window_dates'][0]} to {result['window_dates'][1]}. Latest available observation: {result['latest_observation']}. Age at window start: {result['pat_profile']['age']:g}. CPU inference; fixed seed {result['seed']}.")
    tabs = st.tabs(["Daily trajectory", "Synthetic 30-day states", "Research notes", "Model limitations"])
    with tabs[0]:
        trajectory = result["trajectory_simulation"]
        st.caption(trajectory["band_type"] + ". Curves below show the constrained central trajectory; no calibrated risk band is displayed.")
        frame = pd.DataFrame(trajectory["p50"], columns=trajectory["target_cols"], index=pd.RangeIndex(1, len(trajectory["p50"]) + 1, name="Future day"))
        for column in frame:
            st.line_chart(frame[[column]])
        st.download_button("Download research trajectory", frame.to_csv(), "research_trajectory.csv", "text/csv")
    with tabs[1]:
        st.caption("The conditional tabular generator and daily GRU are separate models. Samples are constrained for presentation. Sensor-low counts are 30-day readings, not hypoglycemic episodes.")
        st.dataframe(result["ctgan_samples"], hide_index=True, width="stretch")
        st.download_button("Download synthetic states", result["ctgan_samples"].to_csv(index=False), "research_synthetic_states.csv", "text/csv")
    with tabs[2]:
        st.text(result["doctor_note"])
        st.download_button("Download research note", result["doctor_note"], "research_note.txt", "text/plain")
    with tabs[3]:
        st.write(result["model_status"])
        st.write("Outputs are from live VAE, conditional generator and GRU inference. No static latent fallback is used. Each phase uses the same consecutive window. Display constraints do not improve the learned model, prove privacy, or establish external clinical validity.")


if workspace == "My medical reports":
    show_reports()
elif workspace == "My glucose simulation":
    show_patient_simulation()
else:
    show_research()
