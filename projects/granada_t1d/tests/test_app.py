from pathlib import Path
import os
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_paste_review_explain_chat_edit_clear():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.text_area(key="report_pasted").set_value("Report date: 2026-10-01\nGlucose 146 mg/dL 70-99\nHbA1c 8.2 % 4-6")
    button(app, "Read report").click().run()
    assert not app.exception
    assert button(app, "Explain reviewed findings").disabled
    next(c for c in app.checkbox if c.label.startswith("I checked")).check().run()
    button(app, "Explain reviewed findings").click().run()
    assert not app.exception and len(app.session_state.analysis["findings"]) == 2
    assert len(app.tabs) == 4 and len(app.metric) == 3
    assert len(app.get("download_button")) == 2
    assert button(app, "Organize with LLM").disabled
    app.text_input(key="api_key").set_value("fake-session-key").run()
    assert os.environ.get("OPENAI_API_KEY") is None
    button(app, "Clear API key").click().run()
    assert app.text_input(key="api_key").value == ""
    app.chat_input[0].set_value("Explain glucose").run()
    assert not app.exception
    assert "146" in app.session_state.chat_messages[-1]["content"]
    review = next(a for a in app.text_area if a.label.startswith("Page 1"))
    review.set_value("Report date: 2026-10-01\nGlucose 100 mg/dL 70-99").run()
    assert button(app, "Explain reviewed findings").disabled
    assert len(app.tabs) == 0  # old results hidden until re-reviewed
    next(c for c in app.checkbox if c.label.startswith("I checked")).check().run()
    button(app, "Explain reviewed findings").click().run()
    assert app.session_state.analysis["findings"][0]["value"] == 100
    button(app, "Clear reports and API key").click().run()
    assert not app.exception and "analysis" not in app.session_state
    assert app.text_area(key="report_pasted").value == ""


def test_app_invalid_input_and_fictional_example():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    button(app, "Read report").click().run()
    assert not app.exception and app.error
    button(app, "Try a fictional example").click().run()
    assert not app.exception
    next(c for c in app.checkbox if c.label.startswith("I checked")).check().run()
    button(app, "Explain reviewed findings").click().run()
    assert not app.exception and len(app.session_state.analysis["findings"]) == 5


def test_sessions_do_not_share_report_or_key():
    a = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    a.text_area(key="report_pasted").set_value("Glucose 146 mg/dL 70-99")
    button(a, "Read report").click().run()
    next(c for c in a.checkbox if c.label.startswith("I checked")).check().run()
    button(a, "Explain reviewed findings").click().run()
    a.text_input(key="api_key").set_value("private-key-one").run()
    b = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert "analysis" not in b.session_state
    assert "api_key" not in b.session_state
    assert b.text_area(key="report_pasted").value == ""


def test_public_mode_fails_closed_without_access_token(monkeypatch):
    monkeypatch.setenv("APP_PUBLIC_DEPLOYMENT", "1")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception and app.error
    assert len(app.text_area) == 0


def test_auth_wrong_and_correct_token(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "test-access-token")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.text_input[0].set_value("wrong")
    button(app, "Sign in").click().run()
    assert app.error and not app.text_area
    app.text_input[0].set_value("test-access-token")
    button(app, "Sign in").click().run()
    assert not app.exception and app.text_area
    button(app, "Sign out").click().run()
    assert not app.text_area and not app.exception


def test_research_is_explicit_opt_in():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value("Diabetes research").run()
    assert not app.exception and len(app.checkbox) == 1
    assert not app.selectbox


def test_research_live_ui_and_stale_settings():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.radio[0].set_value("Diabetes research").run()
    app.checkbox[0].check().run()
    assert not app.exception
    app.selectbox[0].set_value("LIB193266").run()
    app.slider[0].set_value(10).run()
    button(app, "Run research simulation").click().run()
    assert not app.exception and len(app.tabs) == 4
    assert len(app.get("download_button")) == 3
    assert app.session_state.research_result["patient_id"] == "LIB193266"
    app.slider[0].set_value(20).run()
    assert not app.exception and not app.tabs  # prior outputs cannot masquerade as new settings


def test_clinical_history_summary_and_simulation_navigation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    button(app, "Try a fictional example").click().run()
    next(c for c in app.checkbox if c.label.startswith("I checked")).check().run()
    button(app, "Explain reviewed findings").click().run()
    assert not app.exception and len(app.session_state.analysis["clinical_events"]) == 3
    assert len(app.session_state.analysis["timeline"]) == 8
    assert button(app, "Compose patient summary with LLM").disabled
    button(app, "Check my history for glucose simulation").click().run()
    assert not app.exception and app.radio(key="workspace").value == "My glucose simulation"
    app.radio(key="simulation_format").set_value("Legacy multimodal daily features").run()
    assert app.text_area(key="simulation_pasted") and len(app.get("download_button")) == 1


def test_uploaded_history_ui_inference_and_stale_result():
    import pandas as pd
    from healthcare_app.research import latest_contiguous_window
    from healthcare_app.model_inputs import model_schema
    data = pd.read_csv(ROOT / "Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv")
    frame = latest_contiguous_window(data[data.Patient_ID == "LIB193266"])[["date"] + model_schema()["input_cols"]]
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.radio(key="workspace").set_value("My glucose simulation").run()
    app.radio(key="simulation_format").set_value("Legacy multimodal daily features").run()
    app.text_area(key="simulation_pasted").set_value(frame.to_csv(index=False))
    button(app, "Validate daily history").click().run()
    assert not app.exception and app.success
    assert button(app, "Simulate my glucose history").disabled
    app.checkbox[0].check().run()
    button(app, "Simulate my glucose history").click().run()
    assert not app.exception and app.session_state.simulation_result["patient_id"] == "Uploaded patient"
    assert len(app.get("download_button")) == 3
    app.slider(key="simulation_count").set_value(10).run()
    assert not app.exception and len(app.get("download_button")) == 1
    app.text_area(key="simulation_pasted").set_value("date,glucose_mean\n2026-10-01,146\n")
    button(app, "Validate daily history").click().run()
    assert app.error and "simulation_result" not in app.session_state


def test_measured_cgm_ui_validation_and_corrected_simulation():
    import numpy as np
    import pandas as pd
    timestamps = pd.date_range("2026-09-01", periods=30 * 96, freq="15min")
    frame = pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"), "glucose_mg_dl": 150 + 30 * np.sin(np.arange(len(timestamps)) / 8)})
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.radio(key="workspace").set_value("My glucose simulation").run()
    app.text_area(key="simulation_cgm_pasted").set_value(frame.to_csv(index=False))
    button(app, "Validate measured CGM").click().run()
    assert not app.exception and app.checkbox
    assert button(app, "Run corrected glucose simulation").disabled
    app.checkbox[0].check().run()
    button(app, "Run corrected glucose simulation").click().run()
    assert not app.exception and "simulation_v2_result" in app.session_state
    assert app.session_state.simulation_v2_result["point"].shape == (60, 6)
    assert len(app.get("download_button")) == 3
    app.slider(key="simulation_v2_count").set_value(10).run()
    assert len(app.get("download_button")) == 1
    button(app, "Clear reports and API key").click().run()
    assert not app.exception and "simulation_v2_result" not in app.session_state
    assert "simulation_cgm_daily" not in app.session_state
    assert app.text_area(key="simulation_cgm_pasted").value == ""
    assert app.session_state.upload_generation == 1
