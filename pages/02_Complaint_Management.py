"""
Complaint Management Page for GrievanceRoute.
Handles validated citizen submissions, live AI triage prediction,
SQLite persistence, and full lifecycle status tracking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.services.complaint_service import ComplaintService
from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.data.generator import PUNE_ZONES
from grievanceroute.components.voice_input import render_voice_input


st.set_page_config(page_title="Complaint Management — GrievanceRoute", page_icon="📝", layout="wide")
apply_custom_theme()

render_hero(
    title="Civic Complaint Intake & Management",
    subtitle="Submit grievances via speech or text, trigger automatic multilingual NLP triage, and manage operational lifecycle records.",
    badge="Unit I & III • Form & Persistence"
)

repo = GrievanceRepository()
complaint_svc = ComplaintService(repo=repo)

tab_submit, tab_explore = st.tabs(["📝 File New Complaint", "🔍 Search & Audit Trail"])

with tab_submit:
    st.subheader("Citizen Complaint Submission Form")
    st.markdown("Citizens can report civic issues by **speaking in English, Hindi, or Marathi** or typing directly. The AI pipeline categorizes the grievance, predicts urgency, and persists the record into SQLite.")

    # Multilingual Voice-to-Text Input Widget
    with st.expander("🎙️ **Voice-to-Text Complaint Input (Click to speak in English / हिन्दी / मराठी)**", expanded=True):
        st.caption("Select language, press the microphone button, and speak. Transcribed text automatically populates the description field below for review.")
        voice_event = render_voice_input(key="citizen_voice_mic")

        if voice_event:
            if voice_event.get("is_clear"):
                st.session_state["complaint_text_input"] = ""
                st.session_state["last_transferred_voice"] = ""
                st.rerun()
            elif voice_event.get("transcript"):
                transcript = voice_event["transcript"].strip()
                last_trans = st.session_state.get("last_transferred_voice")
                if transcript and transcript != last_trans:
                    st.session_state["last_transferred_voice"] = transcript
                    st.session_state["complaint_text_input"] = transcript
                    lang_name = voice_event.get("language_name", "speech")
                    st.toast(f"Transcribed {lang_name} populated into description field!", icon="🎙️")
                    st.rerun()

    with st.form("complaint_form", clear_on_submit=False):
        c_text = st.text_area(
            "Complaint Description *",
            value=st.session_state.get("complaint_text_input", ""),
            placeholder="Describe the civic issue in detail (e.g. Deep pothole near Balgandharva chowk causing bike accidents...), or speak into the microphone above.",
            height=130,
            help="Multilingual input supported (English, Hindi, Marathi). You can freely edit, correct, or augment the recognized speech here before final submission.",
        )


        col_z, col_lat, col_lon = st.columns(3)
        with col_z:
            zone_choice = st.selectbox("Pune Service Zone *", list(PUNE_ZONES.keys()))
        default_coords = PUNE_ZONES[zone_choice]
        with col_lat:
            lat_input = st.number_input("Latitude", value=default_coords["lat"], format="%.6f")
        with col_lon:
            lon_input = st.number_input("Longitude", value=default_coords["lon"], format="%.6f")

        col_w, col_ch1, col_ch2, col_ch3 = st.columns(4)
        with col_w:
            weather_choice = st.selectbox("Weather Condition", ["Clear", "Rainy", "Cloudy", "Stormy"])
        with col_ch1:
            is_monsoon = st.checkbox("Monsoon Season", value=False)
        with col_ch2:
            is_low_lying = st.checkbox("Low-Lying / Flood Risk", value=default_coords["low_lying"])
        with col_ch3:
            is_repeat = st.checkbox("Repeat Complaint", value=False)

        submit_btn = st.form_submit_button("📨 Submit & Triage Complaint", type="primary", use_container_width=True)

    if submit_btn:
        success, message, record = complaint_svc.submit_complaint(
            complaint_text=c_text,
            latitude=lat_input,
            longitude=lon_input,
            zone=zone_choice,
            weather=weather_choice,
            monsoon=is_monsoon,
            low_lying_area=is_low_lying,
            repeat_complaint=is_repeat,
        )

        if not success:
            st.error(f"Validation Error: {message}")
        else:
            st.success(f"Success! {message}")

            # Show Triage Decision Card
            st.markdown("### 🤖 Automatic AI Triage Decision")
            if record.get("safety_override"):
                st.error(f"🚨 **Safety Rule Triggered:** {record['safety_warning']}")

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric("Predicted Category", record["complaint_type"].title())
            with col_res2:
                conf_pct = record.get("confidence", 1.0) * 100
                st.metric("Model Confidence", f"{conf_pct:.1f}%")
            with col_res3:
                urg = record["urgency_score"]
                st.metric("Urgency Level", f"{urg} / 5", "Priority Score")
            with col_res4:
                st.metric("Assigned Crew", record["crew_required"])

            if record.get("detected_language"):
                st.info(f"🌐 **Language Detected:** `{record['detected_language']}` • Native text stored in SQLite; multilingual triage applied.")

            st.caption(f"Estimated Resolution Time: **{record.get('actual_resolution_hours', 24.0):.1f} hours**. Record stored in SQLite `complaints` table.")


with tab_explore:
    st.subheader("Historical Complaints Database Explorer")
    all_df = repo.get_all_complaints(limit=1000)

    if all_df.empty:
        st.info("No records in database yet.")
    else:
        # Search & Filter controls
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            search_query = st.text_input("Filter by text keyword:", placeholder="e.g. pothole, wire, pipe...")
        with f2:
            status_filter = st.multiselect("Status:", ["All"] + list(all_df["status"].unique()), default=["All"])
        with f3:
            cat_filter = st.multiselect("Category:", ["All"] + list(all_df["complaint_type"].unique()), default=["All"])
        with f4:
            zone_filter = st.multiselect("Zone:", ["All"] + list(all_df["zone"].unique()), default=["All"])

        filtered_df = all_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["complaint_text"].str.contains(search_query, case=False, na=False)]
        if "All" not in status_filter and status_filter:
            filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
        if "All" not in cat_filter and cat_filter:
            filtered_df = filtered_df[filtered_df["complaint_type"].isin(cat_filter)]
        if "All" not in zone_filter and zone_filter:
            filtered_df = filtered_df[filtered_df["zone"].isin(zone_filter)]

        st.markdown(f"Showing **{len(filtered_df)}** of **{len(all_df)}** complaints:")
        st.dataframe(
            filtered_df[["complaint_id", "complaint_text", "complaint_type", "zone", "urgency_score", "status", "crew_required", "timestamp"]],
            use_container_width=True,
            height=380
        )

        st.divider()
        st.subheader("📜 Complaint Lifecycle Audit Trail")
        selected_audit_id = st.selectbox("Select Complaint ID to view status history:", all_df["complaint_id"].tolist()[:100])
        with repo.get_connection() as conn:
            hist_df = pd.read_sql_query("SELECT * FROM status_history WHERE complaint_id = ? ORDER BY changed_at DESC", conn, params=(selected_audit_id,))

        if not hist_df.empty:
            st.dataframe(hist_df[["id", "complaint_id", "old_status", "new_status", "changed_at", "notes"]], use_container_width=True)
        else:
            st.info("No status change log for this complaint.")
