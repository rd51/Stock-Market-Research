from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

OUTPUT = "analysis_for_paper/stock_market_analysis_summary.pdf"

def build_pdf():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Stock Market AI Analytics — Analysis Summary", styles['Title']))
    story.append(Spacer(1, 12))

    # Executive summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_summary = (
        "This document summarizes the recent work performed on the Stock Market AI Analytics "
        "dashboard during the development session. It captures key fixes, UX changes, data-loading behavior, "
        "and recommendations to include in a research paper."
    )
    story.append(Paragraph(exec_summary, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Key changes
    story.append(Paragraph("Key Changes Applied", styles['Heading2']))
    key_changes = (
        "- Implemented robust sidebar filters: date-range handling, regime selector, model multiselect, and "
        "toggle to show model predictions.\n"
        "- Fixed `st.date_input` handling to accept both single-date and (start,end) tuples safely.\n"
        "- Repaired truncated and garbled lines in the code base that caused runtime/compile errors.\n"
        "- Removed the Model Comparison page per user request and updated navigation and quick-links accordingly.\n"
        "- Restored utility helpers `create_header()` and `create_status_metrics()` after removal introduced NameErrors.\n"
        "- Added an automatic loader (temporarily) to find model comparison summary files; later removed when page deleted."
    )
    story.append(Paragraph(key_changes.replace("\\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 12))

    # Data and environment
    story.append(Paragraph("Data and Environment", styles['Heading2']))
    data_env = (
        "- Sample/stationary data is read from `stationary_data.csv` when available; fallback synthetic data used otherwise.\n"
        "- Prediction history is loaded from `data/cache/prediction_history.csv` if present.\n"
        "- Models are loaded from the `models/` folder; dummy placeholders are used when models are absent.\n"
        "- Development environment: Python virtualenv, Streamlit app served locally on port 8501."
    )
    story.append(Paragraph(data_env.replace("\\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 12))

    # Observations
    story.append(Paragraph("Observations", styles['Heading2']))
    observations = (
        "- The dashboard compiles and runs locally after the fixes; Streamlit served successfully at http://localhost:8501.\n"
        "- Several deprecation warnings about Streamlit API (e.g., `use_container_width`) were observed; they are non-blocking but should be updated for future compatibility.\n"
        "- Model comparison functionality depended on external CSV/JSON artifacts; absence leads to empty-state messages.\n"
        "- Tests and some analysis scripts may still reference the removed Model Comparison page; update tests if needed."
    )
    story.append(Paragraph(observations.replace("\\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 12))

    # Recommendations for paper
    story.append(Paragraph("Recommendations for Research Paper", styles['Heading2']))
    recs = (
        "1. Document the data provenance and preprocessing pipeline (`stationary_data.csv`, `preprocessing_pipeline.py`).\n"
        "2. Include a short section describing UI changes and how filters affect analysis (date-range/regime/model selection).\n"
        "3. Note the robustness fixes (date handling, missing-file fallbacks, helper restorations) as part of engineering rigor.\n"
        "4. For model comparison results, specify required output schemas and include example CSV/JSON in the repository to reproduce figures.\n"
        "5. Replace deprecated Streamlit APIs to future-proof reproducibility.\n"
    )
    story.append(Paragraph(recs.replace("\\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 12))

    # Appendix: Files modified
    story.append(Paragraph("Appendix: Files Modified / Reviewed", styles['Heading2']))
    appendix = (
        "- `app.py` — main entry, sidebar filters, header/status helpers, data/model loading.\n"
        "- `pages/04_model_comparison.py` — inspected and deleted per request.\n"
        "- `pages/01_home.py`, `temp_home.py` — quick-links updated to reflect page removal.\n"
        "- Other scripts reviewed: `comparison_report.py`, `model_evaluation.py`, `results_logger.py`."
    )
    story.append(Paragraph(appendix.replace("\\n", "<br/>"), styles['BodyText']))

    doc.build(story)

if __name__ == '__main__':
    import os
    os.makedirs('analysis_for_paper', exist_ok=True)
    try:
        build_pdf()
        print('PDF generated:', OUTPUT)
    except Exception as e:
        print('Failed to generate PDF:', e)
