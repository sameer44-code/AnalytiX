import os
import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from xml.sax.saxutils import escape as xml_escape

def generate_pdf_report(
    output_pdf_path: str,
    dataset_name: str,
    quality_analysis: Dict[str, Any],
    cleaning_summary: Dict[str, Any],
    audit_logs: List[Dict[str, Any]]
) -> str:
    """
    Generates a formal, executive-ready enterprise PDF report using ReportLab.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Define corporate palette
    c_navy = colors.HexColor("#0f172a")
    c_blue = colors.HexColor("#1e3a8a")
    c_accent = colors.HexColor("#0284c7")
    c_slate = colors.HexColor("#475569")
    c_light_bg = colors.HexColor("#f8fafc")
    c_border = colors.HexColor("#cbd5e1")
    c_white = colors.white

    # Custom styles
    header_style = ParagraphStyle(
        'DocHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=c_navy
    )
    subtitle_style = ParagraphStyle(
        'DocSubHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_slate
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_blue,
        spaceBefore=14,
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=c_navy,
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=body_style,
        fontSize=8,
        leading=10
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=bold_body,
        fontSize=8,
        leading=10
    )
    finding_title_style = ParagraphStyle(
        'FindingTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_blue
    )

    story = []

    # 1. Header Banner
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%B %d, %Y - %H:%M UTC')
    story.append(Paragraph("ENTERPRISE AI DATA INTELLIGENCE REPORT", header_style))
    story.append(Paragraph(f"Dataset Target: <b>{dataset_name}</b> | Generated: {now_utc}", subtitle_style))
    story.append(Paragraph("Classification: CONFIDENTIAL / INTERNAL BUSINESS USE ONLY", ParagraphStyle('ClassLabel', parent=subtitle_style, fontSize=8, textColor=colors.HexColor("#b91c1c"))))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=2, spaceAfter=14))

    # 2. Executive Summary Metrics Table
    story.append(Paragraph("1. Executive Summary & Health Index", h1_style))
    story.append(Paragraph(
        "A rigorous multi-dimensional quality assessment was conducted across all records to evaluate completeness, "
        "uniqueness, structural consistency, and statistical anomaly exposure.",
        body_style
    ))
    story.append(Spacer(1, 8))

    q_score = quality_analysis.get("overall_quality_score", 0.0)
    completeness = quality_analysis.get("completeness_score", 0.0)
    uniqueness = quality_analysis.get("uniqueness_score", 0.0)
    consistency = quality_analysis.get("consistency_score", 0.0)
    validity = quality_analysis.get("validity_score", 0.0)

    rows_cnt = quality_analysis.get("total_rows", 0)
    cols_cnt = quality_analysis.get("total_columns", 0)
    missing_cells = quality_analysis.get("total_missing_cells", 0)
    missing_pct = quality_analysis.get("missing_cells_percentage", 0.0)
    dups_cnt = quality_analysis.get("duplicate_rows_count", 0)

    summary_data = [
        [
            Paragraph("Overall Quality Score", table_cell_bold),
            Paragraph(f"<b>{q_score} / 100</b>", table_cell_bold),
            Paragraph("Total Records", table_cell_bold),
            Paragraph(f"{rows_cnt:,}", table_cell),
        ],
        [
            Paragraph("Completeness Index", table_cell),
            Paragraph(f"{completeness}%", table_cell),
            Paragraph("Total Features", table_cell),
            Paragraph(f"{cols_cnt}", table_cell),
        ],
        [
            Paragraph("Uniqueness Index", table_cell),
            Paragraph(f"{uniqueness}%", table_cell),
            Paragraph("Missing Cells", table_cell),
            Paragraph(f"{missing_cells:,} ({missing_pct}%)", table_cell),
        ],
        [
            Paragraph("Consistency & Validity", table_cell),
            Paragraph(f"{consistency}% / {validity}%", table_cell),
            Paragraph("Duplicate Rows", table_cell),
            Paragraph(f"{dups_cnt:,}", table_cell),
        ],
    ]

    t_summary = Table(summary_data, colWidths=[130, 130, 130, 140])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TEXTCOLOR', (0, 0), (-1, -1), c_navy),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#e0f2fe")),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # 3. Data Processing Issues & Anomalies
    story.append(Paragraph("2. Data Quality Anomalies & Issue Register", h1_style))
    inconsistencies = quality_analysis.get("inconsistencies", [])
    outliers = quality_analysis.get("outliers", [])

    if inconsistencies or outliers:
        issue_rows = [[
            Paragraph("Column Target", table_cell_bold),
            Paragraph("Issue Classification", table_cell_bold),
            Paragraph("Affected Volume", table_cell_bold),
            Paragraph("Detailed Description", table_cell_bold)
        ]]
        for inc in inconsistencies[:6]:
            issue_rows.append([
                Paragraph(xml_escape(inc.get("column", "N/A")), table_cell_bold),
                Paragraph(f"<font color='#b91c1c'>{xml_escape(inc.get('issue_type', 'N/A'))}</font>", table_cell),
                Paragraph(str(inc.get("affected_count", "0")), table_cell),
                Paragraph(xml_escape(inc.get("description", "N/A")), table_cell),
            ])
        for out in outliers[:4]:
            issue_rows.append([
                Paragraph(out.get("column", "N/A"), table_cell_bold),
                Paragraph("<font color='#d97706'>Tail Outlier (IQR)</font>", table_cell),
                Paragraph(f"{out.get('outlier_count', 0)} ({out.get('outlier_percentage', 0)}%)", table_cell),
                Paragraph(f"Values outside [{out.get('lower_bound')}, {out.get('upper_bound')}].", table_cell),
            ])

        t_issues = Table(issue_rows, colWidths=[100, 120, 80, 230])
        t_issues.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_issues)
    else:
        story.append(Paragraph("No critical structural anomalies or high-severity inconsistencies detected.", body_style))

    story.append(Spacer(1, 14))

    # 4. User-Directed Cleaning Audit Trail
    story.append(Paragraph("3. Data Cleaning Execution Audit Trail", h1_style))
    story.append(Paragraph(
        "<b>Governing Principle:</b> In compliance with corporate data integrity protocols, all cleaning transformations "
        "were strictly authorized by the analyst. No automated alterations were applied without explicit user directive.",
        body_style
    ))
    story.append(Spacer(1, 6))

    if audit_logs:
        audit_table_data = [[
            Paragraph("Step", table_cell_bold),
            Paragraph("Column", table_cell_bold),
            Paragraph("Action", table_cell_bold),
            Paragraph("Decision Details & Rationale", table_cell_bold)
        ]]
        for log in audit_logs:
            audit_table_data.append([
                Paragraph(str(log.get("step_number", "-")), table_cell),
                Paragraph(str(log.get("target_column", "ALL")), table_cell_bold),
                Paragraph(str(log.get("action_type", "")).upper(), table_cell),
                Paragraph(f"<b>{xml_escape(str(log.get('decision_details', '')))}</b><br/><font color='#475569'>{xml_escape(str(log.get('rationale', '')))}</font>", table_cell),
            ])

        t_audit = Table(audit_table_data, colWidths=[35, 90, 80, 325])
        t_audit.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_audit)
    else:
        story.append(Paragraph("<i>No transformations applied yet. Dataset remains in pristine raw state.</i>", body_style))

    story.append(Spacer(1, 14))

    # 4. AI Analytical Recommendations
    story.append(Paragraph("4. AI Analytical Recommendations", h1_style))
    next_inv = quality_analysis.get("next_investigations", [])
    if next_inv:
        for item in next_inv:
            story.append(Paragraph(f"• <b>Investigation Target:</b> {xml_escape(str(item))}", body_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("• Continue longitudinal monitoring of core transactional volume.", body_style))

    # Build document
    doc.build(story)
    return output_pdf_path
