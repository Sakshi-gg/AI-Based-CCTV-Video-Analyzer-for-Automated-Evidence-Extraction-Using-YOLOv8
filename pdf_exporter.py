"""
pdf_exporter.py
Generates a complete forensic PDF report for a CCTV analysis case.
Includes: cover page, video metadata, entity tracking table,
          evidence frames (2-up grid), anomaly log, and AI summary.
"""

import os
import io
import time
import sqlite3
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from database import DB_PATH

# ── Page geometry ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 2 * cm
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1A3C6E")
MID_BLUE   = colors.HexColor("#2E6099")
LIGHT_BLUE = colors.HexColor("#D6E4F7")
WHITE      = colors.white
BLACK      = colors.black
GRAY       = colors.HexColor("#555555")
LIGHT_GRAY = colors.HexColor("#F2F2F2")
ALT_ROW    = colors.HexColor("#EBF2FB")
RED        = colors.HexColor("#C62828")
GREEN      = colors.HexColor("#2E7D32")
ORANGE     = colors.HexColor("#E65100")


# ── Styles ─────────────────────────────────────────────────────────────────────
def build_styles():
    return {
        "body": ParagraphStyle(
            "body", fontSize=9, fontName="Helvetica",
            textColor=BLACK, alignment=TA_JUSTIFY,
            spaceAfter=5, leading=13
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=8, fontName="Helvetica",
            textColor=GRAY, alignment=TA_CENTER, spaceAfter=3
        ),
        "caption_high": ParagraphStyle(
            "caption_high", fontSize=8, fontName="Helvetica-Bold",
            textColor=RED, alignment=TA_CENTER, spaceAfter=3
        ),
        "footer_note": ParagraphStyle(
            "footer_note", fontSize=7, fontName="Helvetica",
            textColor=GRAY, alignment=TA_CENTER, spaceAfter=0
        ),
        # Section header text style — background applied via Table wrapper
        "sh_text": ParagraphStyle(
            "sh_text", fontSize=12, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_LEFT
        ),
    }


def section_header(title):
    """
    Renders a bold white-on-dark-blue section header using a Table.
    More reliable than ParagraphStyle backColor in ReportLab.
    """
    styles = build_styles()
    t = Table([[Paragraph(title, styles["sh_text"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def truncate_path(path, max_len=60):
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def frame_to_reportlab_image(frame_rgb, width, height):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        return None
    img = Image(io.BytesIO(buffer.tobytes()), width=width, height=height)
    img.hAlign = 'CENTER'
    return img


def generate_pdf_report(evidence_log, video_metadata, case_id,
                         last_analysis_rate, ai_summary=None,
                         output_path=None):
    if output_path is None:
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        output_dir = "video_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"forensic_report_{timestamp}.pdf")

    styles  = build_styles()
    content = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=R_MARGIN, leftMargin=L_MARGIN,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Forensic Evidence Report",
        author="AI-Based CCTV Video Analyzer"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    content.append(Spacer(1, 2.5 * cm))

    # Main title banner
    banner = Table([["FORENSIC EVIDENCE REPORT"]], colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 22),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))
    content.append(banner)
    content.append(Spacer(1, 0.25 * cm))

    # Subtitle banner
    sub_table = Table(
        [["AI-Based CCTV Video Analyzer for Automated Evidence Extraction"]],
        colWidths=[CONTENT_W]
    )
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), MID_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(sub_table)
    content.append(Spacer(1, 1 * cm))

    # Case detail rows
    filename   = video_metadata.get("Filename", "Unknown")
    date       = video_metadata.get("Processing Date", time.strftime('%Y-%m-%d %H:%M:%S'))
    duration   = video_metadata.get("Duration", "N/A")
    resolution = video_metadata.get("Resolution (W x H)", "N/A")

    label_style = ParagraphStyle("lbl", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=DARK_BLUE, alignment=TA_LEFT)
    value_style = ParagraphStyle("val", fontSize=9, fontName="Helvetica",
                                  textColor=BLACK, alignment=TA_LEFT)

    details_data = [
        [Paragraph("Case ID",         label_style), Paragraph(str(case_id),          value_style)],
        [Paragraph("Video File",       label_style), Paragraph(filename,               value_style)],
        [Paragraph("Duration",         label_style), Paragraph(duration,               value_style)],
        [Paragraph("Resolution",       label_style), Paragraph(resolution,             value_style)],
        [Paragraph("Analysis Date",    label_style), Paragraph(date,                   value_style)],
        [Paragraph("Evidence Frames",  label_style), Paragraph(str(len(evidence_log)), value_style)],
        [Paragraph("Analysis Rate",    label_style), Paragraph(last_analysis_rate,     value_style)],
    ]
    details_table = Table(details_data, colWidths=[5 * cm, CONTENT_W - 5 * cm])
    details_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT_BLUE),
        ("BACKGROUND",    (1, 0), (1, -1), WHITE),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT_BLUE, ALT_ROW]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#BBCDE0")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    content.append(details_table)
    content.append(Spacer(1, 1 * cm))

    # Confidential banner
    conf_table = Table(
        [["⚠  CONFIDENTIAL — FOR LAW ENFORCEMENT USE ONLY  ⚠"]],
        colWidths=[CONTENT_W]
    )
    conf_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFEBEE")),
        ("TEXTCOLOR",     (0, 0), (-1, -1), RED),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 1, RED),
    ]))
    content.append(conf_table)
    content.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: VIDEO METADATA
    # ══════════════════════════════════════════════════════════════════════════
    content.append(section_header("1.  Video Metadata"))
    content.append(Spacer(1, 0.3 * cm))

    meta_rows = [["Property", "Value"]]
    for key, value in video_metadata.items():
        val = truncate_path(str(value), 65) if key == "File Path" else str(value)
        meta_rows.append([str(key), val])
    meta_rows.append(["Analysis Rate", last_analysis_rate])

    meta_table = Table(meta_rows, colWidths=[5.5 * cm, CONTENT_W - 5.5 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 1), (0, -1),  DARK_BLUE),
        ("TEXTCOLOR",     (1, 1), (1, -1),  BLACK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    content.append(meta_table)
    content.append(Spacer(1, 0.6 * cm))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: ENTITY TRACKING SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    content.append(section_header("2.  Tracked Entity Summary"))
    content.append(Spacer(1, 0.3 * cm))

    try:
        conn = sqlite3.connect(DB_PATH)
        entity_rows = conn.execute("""
            SELECT entity_label, class_type,
                   MIN(video_time) as first_seen,
                   MAX(video_time) as last_seen,
                   COUNT(*) as appearances
            FROM entity_tracks
            WHERE case_id=?
            GROUP BY entity_label
            ORDER BY appearances DESC
        """, (case_id,)).fetchall()
        conn.close()
    except Exception:
        entity_rows = []

    if entity_rows:
        col_w = CONTENT_W / 5
        track_data = [["Entity ID", "Type", "First Seen", "Last Seen", "Appearances"]]
        for row in entity_rows:
            track_data.append([str(v) for v in row])

        track_table = Table(track_data, colWidths=[col_w] * 5)
        track_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, ALT_ROW]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        content.append(track_table)
    else:
        content.append(Paragraph(
            "No entity tracking data found for this case. "
            "Ensure Car or Person checkboxes were selected during analysis.",
            styles["body"]
        ))
    content.append(Spacer(1, 0.6 * cm))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: EVIDENCE FRAMES (2-up grid)
    # ══════════════════════════════════════════════════════════════════════════
    content.append(PageBreak())
    content.append(section_header("3.  Evidence Frames"))
    content.append(Spacer(1, 0.3 * cm))
    content.append(Paragraph(
        f"Total evidence frames captured: {len(evidence_log)}. "
        f"Frames are displayed in pairs. Captions in red indicate high anomaly scores (≥ 0.70).",
        styles["body"]
    ))
    content.append(Spacer(1, 0.3 * cm))

    IMG_W = (CONTENT_W - 0.4 * cm) / 2
    IMG_H = IMG_W * 0.72

    frames_to_show = evidence_log[:20]
    pairs = [frames_to_show[i:i + 2] for i in range(0, len(frames_to_show), 2)]

    for pair in pairs:
        img_cells = []
        cap_cells = []

        for entry in pair:
            try:
                img = frame_to_reportlab_image(entry['frame_rgb'], IMG_W, IMG_H)
                img_cells.append(img if img else Paragraph("(frame unavailable)", styles["caption"]))
            except Exception:
                img_cells.append(Paragraph("(frame unavailable)", styles["caption"]))

            anomaly   = entry.get('anomaly_score', 0.0)
            cap_style = styles["caption_high"] if anomaly >= 0.7 else styles["caption"]
            flag      = " ⚠ HIGH" if anomaly >= 0.7 else ""
            cap_cells.append(Paragraph(
                f"Time: {entry['timestamp']}  |  Det: {entry['detection_count']}  "
                f"|  Anomaly: {anomaly:.2f}{flag}",
                cap_style
            ))

        while len(img_cells) < 2:
            img_cells.append(Paragraph("", styles["caption"]))
            cap_cells.append(Paragraph("", styles["caption"]))

        frame_row = Table(
            [[img_cells[0], img_cells[1]],
             [cap_cells[0], cap_cells[1]]],
            colWidths=[IMG_W, IMG_W],
            rowHeights=[IMG_H, 0.5 * cm]
        )
        frame_row.setStyle(TableStyle([
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]))
        content.append(KeepTogether(frame_row))
        content.append(Spacer(1, 0.2 * cm))

    if len(evidence_log) > 20:
        content.append(Paragraph(
            f"Note: {len(evidence_log) - 20} additional frames not embedded. "
            "Use Save Report & Frames to access all annotated images on disk.",
            styles["footer_note"]
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: ANOMALY SCORE LOG
    # ══════════════════════════════════════════════════════════════════════════
    anomaly_entries = [e for e in evidence_log if e.get('anomaly_score', 0) > 0]
    if anomaly_entries:
        content.append(PageBreak())
        content.append(section_header("4.  Anomaly Score Log"))
        content.append(Spacer(1, 0.3 * cm))
        content.append(Paragraph(
            "Frames with anomaly scores above 0.45 indicate statistically abnormal motion "
            "detected by the LSTM Autoencoder module. Scores above 0.70 are flagged as HIGH priority.",
            styles["body"]
        ))
        content.append(Spacer(1, 0.3 * cm))

        anom_data = [["Frame #", "Timestamp", "Detections", "Anomaly Score", "Priority"]]
        for i, entry in enumerate(anomaly_entries):
            score    = entry.get('anomaly_score', 0.0)
            priority = "HIGH" if score >= 0.7 else ("MEDIUM" if score >= 0.45 else "LOW")
            anom_data.append([
                str(i + 1),
                entry['timestamp'],
                str(entry['detection_count']),
                f"{score:.3f}",
                priority
            ])

        col_w_a = CONTENT_W / 5
        anom_styles_list = [
            ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ]
        for row_i, entry in enumerate(anomaly_entries, start=1):
            score = entry.get('anomaly_score', 0.0)
            if score >= 0.7:
                anom_styles_list.append(("TEXTCOLOR", (3, row_i), (4, row_i), RED))
                anom_styles_list.append(("FONTNAME",  (3, row_i), (4, row_i), "Helvetica-Bold"))
                anom_styles_list.append(("BACKGROUND",(4, row_i), (4, row_i), colors.HexColor("#FFEBEE")))
            elif score >= 0.45:
                anom_styles_list.append(("TEXTCOLOR", (3, row_i), (4, row_i), ORANGE))

        anom_table = Table(anom_data, colWidths=[col_w_a] * 5)
        anom_table.setStyle(TableStyle(anom_styles_list))
        content.append(anom_table)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: AI INVESTIGATION SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    if ai_summary and ai_summary.strip():
        content.append(PageBreak())
        sec_num = "5" if anomaly_entries else "4"
        content.append(section_header(f"{sec_num}.  AI Investigation Summary"))
        content.append(Spacer(1, 0.3 * cm))
        content.append(Paragraph(
            "The following narrative was automatically generated by an AI forensic assistant "
            "based on the entity tracking data. It is provided as an analytical aid only and "
            "does not constitute a legal determination.",
            styles["body"]
        ))
        content.append(Spacer(1, 0.3 * cm))
        content.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
        content.append(Spacer(1, 0.2 * cm))
        for para in ai_summary.strip().split('\n\n'):
            para = para.strip()
            if para:
                content.append(Paragraph(para, styles["body"]))
                content.append(Spacer(1, 0.2 * cm))

    # ── Footer ─────────────────────────────────────────────────────────────────
    content.append(Spacer(1, 1 * cm))
    content.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
    content.append(Spacer(1, 0.2 * cm))
    content.append(Paragraph(
        f"Generated by AI-Based CCTV Video Analyzer  |  "
        f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Case ID: {case_id}",
        styles["footer_note"]
    ))

    doc.build(content)
    return output_path
