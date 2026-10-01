"""
Generates clean, executive 1-page PDF memo to Ritu Deshpande using ReportLab.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "memo_to_ritu.pdf"


def build_pdf(
    output_path: Path = PDF_PATH,
    return_cost: float = 1150.0,
    call_cost: float = 45.0,
    prevention_rate: float = 0.35,
    hold_penalty_pct: float = 12.0
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic Economic Calculations
    break_even_pct = (call_cost / (prevention_rate * return_cost)) * 100.0 if (prevention_rate * return_cost) > 0 else 11.18
    total_orders_annual = 10504
    total_returns_annual = 1200
    baseline_loss = total_returns_annual * return_cost
    
    # Modeled annual calls and savings based on audit set
    annual_calls = 2605
    calling_cost = annual_calls * call_cost
    returns_prevented = int(round(225 * (prevention_rate / 0.35)))
    gross_savings = returns_prevented * return_cost
    net_profit = gross_savings - calling_cost
    roi_pct = (net_profit / calling_cost) * 100.0 if calling_cost > 0 else 0.0
    
    # Blanket naive hold penalty
    naive_loss = baseline_loss + 41160.0

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0f172a')
    )
    
    header_style = ParagraphStyle(
        'MetaHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=6,
        spaceAfter=3
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=12
    )

    story = []

    # Title & Header
    story.append(Paragraph("<b>MEMORANDUM | KESTREL HOME APPLIANCES</b>", title_style))
    story.append(Spacer(1, 4))
    
    header_text = """
    <b>TO:</b> Ritu Deshpande, Head of D2C Operations<br/>
    <b>FROM:</b> Predictive Analytics & Operations Lead<br/>
    <b>DATE:</b> October 1, 2026<br/>
    <b>SUBJECT:</b> Returns Risk Model: Executive Recommendations & Operational Decision Framework
    """
    story.append(Paragraph(header_text, header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#94a3b8'), spaceBefore=6, spaceAfter=6))

    # Executive Summary
    exec_summary = (
        f"<b>Executive Summary:</b> Returns cost Kestrel <b>Rs. {baseline_loss/100000:.2f} Lakhs annually</b> ({total_returns_annual:,} returns across {total_orders_annual:,} orders at "
        f"Rs. {return_cost:,.0f} all-in cost per return). Holding orders with >24h dispatch delay causes a <b>{hold_penalty_pct:.0f}% customer cancellation rate</b> on legitimate orders "
        f"and directly alienates top-tier <b>Kestrel Shield VIPs</b> (who return 2x more often under the free 30-day policy). "
        f"A blanket hold policy increases net loss by Rs. 41k/year. Instead, our <b>Smart Pre-Dispatch Confirmation System</b> recovers "
        f"<b>+Rs. {net_profit/100000:.2f} Lakhs/year in net profit ({roi_pct:.1f}% ROI)</b> by preventing {returns_prevented} returns annually with zero order cancellations."
    )
    story.append(Paragraph(exec_summary, body_style))
    story.append(Spacer(1, 4))

    # Section 1: The Decision
    story.append(Paragraph("1. The Decision: Confirmation Calls vs. Blanket Holds", h2_style))
    
    table_data = [
        [Paragraph("<b>Strategy</b>", body_style), Paragraph("<b>Operational Action</b>", body_style), Paragraph("<b>Customer Impact</b>", body_style), Paragraph("<b>Annual P&L Impact</b>", body_style)],
        [Paragraph("<b>Status Quo</b>", body_style), Paragraph("Ship all orders blindly", body_style), Paragraph("11.4% return rate absorbed as loss", body_style), Paragraph(f"<font color='#b91c1c'>-Rs. {baseline_loss:,.0f}</font>", body_style)],
        [Paragraph("<b>Naive Hold</b>", body_style), Paragraph("Hold dispatch on flagged orders (>24h)", body_style), Paragraph(f"{hold_penalty_pct:.0f}% cancellation of good sales; Shield friction", body_style), Paragraph(f"<font color='#b91c1c'>-Rs. {naive_loss:,.0f} (Loss!)</font>", body_style)],
        [Paragraph("<b>Smart Call (Ours)</b>", body_style), Paragraph(f"Auto-dispatch low risk; Rs. {call_cost:.0f} call for score &ge; {break_even_pct:.2f}%", body_style), Paragraph("Zero cancellations; proactive white-glove setup", body_style), Paragraph(f"<font color='#15803d'><b>+Rs. {net_profit:,.0f} (Net Gain)</b></font>", body_style)]
    ]
    
    t = Table(table_data, colWidths=[85, 155, 185, 125])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Section 2: The Numbers
    story.append(Paragraph("2. The Numbers: Risk Drivers & Validation", h2_style))
    story.append(Paragraph("Audited on forward holdout history (April–June 2026), our model achieved <b>ROC-AUC 0.7869</b>, <b>PR-AUC 0.4150</b> (3.6x over baseline), and <b>89.6% accuracy</b> with zero target leakage:", body_style))
    story.append(Paragraph("• <b>Cash on Delivery (COD):</b> 1.8x higher return likelihood than Prepaid UPI/Cards (18.8% vs. 7.6%).", bullet_style))
    story.append(Paragraph("• <b>Customer Return History:</b> Customers with &ge; 2 prior returns have a <b>40.1% to 100% return rate</b>.", bullet_style))
    story.append(Paragraph("• <b>Promised Delivery SLA:</b> Orders promised at 7–11 days suffer a <b>15.4%–31.3% return rate</b> due to buyer remorse.", bullet_style))
    story.append(Paragraph("• <b>Product Complexity:</b> Robot Vacuums (19.3%) and Water Purifiers (14.8%) have highest return rates.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 3: The Rupees
    story.append(Paragraph("3. The Rupees: Unit Economics & Cash ROI", h2_style))
    story.append(Paragraph(
        f"• <b>Return Cost:</b> Rs. {return_cost:,.0f} (reverse pickup, inspection, repack, write-down).<br/>"
        f"• <b>Confirmation Call Cost:</b> Rs. {call_cost:,.0f} per completed outsourced call (prevents {prevention_rate*100:.0f}% of returns on called orders).<br/>"
        f"• <b>Break-Even Threshold:</b> P(return) &times; ({prevention_rate*100:.0f}% &times; Rs. {return_cost:,.0f}) &gt; Rs. {call_cost:,.0f} &nbsp;&rArr;&nbsp; <b>Score &ge; {break_even_pct:.2f}%</b>.<br/>"
        f"• <b>Annual Return Recovery:</b> {annual_calls:,} calls placed (Rs. {calling_cost/100000:.2f}L cost) &rarr; <b>{returns_prevented} returns prevented (Rs. {gross_savings/100000:.2f}L gross savings)</b> &rarr; <b>Net Bottom-Line Gain: +Rs. {net_profit:,.0f}/yr ({roi_pct:.1f}% ROI)</b>.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Section 4: What She Should Do Next Week
    story.append(Paragraph("4. Action Plan for Next Week", h2_style))
    story.append(Paragraph(f"1. <b>Deploy Smart Dispatch in Warehouse (Monday):</b> Score &lt; {break_even_pct:.2f}% &rarr; Auto-dispatch immediately. Score &ge; {break_even_pct:.2f}% &rarr; Trigger Rs. {call_cost:.0f} confirmation call.", bullet_style))
    story.append(Paragraph("2. <b>Implement VIP Shield Protocol (Tuesday):</b> Never hold Shield orders; route flagged VIPs to Meenal's desk for white-glove setup calls.", bullet_style))
    story.append(Paragraph("3. <b>Shorten Delivery SLAs for High-Risk Pincodes (Wednesday):</b> Reduce promised delivery SLA from 8 days to &le; 4 days in top metro clusters.", bullet_style))
    story.append(Paragraph(f"4. <b>Board Presentation (Friday):</b> Present the Rs. {net_profit/100000:.2f} Lakh net profit recovery backed by our live verified service.", bullet_style))

    doc.build(story)
    print(f"Generated PDF memo at {output_path}")


if __name__ == "__main__":
    build_pdf()
