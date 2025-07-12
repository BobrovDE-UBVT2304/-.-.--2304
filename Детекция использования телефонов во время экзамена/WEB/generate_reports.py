# generate_reports.py
from fpdf import FPDF

def generate(rows):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('ArialUnicode', '', 'arialmt.ttf', uni=True)
    pdf.set_font("ArialUnicode", size=12)

    pdf.cell(200, 10, txt="Отчёт по детекции телефонов", ln=True, align='C')
    pdf.ln(10)

    if not rows:
        pdf.cell(200, 10, txt="Нет данных за выбранный период", ln=True, align='L')
    else:
        for row in rows:
            pdf.cell(200, 10, txt=f"{row[1]} — {row[2]}", ln=True, align='L')

    output_path = "report.pdf"
    pdf.output(output_path)
    return output_path
