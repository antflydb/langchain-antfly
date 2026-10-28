"""Regenerate the small redistributable fixture; no network/model calls."""

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_text((48, 55), "Orchard Operations - Q3 Review", fontsize=22)
    page.insert_textbox(
        pymupdf.Rect(48, 85, 560, 180),
        "This synthetic report is a redistributable test fixture. Orchard ships apples and pears. "
        "The operations owner is Mira Chen. The escalation window is 24 hours. "
        "The Q3 shipment target is 120 crates. Figures contain additional visual evidence.",
        fontsize=12,
    )
    page.insert_text((48, 215), "Shipment table", fontsize=16)
    for row, text in enumerate(
        [
            "Quarter       Apples       Pears",
            "Q1                 40                20",
            "Q2                 60                30",
            "Q3                 80                40",
        ]
    ):
        page.insert_text((60, 248 + row * 27), text, fontsize=13)
    for number, label in [(2, "Quarterly shipment chart"), (3, "Escalation system diagram")]:
        image = Image.new("RGB", (900, 520), "white")
        draw = ImageDraw.Draw(image)
        if number == 2:
            draw.text((35, 25), "Quarterly shipments (crates)", fill="black", font_size=30)
            for x, value, name in [(150, 60, "Q1"), (370, 90, "Q2"), (590, 120, "Q3")]:
                draw.rectangle((x, 430 - value * 2.5, x + 110, 430), fill="#246c9f")
                draw.text((x + 25, 445), name, fill="black", font_size=25)
                draw.text((x + 25, 390 - value * 2.5), str(value), fill="black", font_size=25)
            # This detail is deliberately excluded from stored summary proxies.
            draw.ellipse((745, 45, 805, 105), fill="#df831e")
            draw.text((720, 120), "Audit marker", fill="black", font_size=20)
        else:
            draw.text((35, 25), "Escalation route", fill="black", font_size=30)
            for x, text, color in [
                (45, "Sensor", "#b6d8ed"),
                (335, "Controller", "#e5ddac"),
                (625, "Pager", "#bcdfc5"),
            ]:
                draw.rounded_rectangle((x, 180, x + 220, 300), radius=18, fill=color, outline="black", width=3)
                draw.text((x + 30, 220), text, fill="black", font_size=28)
            for x in [270, 560]:
                draw.line((x, 240, x + 60, 240), fill="black", width=7)
                draw.polygon([(x + 60, 240), (x + 42, 228), (x + 42, 252)], fill="black")
            draw.text((280, 365), "Threshold: 85 degrees C", fill="black", font_size=28)
        image_path = destination / f"figure-{number - 1}.png"
        image.save(image_path)
        page = pdf.new_page(width=612, height=792)
        page.insert_text((48, 55), f"Figure {number - 1}: {label}", fontsize=19)
        page.insert_image(pymupdf.Rect(30, 140, 582, 459), filename=str(image_path))
        page.insert_text((48, 740), "Orchard Operations | synthetic fixture", fontsize=10)
    pdf.set_metadata({"title": "Orchard Operations Q3 Review", "author": "Antfly integration test fixture"})
    pdf.save(destination / "orchard-report.pdf", garbage=4, deflate=True, no_new_id=True)
    pdf.close()


if __name__ == "__main__":
    generate(Path(__file__).parent / "fixtures")
