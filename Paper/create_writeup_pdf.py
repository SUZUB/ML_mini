"""Create the two-page ML mini-project write-up PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Paper" / "ML_Mini_Project_Writeup.pdf"


def build_pdf() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="Structural Damage Image Classification",
        author="ML Mini Project",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="WriteupTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4A5560"),
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=5,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11.1,
            spaceAfter=5,
            textColor=colors.HexColor("#202830"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=10.5,
            leftIndent=10,
            firstLineIndent=-6,
            spaceAfter=2,
            textColor=colors.HexColor("#202830"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#202830"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["TableText"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Footer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#5C6770"),
        )
    )

    body = styles["BodySmall"]
    bullet = styles["BulletSmall"]
    section = styles["Section"]
    table_text = styles["TableText"]
    table_header = styles["TableHeader"]

    story = [
        Paragraph("Structural Damage Image Classification", styles["WriteupTitle"]),
        Paragraph(
            "ML Mini Project | Two-class image classification of undamaged and damaged structures",
            styles["Subtitle"],
        ),
        Paragraph("1. Problem Statement", section),
        Paragraph(
            "Structural inspection produces large collections of images that must be screened for visible damage. "
            "Manual inspection is valuable but time-consuming and can be inconsistent at scale. This project studies "
            "whether machine-learning models can classify an image as undamaged or damaged, where the damaged class "
            "includes cracks and other structural damage. The implementation reproduces the model families reported "
            "in <i>Structural Damage Image Classification</i> by Ho and Troncoso (2018).",
            body,
        ),
        Paragraph(
            "The objective is not to replace an engineer's inspection. It is to build and compare practical image "
            "classification baselines, identify useful feature representations, and provide a reproducible workflow "
            "for a damage-screening experiment.",
            body,
        ),
        Paragraph("2. Dataset", section),
        Paragraph(
            "The project uses the PEER Hub ImageNet Challenge Task 2 Damage State dataset. The supplied NumPy arrays "
            "contain 11,811 training images and 1,460 test images. Each image has shape 224 x 224 x 3. The source "
            "arrays use Caffe-style, mean-subtracted BGR values. The loader restores the channel means, reverses the "
            "channels to RGB, and normalizes each pixel with x = (x / 128) - 1.",
            body,
        ),
        Table(
            [
                [Paragraph("Item", table_header), Paragraph("Project detail", table_header)],
                [Paragraph("Training data", table_text), Paragraph("11,811 images and one-hot labels", table_text)],
                [Paragraph("Held-out data", table_text), Paragraph("1,460 images and one-hot labels", table_text)],
                [Paragraph("Classes", table_text), Paragraph("0 = undamaged; 1 = damaged", table_text)],
                [Paragraph("Split", table_text), Paragraph("Stratified 90% train / 10% validation, seed 42", table_text)],
            ],
            colWidths=[36 * mm, 130 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C4CC")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F7")]),
                ]
            ),
        ),
        Spacer(1, 5),
        Paragraph(
            "Because the labels contain only damaged versus undamaged states, the project cannot report a separate "
            "crack-only accuracy. Cracks are evaluated as part of the damaged class.",
            body,
        ),
        Paragraph("3. Approach", section),
        Paragraph(
            "The workflow compares three classical models with three deep-feature models. Classical models receive "
            "flattened 224 x 224 x 3 images (150,528 features) standardized using a scaler fitted only on the training "
            "split. The transfer-learning models use ImageNet pretrained networks with frozen convolutional bases and "
            "a two-class classification head. The hybrid model uses intermediate InceptionV3 activations as input to "
            "an RBF-kernel SVM.",
            body,
        ),
        Paragraph("4. Implementation", section),
        Paragraph("- <b>KNN:</b> k = 5 with brute-force distance search.", bullet),
        Paragraph("- <b>Logistic regression:</b> L2 regularization, C = 1.0, LBFGS solver.", bullet),
        Paragraph("- <b>SVM:</b> RBF kernel with C = 1.0 and gamma = 0.001.", bullet),
        Paragraph("- <b>MobileNetV1:</b> frozen ImageNet base, global average pooling, and a two-class head.", bullet),
        Paragraph("- <b>InceptionV3:</b> frozen ImageNet base with SGD learning rate 0.01.", bullet),
        Paragraph("- <b>Hybrid:</b> pooled layer-288 InceptionV3 features, standardization, and gamma sweep from 1e-6 to 1e-1.", bullet),
        PageBreak(),
        Paragraph("5. Experimental Results", section),
        Paragraph(
            "To provide a safe, reproducible proof run on a local Windows system, all six models were executed "
            "sequentially with one process at a time. Smoke mode selected a stratified sample of 150 images, giving "
            "135 training images and 15 validation images. TensorFlow thread counts were limited to one. Deep models "
            "ran for two epochs. The following results are from the recorded run on 2026-10-07.",
            body,
        ),
        Table(
            [
                [Paragraph("Model", table_header), Paragraph("Train", table_header), Paragraph("Validation", table_header), Paragraph("Time", table_header)],
                [Paragraph("KNN (k=5)", table_text), Paragraph("68.89%", table_text), Paragraph("66.67%", table_text), Paragraph("0.99 s", table_text)],
                [Paragraph("Logistic regression", table_text), Paragraph("100.00%", table_text), Paragraph("53.33%", table_text), Paragraph("2.77 s", table_text)],
                [Paragraph("SVM RBF", table_text), Paragraph("100.00%", table_text), Paragraph("53.33%", table_text), Paragraph("4.30 s", table_text)],
                [Paragraph("MobileNetV1", table_text), Paragraph("73.33%", table_text), Paragraph("66.67%", table_text), Paragraph("12.96 s", table_text)],
                [Paragraph("InceptionV3", table_text), Paragraph("58.52%", table_text), Paragraph("53.33%", table_text), Paragraph("31.79 s", table_text)],
                [Paragraph("InceptionV3 layer 288 + SVM", table_text), Paragraph("87.41%", table_text), Paragraph("73.33%", table_text), Paragraph("13.89 s", table_text)],
            ],
            colWidths=[91 * mm, 23 * mm, 29 * mm, 23 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                    ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#DCEAF2")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C4CC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ROWBACKGROUNDS", (0, 1), (-1, 5), [colors.white, colors.HexColor("#F1F5F7")]),
                ]
            ),
        ),
        Spacer(1, 7),
        Paragraph(
            "The strongest validation result in this capped run was the InceptionV3 layer-288 hybrid at 73.33%, "
            "followed by MobileNetV1 at 66.67% and KNN at 66.67%. Logistic regression and the raw-pixel RBF SVM "
            "both reached 100% training accuracy but only 53.33% validation accuracy, indicating overfitting on the "
            "small smoke sample. The hybrid gamma sweep selected gamma = 0.001.",
            body,
        ),
        Paragraph("6. Conclusion", section),
        Paragraph(
            "The experiments demonstrate a complete damage-classification pipeline from raw NumPy image arrays to "
            "classical, transfer-learning, and hybrid predictions. Learned visual features were more useful than the "
            "flattened pixel baselines in this smoke evaluation, with the Inception-SVM representation providing the "
            "best validation accuracy. The result is promising for automated screening, but the 15-image validation "
            "set is too small for a final performance claim.",
            body,
        ),
        Paragraph(
            "A stronger final study should run the full training split, evaluate the held-out 1,460-image test set, "
            "report precision, recall, F1-score, and a confusion matrix, and use fixed random seeds for deep-learning "
            "reproducibility. Any operational deployment would also require domain expert review and images from "
            "conditions representative of the intended inspection environment.",
            body,
        ),
        Paragraph(
            "Implementation: code/common.py and code/01_knn.py through code/06_inception_svm_hybrid.py. "
            "Proof runner: run_smoke_150.py. Results: results/results_summary.csv and run_logs/.",
            styles["Footer"],
        ),
    ]

    document.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_pdf()