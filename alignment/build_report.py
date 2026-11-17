"""Build the PDF report from the inspection and comparison outputs."""

import argparse
import csv
from pathlib import Path
from statistics import mean, median

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


HERE = Path(__file__).resolve().parent
NAVY, INK, RULE, BAND = colors.HexColor("#172b4d"), colors.HexColor("#1f2933"), colors.HexColor("#d5dce6"), colors.HexColor("#eef2f7")
RUNS = [
    ("mfa_generic", "Results", "TextGrids", "english_mfa v3.1.0", "english_mfa v3.1.0 (generic, one pronunciation per word)",
     "5 hand-written entries"),
    ("mfa_us", "Results/us_mfa", "TextGrids_us_mfa", "english_mfa v3.1.0", "english_us_mfa (US English, with variants)",
     "6 words from the english_us_mfa G2P model"),
    ("arpa_us", "Results/us_arpa", "TextGrids_us_arpa", "english_us_arpa v3.0.0", "english_us_arpa (ARPAbet, with variants)",
     "4 words from the english_us_arpa G2P model"),
]


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table(data, widths, align_from=1):
    head = ParagraphStyle("head", fontName="Body-Bold", fontSize=8, leading=10.5, textColor=colors.white)
    right = ParagraphStyle("head-right", parent=head, alignment=2)
    data = [[Paragraph(text, right if column >= align_from else head) for column, text in enumerate(data[0])]] + data[1:]
    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Body-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Body"),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10.5),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("ALIGN", (align_from, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ])
    return Table(data, colWidths=widths, style=style, repeatRows=1)


def build(results, destination):
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(fonts / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    body = ParagraphStyle("body", fontName="Body", fontSize=9, leading=13.2, textColor=INK, spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#52606d"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10.5, spaceAfter=0)
    kicker = ParagraphStyle("kicker", parent=small, spaceAfter=2)
    title = ParagraphStyle("title", parent=body, fontName="Body-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=body, fontName="Body-Bold", fontSize=11.5, leading=15, textColor=NAVY,
                             spaceBefore=8, spaceAfter=5)

    overviews = {name: rows(HERE / folder / "overview.csv") for name, folder, *_ in RUNS}
    flags = {name: rows(HERE / folder / "review_flags.csv") for name, folder, *_ in RUNS}
    boundaries = rows(results / "comparison/word_boundaries.csv")
    seconds = sum(float(row["duration_seconds"]) for row in overviews["mfa_generic"])
    words = sum(int(row["words"]) for row in overviews["mfa_generic"])

    def shift(first, second, speaker):
        values = [abs(float(row[f"{second}_{side}"]) - float(row[f"{first}_{side}"])) for row in boundaries
                  if row["speaker"] == speaker for side in ("start", "end")]
        return (str(len(values)), f"{1000 * median(values):.0f}", f"{1000 * mean(values):.0f}", f"{mean(v <= 0.0205 for v in values):.0%}",
                f"{mean(v <= 0.0505 for v in values):.0%}", f"{1000 * max(values):.0f}")

    story = [
        Paragraph("LTRC, IIIT Hyderabad / Assignment 2a", kicker),
        Paragraph("Forced alignment with MFA", title),
        Paragraph(
            f"Montreal Forced Aligner 3.3.10 aligned the six supplied recordings from two speakers ({seconds:.3f} seconds, "
            f"{words} word tokens). One script converts the audio to 16 kHz mono PCM, pairs each WAV with a normalized "
            "transcript, validates dictionary coverage, runs the aligner and checks the exported TextGrids. "
            "The corpus was aligned three times so that the effect of the dictionary and of the acoustic model can be "
            "read separately: adjacent runs differ in one component.", body),
        Paragraph("Models and dictionaries", heading),
        table([["Run", "Acoustic model", "Dictionary", "Out-of-vocabulary words"]]
              + [[name, acoustic, Paragraph(dictionary, cell), Paragraph(missing, cell)]
                 for name, _, _, acoustic, dictionary, missing in RUNS],
              [2.2 * cm, 3.6 * cm, 6.0 * cm, 5.2 * cm], align_from=4),
        Spacer(1, 5),
        Paragraph(
            "Both acoustic models are pretrained MFCC GMM-HMM models. english_mfa uses an IPA phone set and "
            "english_us_arpa uses ARPAbet with stress marks. With the generic dictionary alone, six phone intervals "
            "across five word types (<i>Dukakis, judgeships, Maffy, Melnicove, politicize</i>) were unknown. "
            "Hand-written entries, or a pretrained G2P model in the other two runs, removed every unknown interval. "
            "All 18 TextGrids pass the Praat checks: a words tier and a phones tier, contiguous positive-duration "
            "intervals, and a total duration equal to the audio.", body),
        Paragraph("Exported tiers", heading),
        table([["Recording", "Seconds", "Words", "Phones<br/>generic / us / arpa", "Phones over 0.5 s<br/>generic / us / arpa"]]
              + [[row["clip"], row["duration_seconds"], row["words"],
                  " / ".join(overviews[name][index]["phones"] for name, *_ in RUNS),
                  " / ".join(str(sum(flag["clip"] == row["clip"] for flag in flags[name])) for name, *_ in RUNS)]
                 for index, row in enumerate(overviews["mfa_generic"])],
              [6.0 * cm, 2.0 * cm, 1.8 * cm, 3.6 * cm, 3.6 * cm]),
        Paragraph("Agreement between runs", heading),
        Paragraph(
            "A shift is the absolute difference between two runs for the same word start or end. MFA places "
            "boundaries on a 10 ms frame grid. The corpus has no hand-labelled boundaries, so these numbers measure "
            "disagreement, not accuracy. On the news-reader recordings (F2BJ) the two acoustic models agree within "
            "20 ms on 94% of word boundaries. The few large shifts come from the specific errors described on the "
            "next pages.", body),
        table([["Runs compared", "What changes", "Speaker", "Count", "Median ms", "Mean ms", "≤ 20 ms", "≤ 50 ms", "Max ms"]]
              + [[f"{first} / {second}", change, label, *shift(first, second, speaker)]
                 for first, second, change in (("mfa_generic", "mfa_us", "dictionary"), ("mfa_us", "arpa_us", "acoustic model"))
                 for speaker, label in (("F2BJ", "F2BJ"), ("SESS0131", "ISLE"))],
              [3.6 * cm, 2.6 * cm, 1.8 * cm, 1.5 * cm, 1.7 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm + 0.5 * cm], align_from=3),
        PageBreak(),
        Paragraph("Observation 1 · Trailing silence absorbed by the last phone", heading),
        Image(str(results / "comparison/ISLE_SESS0131_BLOCKD02_01_sprt1.png"), width=17 * cm, height=17 * cm * 0.7),
        Paragraph(
            "With the english_mfa acoustic model the final /t/ of <i>bait</i> runs from 2.050 s to the end of the file at "
            "4.125 s, and in clip 03 the final /d/ of <i>bed</i> runs from 1.790 s to 4.500 s. The spectrogram shows a "
            "release near 2.2 s and then a quiet tail. Changing the dictionary does not help (mfa_us is identical here), "
            "so the cause is the acoustic model, not the pronunciation. english_us_arpa ends <i>bait</i> at 2.250 s and "
            "<i>bed</i> at 2.060 s, labels the tail as silence, and also inserts a pause between <i>white</i> and "
            "<i>not</i>. No arpa_us phone is longer than 0.5 s. Clip 02 ends in a vowel and all three runs agree on it.", body),
        Paragraph(
            "The same figure shows how the phone sets differ. The generic dictionary gives <i>white</i> as ʋ aj ʈ and "
            "<i>said</i> as s aj d, its only entries for those words, where the US dictionaries give w aj t and s ɛ d. "
            "The aligner cannot recover from a dictionary entry that does not match what was said; it can only stretch "
            "the phones it was given.", body),
        PageBreak(),
        Paragraph("Observation 2 · A spoken word labelled as a pause", heading),
        Image(str(results / "comparison/F2BJRLP1_19.2-21.5s.png"), width=17 * cm, height=17 * cm * 0.7),
        Paragraph(
            "The transcript reads “… one of the toughest. As W B U R’s Margo Melnicove reports …”. In mfa_generic, "
            "<i>toughest</i> has the vowel-less entry ʈ ə f s ʈ and ends at 19.68 s; the aligner then places <i>as</i> on "
            "the rest of that word (19.68–19.92 s) and marks 19.92–20.58 s as a pause. The signal contradicts this: "
            "the level between 20.40 and 20.55 s is −29 to −31 dBFS, the same as the average of the whole recording "
            "(−29 dBFS), while 19.95–20.35 s is more than 10 dB quieter. The spoken <i>as</i> was skipped. Both US "
            "dictionaries place <i>toughest</i> up to 20.00 s, a pause, and <i>as</i> at 20.35–20.57 s.", body),
        Paragraph(
            "The initialism shows a second dictionary effect. The generic dictionary has <i>w</i> as d ɐ b and <i>u</i> "
            "as a single ɐ, so the missing syllables of “double-u” are pushed into the next letter and the vowel of "
            "<i>b</i> is stretched to 390 ms. With full letter names in the US dictionaries the four letters receive "
            "plausible durations, and the two acoustic models agree on them within 20 ms.", body),
        Paragraph("Summary", heading),
        Paragraph(
            "• Word and phone tiers are consistent in every run: each word interval is covered exactly by its phones, "
            "and pauses appear as empty intervals in both tiers.<br/>"
            "• Dictionary quality mattered more than the acoustic model for the long recordings. A single mismatched "
            "entry displaced two neighbouring words by up to 0.67 s.<br/>"
            "• The acoustic model mattered for utterance-final silence in the short ISLE clips, which are recordings "
            "of a non-native speaker with long quiet tails.<br/>"
            "• arpa_us is the run recommended for further use: no unknown phones, no phone over 0.5 s, and no "
            "disagreement with the signal in the places examined.<br/>"
            "• Remaining limits: there are no reference boundaries, the checks above are acoustic and visual, and "
            "listening in Praat is still needed before any boundary is treated as ground truth.", body),
        Paragraph("Scripts, TextGrids and tables: alignment/ in the repository. Models: "
                  "<a href='https://mfa-models.readthedocs.io/' color='#1f5fa8'>MFA model documentation</a>.", small),
    ]
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                      bottomMargin=1.6 * cm, title="Forced alignment with MFA").build(story)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "Results")
    parser.add_argument("--output", type=Path, default=HERE / "Alignment_Report.pdf")
    args = parser.parse_args()
    build(args.results, args.output)
