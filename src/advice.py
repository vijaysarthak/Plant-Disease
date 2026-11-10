"""
advice.py — reads data/advice.csv: what to tell the farmer for each decision.

The CSV has one row per group (G0 ... G5) plus two special rows:
    REFER   -> the model is unsure between treatments: ask an expert
    UNKNOWN -> the photo is not something the model knows

Used by: src/predictor.py (and through it, the app).
"""
import csv

from config import ADVICE_FILE


def load_advice(advice_file=ADVICE_FILE):
    """Returns a dictionary:  {"G1": {"title": ..., "what_it_means": ..., "what_to_do": ..., "source": ...}, ...}"""
    advice = {}
    with open(advice_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            code = row.pop("group_code")
            advice[code] = row
    return advice
