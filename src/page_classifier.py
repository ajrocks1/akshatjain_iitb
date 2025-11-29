# src/page_classifier.py
def classify_page(text: str) -> str:
    lowered = text.lower()
    if "pharmacy" in lowered:
        return "Pharmacy"
    elif "final amount" in lowered or "total amount" in lowered or "grand total" in lowered:
        return "Final Bill"
    return "Bill Detail"
