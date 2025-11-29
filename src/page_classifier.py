def classify_page(text: str) -> str:
    """
    Simple heuristic-based classification of page type based on keywords.
    """
    lower_text = text.lower()
    if "pharmacy" in lower_text or "rx" in lower_text:
        return "Pharmacy"
    if "total" in lower_text or "amount" in lower_text or "grand total" in lower_text:
        # Likely a summary or final billing page
        return "Final Bill"
    # Default fallback
    return "Bill Detail"
