# src/simple_parser.py
import re

NUM_RE = re.compile(r'[₹$]?\s*[\d,]+(?:\.\d+)?(?:\/-)?')

def clean_number_token(tok):
    if tok is None: return None
    s = str(tok).strip()
    s = s.replace('₹', '').replace('$', '').replace('/-', '')
    s = s.replace(',', '')
    s = re.sub(r'[^\d\.\-]','', s)
    if s == '' or s == '.' or s == '-':
        return None
    try:
        return float(s)
    except:
        return None

def is_number_token(tok):
    return clean_number_token(tok) is not None

def y_center(w):
    return w.get('top', 0) + w.get('height', 0) / 2

def group_lines(words, y_tol=14):
    if not words: return []
    words_sorted = sorted(words, key=lambda w: y_center(w))
    lines = []
    cur = [words_sorted[0]]
    for w in words_sorted[1:]:
        if abs(y_center(w) - y_center(cur[-1])) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
    lines.append(cur)
    out = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: x.get('left', 0))
        text = " ".join([t['text'] for t in line_sorted])
        out.append({"text": text, "words": line_sorted})
    return out

def parse_items(lines):
    items = []
    for line in lines:
        words = line['words']
        if not words: continue
        numerics = []
        for w in words:
            if is_number_token(w['text']):
                val = clean_number_token(w['text'])
                if val is None: continue
                numerics.append((w['left'], w['text'], val))
        if not numerics:
            continue
        numerics_sorted = sorted(numerics, key=lambda x: x[0])
        amt = None; rate = None; qty = None
        if len(numerics_sorted) >= 1:
            left_amt, raw_amt, val_amt = numerics_sorted[-1]
            amt = val_amt
        if len(numerics_sorted) >= 2:
            left_rate, raw_rate, val_rate = numerics_sorted[-2]
            rate = val_rate
        if len(numerics_sorted) >= 3:
            left_qty, raw_qty, val_qty = numerics_sorted[-3]
            qty = val_qty
        numeric_left_positions = [x[0] for x in numerics_sorted]
        cutoff = min(numeric_left_positions) if numeric_left_positions else None
        if cutoff is not None:
            name_parts = [w['text'] for w in words if w['left'] < cutoff]
            name = " ".join(name_parts).strip()
            if not name:
                line_text = line['text']
                for _, raw, _ in numerics_sorted:
                    line_text = line_text.replace(raw, '')
                name = line_text.strip()
        else:
            name = line['text'].strip()
        name = name if name else None
        items.append({
            "item_name": name,
            "item_amount": round(float(amt), 2) if amt is not None else None,
            "item_rate": round(float(rate), 2) if rate is not None else None,
            "item_quantity": round(float(qty), 2) if qty is not None else None
        })
    dedup = []
    seen = set()
    for it in items:
        key = ( (it['item_name'] or '').lower(), it['item_amount'] )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup

def find_totals(lines):
    totals = {}
    keywords = ['total','subtotal','grand total','amount payable','net amount','balance due','invoice total','total amount']
    for line in reversed(lines):
        t = line['text'].lower()
        if any(k in t for k in keywords):
            nums = [w for w in line['words'] if is_number_token(w['text'])]
            if nums:
                r = sorted(nums, key=lambda x: x['left'])[-1]
                val = clean_number_token(r['text'])
                if val is not None:
                    totals[line['text'].strip()] = round(val, 2)
        else:
            words = line['words']
            if len(words) <= 4:
                nums = [w for w in words if is_number_token(w['text'])]
                if len(nums) == 1:
                    val = clean_number_token(nums[0]['text'])
                    if val is not None:
                        totals[line['text'].strip()] = round(val, 2)
    return totals
