"""One-shot probe: check sacatalog survey order extraction from WY surveys."""
import re
import requests

SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"

def sda(sql, label=""):
    r = requests.post(SDA_URL, json={"format": "JSON+COLUMNNAME", "query": sql}, timeout=60)
    if not r.ok:
        print(f"{label} HTTP {r.status_code}:", r.text[:200])
        return []
    d = r.json()
    rows = d.get("Table", [])
    if label:
        print(f"{label}  cols={rows[0] if rows else 'NONE'}")
    return rows

# Scale → order mapping (SSURGO standard)
def scale_to_order(scale: int) -> int:
    if scale <= 7920:
        return 1
    elif scale <= 31680:
        return 2
    elif scale <= 125000:
        return 3
    elif scale <= 250000:
        return 4
    else:
        return 5

def infer_order_from_fgdc(fgdc: str) -> tuple[int | None, str]:
    """Returns (primary_order, method_used)."""
    if not fgdc:
        return None, "no_fgdc"

    ORDER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    pat_num = re.compile(r"Order\s+(\d)", re.IGNORECASE)
    pat_word = re.compile(r"\b(first|second|third|fourth|fifth)\s+order\b", re.IGNORECASE)

    nums = set(int(m.group(1)) for m in pat_num.finditer(fgdc))
    words = set(ORDER_WORDS[m.group(1).lower()] for m in pat_word.finditer(fgdc))
    text_orders = nums | words

    # Also extract srcscale values
    pat_scale = re.compile(r"<srcscale>\s*(\d+)\s*</srcscale>", re.IGNORECASE)
    scales = [int(m.group(1)) for m in pat_scale.finditer(fgdc)]
    scale_orders = set(scale_to_order(s) for s in scales) if scales else set()

    # Prefer text mentions if available
    all_orders = text_orders or scale_orders
    if not all_orders:
        return None, "no_signal"
    primary = min(all_orders)
    method = "text" if text_orders else "scale"
    return primary, method

# Get ALL WY surveys
rows = sda(
    "SELECT areasymbol, areaname, fgdcmetadata FROM sacatalog WHERE areasymbol LIKE 'WY%'",
    "=== WY sacatalog ==="
)

print(f"\nTotal WY surveys: {len(rows)-1}")
print("\n=== Survey order extraction ===")
no_signal = []
for row in rows[1:]:
    areasymbol, areaname, fgdc = row[0], row[1], row[2] or ""
    order, method = infer_order_from_fgdc(fgdc)
    flag = " ***" if order is None else ""
    print(f"  {areasymbol}  {areaname[:45]:45s}  order={order}  ({method}){flag}")
    if order is None:
        no_signal.append((areasymbol, areaname))

print(f"\nSurveys with no order signal: {len(no_signal)}")
for a, n in no_signal:
    print(f"  {a}  {n}")
