"""
Soil Profile Restrictive Layer Analysis
========================================
Analyzes AIM (Assessment, Inventory, and Monitoring) soil horizon data
to identify profiles encountering bedrock or restrictive layers, classify
the layer type, and extract the depth at which they occur.

Input:  study_soil_horizons.csv  (AIM horizon-level data)
Output: restrictive_layers_summary.csv

Author: Generated for USDA NRCS soil horizon analysis
"""

import csv
import argparse
import re
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# 1. KEYWORD LISTS FOR DETECTION AND CLASSIFICATION
# ---------------------------------------------------------------------------

# All keywords used to flag a horizon as potentially restrictive
DETECTION_KEYWORDS = [
    'bedrock', 'restrictive', 'lithic', 'paralithic', 'petrocalcic',
    'cemented', 'contact', 'R horizon', 'Cr horizon', 'indurated',
    'caliche', 'calcium carbonate', 'stopped digging', 'could not dig',
    'cobble inundation', 'bottoms out',
]

# Keywords used to classify the type of restrictive layer (in priority order)
BEDROCK_KW      = ['bedrock', 'lithic', 'paralithic', 'Cr horizon', 'R horizon', 'bottoms out in bedrock']
PETROCALCIC_KW  = ['petrocalcic', 'caliche', 'calcic', 'calcium carbonate', 'CaCO3', 'CaCo3']
INDURATED_KW    = ['indurated', 'duripan', 'durapan']
CEMENTED_KW     = ['cemented']
COBBLE_KW       = ['cobble inundation', 'stone inundation']


# ---------------------------------------------------------------------------
# 2. CLASSIFICATION FUNCTION
# ---------------------------------------------------------------------------

def classify_layer(notes: str, horizon_name: str) -> str:
    """
    Classify the type of restrictive layer based on field notes and horizon name.

    Priority order:
      1. Lithic / paralithic contact (bedrock)
      2. Petrocalcic / calcic cemented layer
      3. Indurated layer (possible duripan)
      4. Cemented layer (non-calcic)
      5. Stone / cobble inundation
      6. Restrictive (unspecified)
      7. Other flagged horizon
    """
    text = (notes + ' ' + horizon_name).lower()

    if any(k.lower() in text for k in BEDROCK_KW):
        if 'paralithic' in text:
            return 'Paralithic contact'
        return 'Lithic contact (bedrock)'

    if any(k.lower() in text for k in PETROCALCIC_KW):
        return 'Petrocalcic/calcic cemented'

    if any(k.lower() in text for k in INDURATED_KW):
        return 'Indurated layer (duripan?)'

    if any(k.lower() in text for k in CEMENTED_KW):
        return 'Cemented layer'

    if any(k.lower() in text for k in COBBLE_KW):
        return 'Stone/cobble inundation'

    if 'restrictive' in text:
        return 'Restrictive (unspecified)'

    return 'Other flagged'


# ---------------------------------------------------------------------------
# 3. DEPTH EXTRACTION FUNCTION
# ---------------------------------------------------------------------------

def extract_restrictive_depth(notes: str) -> int | None:
    """
    Parse the depth (cm) at which the restrictive layer was encountered
    from free-text field notes.

    Tries two patterns:
      - "at/past/beyond/hit at/hit <N> cm"   (explicit directional phrasing)
      - "<N> cm ... <restrictive keyword>"    (depth mentioned near keyword)

    Returns the depth as an integer, or None if no match found.
    """
    # Pattern 1: directional phrasing before a depth value
    m = re.search(
        r'(?:at|past|beyond|hit\s+at|hit)\s+(\d+)\s*cm',
        notes,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # Pattern 2: depth value followed by a restrictive keyword
    m = re.search(
        r'(\d+)\s*cm.*?(?:restrictive|bedrock|contact|cemented|indurated|caliche|petrocalcic)',
        notes,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# 4. MAIN ANALYSIS
# ---------------------------------------------------------------------------

def build_restrictive_summary(input_csv: str) -> list[dict]:
    """
    Full pipeline:
      1. Load all horizon records.
      2. Flag horizons containing restrictive-layer keywords.
      3. Group flagged horizons by soil profile (PrimaryKey).
      4. For each profile, select the deepest flagged horizon as the
         best representative of the restrictive layer.
      5. Classify the layer type and extract the depth.
            6. Return summary records.
    """

    # -- 4a. Load all horizon records ------------------------------------
    print(f"Reading: {input_csv}")
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    print(f"  Total horizon records: {len(all_rows)}")

    # -- 4b. Flag horizons with restrictive-layer signals ----------------
    flagged_rows = []
    for row in all_rows:
        notes = row.get('HorizonNotes') or ''
        hname = row.get('HorizonName') or ''
        combined = (notes + ' ' + hname).lower()
        if any(kw.lower() in combined for kw in DETECTION_KEYWORDS):
            flagged_rows.append(row)

    print(f"  Flagged horizon records: {len(flagged_rows)}")

    # -- 4c. Group flagged horizons by profile ---------------------------
    profiles: dict[str, list[dict]] = defaultdict(list)
    for row in flagged_rows:
        profiles[row['PrimaryKey']].append(row)

    print(f"  Unique profiles with restrictive signal: {len(profiles)}")

    # -- 4d. Select representative horizon and classify ------------------
    summary = []
    for pk, horizons in profiles.items():

        # Sort by HorizonDepthLower descending; pick the deepest flagged horizon
        def safe_lower(h: dict) -> int:
            try:
                return int(h['HorizonDepthLower'])
            except (ValueError, KeyError):
                return 0

        horizons_sorted = sorted(horizons, key=safe_lower, reverse=True)
        best = horizons_sorted[0]

        notes = best.get('HorizonNotes') or ''
        hname = best.get('HorizonName') or ''

        # Classify layer type
        layer_type = classify_layer(notes, hname)

        # Extract restrictive depth from notes; fall back to HorizonDepthLower
        depth = extract_restrictive_depth(notes)
        if depth is None:
            depth = safe_lower(best) or None

        # Build a short human-readable profile ID from the PrimaryKey
        parts = pk.split('_')
        short_id = '_'.join(parts[2:4]) if len(parts) >= 4 else pk

        summary.append({
            'ProfileKey':          pk,
            'ShortID':             short_id,
            'RestrictiveDepth_cm': depth,
            'LayerType':           layer_type,
            'Notes':               notes[:250],  # truncate for readability
        })

    # Sort output by layer type, then depth
    summary.sort(key=lambda x: (x['LayerType'], x['RestrictiveDepth_cm'] or 999))

    # -- 4e. Print summary statistics ------------------------------------
    print("\n--- Profiles by restrictive layer type ---")
    counts = Counter(s['LayerType'] for s in summary)
    for layer_type, count in counts.most_common():
        print(f"  {layer_type:<35} {count:>4} profiles")

    depths = [s['RestrictiveDepth_cm'] for s in summary if s['RestrictiveDepth_cm'] is not None]
    if depths:
        depths_sorted = sorted(depths)
        n = len(depths_sorted)
        median = (
            depths_sorted[n // 2]
            if n % 2 == 1
            else (depths_sorted[n // 2 - 1] + depths_sorted[n // 2]) / 2
        )
        print(f"\n--- Depth statistics (cm, n={n}) ---")
        print(f"  Min:    {min(depths_sorted)}")
        print(f"  Max:    {max(depths_sorted)}")
        print(f"  Median: {median}")
        print(f"  Mean:   {sum(depths_sorted) / n:.1f}")

    print(f"\nTotal profiles in summary: {len(summary)}")

    return summary


def write_summary_csv(summary: list[dict], output_csv: str) -> None:
    """Write restrictive summary rows to CSV."""
    fieldnames = ['ProfileKey', 'ShortID', 'RestrictiveDepth_cm', 'LayerType', 'Notes']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nOutput written to: {output_csv}")


def append_to_plot_characteristics(summary: list[dict], plot_csv: str) -> None:
    """
    Append restrictive layer fields to plot-level AIM data.

    Added columns:
      - RestrictiveLayerDepth_cm
      - RestrictiveLayerType
            - bedrock
    """
    print(f"\nUpdating plot file: {plot_csv}")
    with open(plot_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        plot_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    depth_field = 'RestrictiveLayerDepth_cm'
    type_field = 'RestrictiveLayerType'
    bedrock_field = 'bedrock'

    if depth_field not in fieldnames:
        fieldnames.append(depth_field)
    if type_field not in fieldnames:
        fieldnames.append(type_field)
    if bedrock_field not in fieldnames:
        fieldnames.append(bedrock_field)

    summary_by_profile = {row['ProfileKey']: row for row in summary}

    matched = 0
    for row in plot_rows:
        profile_key = row.get('PrimaryKey', '')
        match = summary_by_profile.get(profile_key)
        if match is None:
            row[depth_field] = ''
            row[type_field] = ''
            row[bedrock_field] = ''
            continue

        depth_val = match.get('RestrictiveDepth_cm')
        row[depth_field] = '' if depth_val is None else str(depth_val)
        layer_type = match.get('LayerType', '')
        row[type_field] = layer_type
        # Populate bedrock depth only for lithic contacts.
        if layer_type == 'Lithic contact (bedrock)' and depth_val is not None:
            row[bedrock_field] = str(depth_val)
        else:
            row[bedrock_field] = ''
        matched += 1

    with open(plot_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)

    print(f"  Plot rows updated: {len(plot_rows)}")
    print(f"  Profiles with restrictive layer populated: {matched}")


def run_analysis(input_csv: str, output_csv: str, plot_csv: str | None = None) -> None:
    summary = build_restrictive_summary(input_csv)
    write_summary_csv(summary, output_csv)
    if plot_csv:
        append_to_plot_characteristics(summary, plot_csv)


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze restrictive layers from AIM horizon data.')
    parser.add_argument(
        '--input-csv',
        default='Data/aim_data/study_soil_horizons.csv',
        help='Path to horizon CSV input',
    )
    parser.add_argument(
        '--output-csv',
        default='Data/aim_data/restrictive_layers_summary.csv',
        help='Path to restrictive layer summary output CSV',
    )
    parser.add_argument(
        '--plot-csv',
        default='Data/aim_data/study_plot_characteristics.csv',
        help='Path to plot-level CSV to append restrictive-layer fields',
    )
    args = parser.parse_args()

    run_analysis(args.input_csv, args.output_csv, args.plot_csv)