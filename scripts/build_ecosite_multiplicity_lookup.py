from pathlib import Path

import pandas as pd


def normalize_compname(series: pd.Series) -> pd.Series:
    out = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    out = out.replace({"": pd.NA, "na": pd.NA, "null": pd.NA})
    return out


def normalize_ecoclassid(series: pd.Series) -> pd.Series:
    out = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^[RF]+", "", regex=True)
        .str.replace(r"[,;|].*$", "", regex=True)
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    out = out.replace({"": pd.NA, "NA": pd.NA, "NULL": pd.NA})
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "Data" / "aim_data"
    output_path = data_dir / "compname_mlra_ecosite_multiplicity.csv"

    input_files = sorted(data_dir.glob("*_compname_ecosite_raw_pairs.csv"))
    if not input_files:
        raise FileNotFoundError(f"No raw pairs files found in {data_dir}")

    frames = []
    for path in input_files:
        df = pd.read_csv(path)
        required = {"mlrasymbol", "compname", "ecoclassid"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

        keep = df[["mlrasymbol", "compname", "ecoclassid"]].copy()
        keep["mlrasymbol"] = keep["mlrasymbol"].astype(str).str.strip().str.lower()
        keep["compname_norm"] = normalize_compname(keep["compname"])
        keep["ecoclassid_norm"] = normalize_ecoclassid(keep["ecoclassid"])
        keep = keep.dropna(subset=["mlrasymbol", "compname_norm", "ecoclassid_norm"])
        frames.append(keep[["mlrasymbol", "compname_norm", "ecoclassid_norm"]])

    merged = pd.concat(frames, ignore_index=True)

    lookup = (
        merged.groupby(["compname_norm", "mlrasymbol"], as_index=False)["ecoclassid_norm"]
        .nunique()
        .rename(columns={"ecoclassid_norm": "n_ecosites"})
        .sort_values(["mlrasymbol", "compname_norm"], kind="stable")
    )

    lookup.to_csv(output_path, index=False)

    print(f"Wrote {len(lookup):,} rows to {output_path}")


if __name__ == "__main__":
    main()
