from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "article_id",
    "source",
    "text",
    "published_at"
]

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a standardised CSV dataset.

    Works independently of the original dataset source.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "Expected a CSV file."
        )

    df = pd.read_csv(file_path)

    # Validate required columns.
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Standardise column order.
    df = df[REQUIRED_COLUMNS].copy()

    # Validate identifiers.
    if df["article_id"].isna().any():
        raise ValueError("Missing article IDs detected.")

    if df["article_id"].eq("").any():
        raise ValueError("Empty article IDs detected.")

    if df["source"].isna().any():
        raise ValueError("Missing sources detected.")

    if df["article_id"].duplicated().any():
        raise ValueError("Duplicate article IDs detected.")

    # Remove records without article text.
    df = df[
        df["text"].notna() & df["text"].ne("")
    ].copy()

    # Parse dates without assuming a particular source format.
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce"
    )

    print("\nDataset loaded successfully.")
    print(f"Articles: {len(df)}")
    print(f"Sources: {df['source'].nunique()}")

    return df.reset_index(drop=True)