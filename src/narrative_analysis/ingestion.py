from pathlib import Path
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STANDARD_COLUMNS = [
    "article_id",
    "source",
    "title",
    "text",
    "published_at",
    "url"
]

INPUT_FILES = [
    (
        "mzanzi",
        PROJECT_ROOT / "data/raw/Fake News (mzanzi stories) Fake.xlsx"
    ),
    (
        "sa_news",
        PROJECT_ROOT / "data/raw/Fake News (sa-news.com) Fake.xlsx"
    ),
    (
        "whatsapp",
        PROJECT_ROOT / "data/raw/Fake News (whatsappgroup.co.za) Fake.xlsx"
    )
]

OUTPUT_FILE = (
    PROJECT_ROOT / "data/interim/fake_news_combined.csv"
)

# --------------------------------------------------
# STANDARDISE EXCEL COLUMNS
# --------------------------------------------------

def standardize_columns(
    df: pd.DataFrame,
    source: str,
    source_file: str,
) -> pd.DataFrame:
    """
    Extract the correct Excel columns and standardize the schema.

    Excel column positions are zero-based:
    1 = published date
    3 = title
    5 = article text
    7 = url
    """
    df = df.copy()

    # The Excel layout requires at least 8 columns.
    if df.shape[1] < 8:
        raise ValueError(
            f"{source_file} has {df.shape[1]} columns; "
            "at least 8 are required to extract the expected fields."
        )

    # Extract fields using the positions from your working code.
    cleaned = pd.DataFrame({
        "published_at": df.iloc[:, 1],
        "title": df.iloc[:, 3],
        "text": df.iloc[:, 5],
        "url": df.iloc[:, 7]
    })

    # Record dataset provenance.
    cleaned["source"] = source
    cleaned["article_id"] = pd.NA

    # Ensure every canonical column exists.
    for column in STANDARD_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    # Keep a consistent schema and column order.
    return cleaned[STANDARD_COLUMNS]

# --------------------------------------------------
# PARSE PUBLICATION DATES
# --------------------------------------------------

def parse_publication_dates(
    dates: pd.Series,
    source: str,
) -> pd.Series:
    """Parse Excel dates and dataset-specific date strings."""

    # Preserve dates Excel has already interpreted as datetimes.
    parsed = pd.to_datetime(
        dates,
        errors="coerce"
    )

    # Only try explicit string formats for values not yet parsed.
    raw_strings = dates.astype("string").str.strip()

    date_formats = {
        "mzanzi": ["%Y/%m/%d"],
        "sa_news": ["%d/%m/%Y"],
        "whatsapp": [
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%B %d, %Y"
        ]
    }

    if source not in date_formats:
        raise ValueError(
            f"Unknown dataset: {source}"
        )

    for date_format in date_formats[source]:

        remaining = parsed.isna() & raw_strings.notna()

        if not remaining.any():
            break

        parsed.loc[remaining] = pd.to_datetime(
            raw_strings.loc[remaining],
            format=date_format,
            errors="coerce",
            utc=True
        )

    return parsed

# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Validate essential fields without silently dropping rows."""
    df = df.copy()

    # Clean text fields.
    for column in ["title", "text", "url"]:
        df[column] = df[column].astype("string").str.strip()

    missing_text = (
        df["text"].isna()
        | df["text"].eq("").fillna(False)
    )

    if missing_text.any():
        print(
            f"Warning: {missing_text.sum()} records in "
            f"{source} have missing or empty text."
        )

    # Parse dates consistently; invalid dates become NaT.
    df["published_at"] = parse_publication_dates(
        df["published_at"],
        source
    )

    invalid_dates = df["published_at"].isna().sum()

    if invalid_dates:
        print(
            f"Warning: {invalid_dates} records in "
            f"{source} have missing or invalid dates."
        )

    # Generate deterministic IDs for records without an ID.
    df["article_id"] = df["article_id"].astype("string")

    missing_id = (
        df["article_id"].isna()
        | df["article_id"].str.strip().eq("").fillna(False)
    )

    # Use the original row position, starting at 1.
    df.loc[missing_id, "article_id"] = [
        f"{source}_{i + 1:07d}"
        for i in df.index[missing_id]
    ]

    # Check for duplicate article IDs within this dataset.
    duplicated = df["article_id"].duplicated(keep=False)

    if duplicated.any():
        print(
            f"Warning: {duplicated.sum()} rows in "
            f"{source} have duplicated article IDs."
        )

    return df

# --------------------------------------------------
# EXCEL INGESTION
# --------------------------------------------------

def ingest_excel_datasets(
    input_files=None,
    output_file=OUTPUT_FILE
) -> pd.DataFrame:

    if input_files is None:
        input_files = INPUT_FILES

    dataframes = []

    for source, file in input_files:

        file = Path(file)

        if not file.exists():
            raise FileNotFoundError(file)

        print(f"\nProcessing: {file.name}")

        raw_df = pd.read_excel(
            file,
            header=None,
            dtype=object
        )

        cleaned = standardize_columns(
            raw_df,
            source,
            file.name
        )

        cleaned = validate_dataset(
            cleaned,
            source
        )

        dataframes.append(cleaned)

        print(f"Cleaned records: {len(cleaned)}")

    if not dataframes:
        raise ValueError("No Excel datasets were supplied.")

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Check IDs across all datasets.
    if combined_df["article_id"].duplicated().any():
        raise ValueError(
            "Duplicate article IDs detected across datasets."
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined_df.to_csv(
        output_file,
        index=False
    )

    print("\nExcel ingestion completed.")
    print(f"Total records: {len(combined_df)}")
    print(f"Saved to: {output_file}")

    return combined_df


if __name__ == "__main__":
    ingest_excel_datasets()