# Datasets

Place the original project tables here:

- `Fake.csv`
- `True.csv`

Expected columns: `title`, `text`, `subject`, `date`.

These files were copied from the original Downloads copies used by `original/fake_news_detection.py`. They are gitignored because of size.

## Label mapping (from the original script)

- Rows from `Fake.csv` receive label `0` (fake)
- Rows from `True.csv` receive label `1` (real)

## Known collection artefacts

True articles often begin with a wire dateline such as `CITY (Reuters) -`. Fake articles almost never do. Subject names are also disjoint across the two files. The classifier can exploit those shortcuts.

The dataset is the widely used ISOT Fake News collection (Ahmed, Traore, Saad). Use it under the terms of the source you obtained it from (commonly Kaggle).
