# BAAC 2024 Data Quality Streamlit Report

## How to launch

Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## How data loading works

The application loads the four CSV files automatically from the local `data` folder:

- `data/caract-2024.csv`
- `data/lieux-2024.csv`
- `data/usagers-2024.csv`
- `data/vehicules-2024.csv`

You do not need to upload the files in Streamlit each time.

## How to use another dataset

Replace the four CSV files in the `data` folder, or edit the `USER DATA SETTINGS` block at the top of `streamlit_app.py`:

```python
DEFAULT_DATA_DIR = "data"
DATASET_FILES = {
    "caract": "caract-2024.csv",
    "lieux": "lieux-2024.csv",
    "usagers": "usagers-2024.csv",
    "vehicules": "vehicules-2024.csv",
}
```

For example, for another year, change the filenames to `caract-2023.csv`, `lieux-2023.csv`, `usagers-2023.csv`, and `vehicules-2023.csv`.

## Main checks included

- Primary key duplicates
- Full duplicate rows
- Month from 1 to 12 and day from 1 to 31
- Valid calendar date and HH:MM time format
- Coordinates inside France and French overseas bounding boxes
- Data type coherence
- Outliers and extreme values
- Age between 0 and 110
- Negative-value policy
- Referential integrity between the four tables
- Missing and hidden missing values such as `-1`
