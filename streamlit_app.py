
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="BAAC 2024 Data Quality Report",
    page_icon="✅",
    layout="wide",
)

# =============================================================================
# USER DATA SETTINGS
# =============================================================================
# To use another BAAC extract without uploading files in Streamlit every time:
# 1) Put your four CSV files in the folder called "data" next to this script.
# 2) Change the filenames below if your files have different names.
# Example: for 2023 data, replace "caract-2024.csv" by "caract-2023.csv".
# =============================================================================
DEFAULT_DATA_DIR = "data"

DATASET_FILES = {
    "caract": "caract-2024.csv",
    "lieux": "lieux-2024.csv",
    "usagers": "usagers-2024.csv",
    "vehicules": "vehicules-2024.csv",
}

EXPECTED_COLUMNS = {
    "caract": ["Num_Acc", "jour", "mois", "an", "hrmn", "lum", "dep", "com", "agg", "int", "atm", "col", "adr", "lat", "long"],
    "lieux": ["Num_Acc", "catr", "voie", "v1", "v2", "circ", "nbv", "vosp", "prof", "pr", "pr1", "plan", "lartpc", "larrout", "surf", "infra", "situ", "vma"],
    "usagers": ["Num_Acc", "id_usager", "id_vehicule", "num_veh", "place", "catu", "grav", "sexe", "an_nais", "trajet", "secu1", "secu2", "secu3", "locp", "actp", "etatp"],
    "vehicules": ["Num_Acc", "id_vehicule", "num_veh", "senc", "catv", "obs", "obsm", "choc", "manv", "motor", "occutc"],
}

DOMAINS = {
    "caract": {
        "lum": {"1", "2", "3", "4", "5"},
        "agg": {"1", "2"},
        "int": set(map(str, range(1, 10))),
        "atm": {"-1"} | set(map(str, range(1, 10))),
        "col": {"-1"} | set(map(str, range(1, 8))),
    },
    "lieux": {
        "catr": {"1", "2", "3", "4", "5", "6", "7", "9"},
        "circ": {"-1", "1", "2", "3", "4"},
        "vosp": {"-1", "0", "1", "2", "3"},
        "prof": {"-1", "1", "2", "3", "4"},
        "plan": {"-1", "1", "2", "3", "4"},
        "surf": {"-1"} | set(map(str, range(1, 10))),
        "infra": {"-1"} | set(map(str, range(0, 10))),
        "situ": {"-1"} | set(map(str, range(1, 9))),
    },
    "usagers": {
        "place": {"-1"} | set(map(str, range(1, 11))),
        "catu": {"1", "2", "3"},
        "grav": {"1", "2", "3", "4"},
        "sexe": {"-1", "1", "2"},
        "trajet": {"-1", "0", "1", "2", "3", "4", "5", "9"},
        "secu1": {"-1"} | set(map(str, range(0, 10))),
        "secu2": {"-1"} | set(map(str, range(0, 10))),
        "secu3": {"-1"} | set(map(str, range(0, 10))),
        "locp": {"-1"} | set(map(str, range(0, 11))),
        "actp": {"-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B"},
        "etatp": {"-1", "1", "2", "3"},
    },
    "vehicules": {
        "senc": {"-1", "0", "1", "2", "3"},
        "catv": {"-1", "0", "1", "2", "3", "7", "10", "13", "14", "15", "16", "17", "20", "21", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "50", "60", "80", "99"},
        "obs": {"-1"} | set(map(str, range(0, 18))),
        "obsm": {"-1"} | set(map(str, range(0, 10))),
        "choc": {"-1"} | set(map(str, range(0, 10))),
        "manv": {"-1"} | set(map(str, range(0, 27))),
        "motor": {"-1"} | set(map(str, range(0, 7))),
    },
}


EXPECTED_TYPE_COLUMNS = {
    "caract": {
        "numeric": ["Num_Acc", "jour", "mois", "an", "lum", "dep", "com", "agg", "int", "atm", "col", "lat", "long"],
        "time": ["hrmn"],
        "text": ["adr"],
    },
    "lieux": {
        "numeric": ["Num_Acc", "catr", "v1", "v2", "circ", "nbv", "vosp", "prof", "pr", "pr1", "plan", "lartpc", "larrout", "surf", "infra", "situ", "vma"],
        "text": ["voie"],
    },
    "usagers": {
        "numeric": ["Num_Acc", "id_usager", "id_vehicule", "place", "catu", "grav", "sexe", "an_nais", "trajet", "secu1", "secu2", "secu3", "locp", "etatp"],
        "text": ["num_veh", "actp"],
    },
    "vehicules": {
        "numeric": ["Num_Acc", "id_vehicule", "senc", "catv", "obs", "obsm", "choc", "manv", "motor", "occutc"],
        "text": ["num_veh"],
    },
}

# Negative values are only accepted in specific columns where -1 means "unknown".
# For all other numeric fields, negative values are considered suspicious.
NEGATIVE_ALLOWED_COLUMNS = {
    "caract": {"lat", "long", "atm", "col"},
    "lieux": {"circ", "nbv", "vosp", "prof", "pr", "pr1", "plan", "larrout", "surf", "infra", "situ", "vma"},
    "usagers": {"place", "sexe", "trajet", "secu1", "secu2", "secu3", "locp", "etatp"},
    "vehicules": {"senc", "catv", "obs", "obsm", "choc", "manv", "motor"},
}

OUTLIER_COLUMNS = {
    "usagers": ["age"],
    "lieux": ["vma", "nbv", "lartpc", "larrout"],
    "vehicules": ["occutc"],
}

PROFILING_TO_CONTROLS = [
    {
        "Profiling signal": "Very sparse columns such as lartpc, occutc and v2",
        "Why it matters": "These fields are almost empty because they only apply in specific cases.",
        "Controls added": "Missing-rate dashboard, structural missingness flag, and sample inspection.",
    },
    {
        "Profiling signal": "Hidden missing values coded as -1 or sometimes 0",
        "Why it matters": "These values look numeric but often mean unknown or not applicable.",
        "Controls added": "Unknown-code dashboard and negative-value policy checks.",
    },
    {
        "Profiling signal": "Date fields are split into day, month, year and time",
        "Why it matters": "The fields can look valid separately but fail as a real calendar date.",
        "Controls added": "Month 1-12, day 1-31, full calendar date and HH:MM time checks.",
    },
    {
        "Profiling signal": "Coordinates are numeric but can still be geographically wrong",
        "Why it matters": "A valid latitude/longitude can still point outside France.",
        "Controls added": "Latitude/longitude parsing and bounding-box checks for mainland France and overseas territories.",
    },
    {
        "Profiling signal": "Speed limits and physical measures show unusual values",
        "Why it matters": "Extreme values can distort road-safety analysis.",
        "Controls added": "Range checks, standard speed-limit checks and IQR outlier detection.",
    },
    {
        "Profiling signal": "The four files are connected by accident, vehicle and user identifiers",
        "Why it matters": "Broken joins can duplicate or lose records in the final dataset.",
        "Controls added": "Primary-key uniqueness, full duplicates, foreign keys and coverage checks.",
    },
]

FRANCE_BBOXES = {
    "Mainland + Corsica": {"lat": (41.0, 51.3), "lon": (-5.5, 10.2)},
    "Guadeloupe / Saint-Martin / Saint-Barthélemy": {"lat": (15.7, 18.2), "lon": (-63.3, -60.9)},
    "Martinique": {"lat": (14.3, 14.9), "lon": (-61.3, -60.7)},
    "French Guiana": {"lat": (2.0, 6.0), "lon": (-54.8, -51.0)},
    "Réunion": {"lat": (-21.5, -20.8), "lon": (55.1, 55.9)},
    "Mayotte": {"lat": (-13.1, -12.5), "lon": (44.9, 45.4)},
    "Saint-Pierre-et-Miquelon": {"lat": (46.6, 47.1), "lon": (-56.6, -55.9)},
    "Wallis-et-Futuna": {"lat": (-14.5, -13.0), "lon": (-178.3, -176.0)},
    "French Polynesia": {"lat": (-28.0, -7.0), "lon": (-155.0, -134.0)},
    "New Caledonia": {"lat": (-23.5, -19.0), "lon": (163.0, 168.5)},
}

GRAV_LABELS = {"1": "Unharmed", "2": "Killed", "3": "Hospitalized injured", "4": "Slightly injured"}
CATU_LABELS = {"1": "Driver", "2": "Passenger", "3": "Pedestrian"}
AGG_LABELS = {"1": "Outside built-up area", "2": "Built-up area"}
LUM_LABELS = {
    "1": "Daylight",
    "2": "Twilight / dawn",
    "3": "Night without public lighting",
    "4": "Night with public lighting off",
    "5": "Night with public lighting on",
}


def clean_str_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype("string").str.strip()
        out.loc[out[col].isin(["", "nan", "NaN", "N/A", "None", "<NA>"]), col] = pd.NA
    return out


def normalize_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )


def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"": pd.NA, "N/A": pd.NA, "nan": pd.NA, "<NA>": pd.NA}),
        errors="coerce",
    )


def missing_mask(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype("string").str.strip().isin(["", "N/A", "nan", "NaN", "None", "<NA>"])


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(n / d * 100, 3)


def in_any_france_bbox(lat: float, lon: float) -> bool:
    if pd.isna(lat) or pd.isna(lon):
        return False
    for box in FRANCE_BBOXES.values():
        if box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]:
            return True
    return False


def make_rule(rule_id, dataset, dimension, severity, check, total, fail_count, description):
    status = "PASS" if int(fail_count) == 0 else ("WARN" if severity == "warning" else "FAIL")
    return {
        "rule_id": rule_id,
        "dataset": dataset,
        "dimension": dimension,
        "severity": severity,
        "status": status,
        "check": check,
        "total_records": int(total),
        "failing_records": int(fail_count),
        "failure_rate_pct": pct(int(fail_count), int(total)),
        "description": description,
    }


@st.cache_data(show_spinner=False)
def read_csv_object(file_or_path):
    return pd.read_csv(file_or_path, sep=";", dtype=str, encoding="utf-8")


def app_directory() -> Path:
    """Return the directory where this Streamlit script is stored."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def candidate_dataset_paths(default_filename: str):
    """Search common locations and name variants for a dataset file."""
    script_dir = app_directory()
    cwd = Path.cwd().resolve()
    data_dirs = [
        script_dir / DEFAULT_DATA_DIR,
        cwd / DEFAULT_DATA_DIR,
        script_dir,
        cwd,
    ]

    variants = [default_filename]
    if "(2)" not in default_filename:
        variants.append(default_filename.replace(".csv", "(2).csv"))
    else:
        variants.append(default_filename.replace("(2)", ""))

    candidates = []
    for base in data_dirs:
        for filename in variants:
            candidates.append(base / filename)

    # Also accept any file that starts with the logical prefix, for example caract-2023.csv.
    logical_prefix = default_filename.split("-")[0]
    for base in data_dirs:
        if base.exists():
            candidates.extend(sorted(base.glob(f"{logical_prefix}-*.csv")))

    seen = set()
    unique_candidates = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(path)
    return unique_candidates


def load_datasets(uploaded_files):
    datasets = {}
    errors = []
    loaded_from = {}

    for name, default_filename in DATASET_FILES.items():
        upload = uploaded_files.get(name)
        try:
            if upload is not None:
                datasets[name] = read_csv_object(upload)
                loaded_from[name] = "uploaded file"
                continue

            matching_path = next((p for p in candidate_dataset_paths(default_filename) if p.exists()), None)
            if matching_path is not None:
                datasets[name] = read_csv_object(matching_path)
                loaded_from[name] = str(matching_path)
            else:
                searched = ", ".join(str(p) for p in candidate_dataset_paths(default_filename))
                errors.append(
                    f"Missing dataset `{name}`. Upload it in the sidebar or place `{default_filename}` "
                    f"next to this app. Also searched: {searched}"
                )
        except Exception as exc:
            errors.append(f"Could not read `{name}`: {exc}")

    return datasets, errors, loaded_from


def prepare_data(raw_dfs, accident_year):
    d = {name: clean_str_df(df) for name, df in raw_dfs.items()}
    for name, df in d.items():
        for col in ["Num_Acc", "id_usager", "id_vehicule", "num_veh"]:
            if col in df.columns:
                df[col + "_norm"] = normalize_id(df[col])

    car = d["caract"].copy()
    car["lat_num"] = to_number(car["lat"])
    car["long_num"] = to_number(car["long"])
    for col in ["jour", "mois", "an"]:
        car[col + "_num"] = to_number(car[col])
    car["accident_date"] = pd.to_datetime(
        car["an"].astype("string").str.zfill(4)
        + "-"
        + car["mois"].astype("string").str.zfill(2)
        + "-"
        + car["jour"].astype("string").str.zfill(2),
        errors="coerce",
    )
    car["accident_hour"] = pd.to_datetime(car["hrmn"], format="%H:%M", errors="coerce").dt.hour
    car["coord_in_france_bbox"] = [
        in_any_france_bbox(lat, lon) for lat, lon in zip(car["lat_num"], car["long_num"])
    ]

    usa = d["usagers"].copy()
    usa["birth_year"] = to_number(usa["an_nais"])
    usa["age"] = accident_year - usa["birth_year"]
    usa["grav_label"] = usa["grav"].astype("string").str.strip().map(GRAV_LABELS).fillna(usa["grav"])
    usa["catu_label"] = usa["catu"].astype("string").str.strip().map(CATU_LABELS).fillna(usa["catu"])

    veh = d["vehicules"].copy()
    lie = d["lieux"].copy()

    return {"caract": car, "lieux": lie, "usagers": usa, "vehicules": veh}


def type_coherence_findings(data):
    findings = []
    samples = {}
    for dataset, spec in EXPECTED_TYPE_COLUMNS.items():
        df = data[dataset]
        for col in spec.get("numeric", []):
            if col not in df.columns:
                continue
            source = df[col]
            non_missing = ~missing_mask(source)
            parsed = to_number(source)
            invalid_type = non_missing & parsed.isna()
            findings.append(
                make_rule(
                    f"{dataset.upper()}_{col.upper()}_NUMERIC_TYPE",
                    dataset,
                    "Type coherence",
                    "error",
                    f"{col} can be parsed as numeric",
                    len(df),
                    invalid_type.sum(),
                    f"{col} should be numeric or empty/unknown.",
                )
            )
            if invalid_type.sum():
                samples[f"{dataset.upper()}_{col.upper()}_NUMERIC_TYPE"] = df.loc[invalid_type, [col]].head(200)

        for col in spec.get("time", []):
            if col not in df.columns:
                continue
            invalid_time = ~df[col].astype("string").str.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", na=False)
            findings.append(
                make_rule(
                    f"{dataset.upper()}_{col.upper()}_TIME_TYPE",
                    dataset,
                    "Type coherence",
                    "error",
                    f"{col} follows HH:MM time format",
                    len(df),
                    invalid_time.sum(),
                    f"{col} should be a 24-hour time string such as 08:35.",
                )
            )
            if invalid_time.sum():
                samples[f"{dataset.upper()}_{col.upper()}_TIME_TYPE"] = df.loc[invalid_time, [col]].head(200)
    return findings, samples


def negative_value_findings(data):
    """Run targeted negative-value checks.

    We do not mark every negative number as wrong because BAAC uses -1 as an
    official unknown code in several categorical variables. Instead, we check
    fields where negative values would clearly be suspicious or where the only
    accepted negative value is -1.
    """
    rules = [
        ("caract", "jour", 1, "Day cannot be negative or zero"),
        ("caract", "mois", 1, "Month cannot be negative or zero"),
        ("caract", "an", 1900, "Year must be a realistic positive year"),
        ("usagers", "an_nais", 1900, "Birth year must be a realistic positive year when populated"),
        ("lieux", "nbv", -1, "Number of lanes can be -1 unknown, but not below -1"),
        ("lieux", "pr", -1, "Road reference point can be -1 unknown, but not below -1"),
        ("lieux", "pr1", -1, "Distance from reference point can be -1 unknown, but not below -1"),
        ("lieux", "lartpc", 0, "Central reservation width cannot be negative when populated"),
        ("lieux", "larrout", -1, "Roadway width can be -1 unknown, but not below -1"),
        ("lieux", "vma", -1, "Speed limit can be -1 unknown, but not below -1"),
        ("vehicules", "occutc", 0, "Public transport occupant count cannot be negative when populated"),
    ]
    findings = []
    samples = {}
    for dataset, col, minimum, description in rules:
        df = data[dataset]
        if col not in df.columns:
            continue
        values = to_number(df[col])
        invalid = values.notna() & (values < minimum)
        findings.append(
            make_rule(
                f"{dataset.upper()}_{col.upper()}_NEGATIVE_POLICY",
                dataset,
                "Validity",
                "error",
                f"{col} respects the negative-value policy",
                len(df),
                invalid.sum(),
                description,
            )
        )
        if invalid.sum():
            keep_cols = [c for c in ["Num_Acc", col] if c in df.columns]
            samples[f"{dataset.upper()}_{col.upper()}_NEGATIVE_POLICY"] = df.loc[invalid, keep_cols].head(200)
    return findings, samples


def iqr_outlier_findings(data):
    findings = []
    samples = {}
    for dataset, cols in OUTLIER_COLUMNS.items():
        df = data[dataset]
        for col in cols:
            if col not in df.columns:
                continue
            values = to_number(df[col]) if col != "age" else df[col]
            values = pd.to_numeric(values, errors="coerce")
            valid = values.dropna()
            # Ignore -1 unknown codes for outlier calculation.
            valid = valid[valid >= 0]
            if len(valid) < 10:
                continue
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower = max(0, q1 - 1.5 * iqr)
            upper = q3 + 1.5 * iqr
            outlier = (values < lower) | (values > upper)
            outlier = outlier.fillna(False)
            findings.append(
                make_rule(
                    f"{dataset.upper()}_{col.upper()}_IQR_OUTLIER",
                    dataset,
                    "Outlier",
                    "warning",
                    f"{col} has no statistical outlier based on IQR",
                    len(df),
                    outlier.sum(),
                    f"Values outside [{round(lower, 2)}, {round(upper, 2)}] are statistical outliers. They are not always wrong, but they should be reviewed.",
                )
            )
            if outlier.sum():
                keep_cols = [c for c in ["Num_Acc", col] if c in df.columns]
                samples[f"{dataset.upper()}_{col.upper()}_IQR_OUTLIER"] = df.loc[outlier, keep_cols].head(200)
    return findings, samples


def render_explanation_box(title, text):
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.7rem; border: 1px solid rgba(128,128,128,0.25); background-color: rgba(128,128,128,0.06); margin-bottom: 1rem;">
        <b>{title}</b><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_findings(raw_dfs, accident_year):
    d = prepare_data(raw_dfs, accident_year)
    car, lie, usa, veh = d["caract"], d["lieux"], d["usagers"], d["vehicules"]
    findings = []
    samples = {}

    for name, cols in EXPECTED_COLUMNS.items():
        missing_cols = [c for c in cols if c not in raw_dfs[name].columns]
        findings.append(
            make_rule(
                f"{name.upper()}_SCHEMA_COLUMNS",
                name,
                "Schema",
                "error",
                "Expected columns are present",
                len(cols),
                len(missing_cols),
                f"Missing columns: {missing_cols}" if missing_cols else "All expected columns are present.",
            )
        )

    type_findings, type_samples = type_coherence_findings(d)
    findings.extend(type_findings)
    samples.update(type_samples)

    negative_findings, negative_samples = negative_value_findings(d)
    findings.extend(negative_findings)
    samples.update(negative_samples)

    for name, pk in {"caract": ["Num_Acc_norm"], "usagers": ["id_usager_norm"], "vehicules": ["id_vehicule_norm"]}.items():
        df = d[name]
        duplicate = df.duplicated(pk, keep=False)
        findings.append(make_rule(f"{name.upper()}_PK_UNIQUE", name, "Uniqueness", "error", f"Primary key {pk} is unique", len(df), duplicate.sum(), "Primary key duplicates must be investigated."))
        if duplicate.sum():
            samples[f"{name.upper()}_PK_UNIQUE"] = df.loc[duplicate].head(200)

        null_pk = df[pk].isna().any(axis=1)
        findings.append(make_rule(f"{name.upper()}_PK_NOT_NULL", name, "Completeness", "error", f"Primary key {pk} is not null", len(df), null_pk.sum(), "Primary key cannot contain null values."))

    duplicate_lieux_accident = lie.duplicated(["Num_Acc_norm"], keep=False)
    findings.append(make_rule("LIEUX_ACCIDENT_GRAIN", "lieux", "Uniqueness", "warning", "One location row per accident", len(lie), duplicate_lieux_accident.sum(), "Multiple location rows for the same accident ID should be validated against the intended data grain."))
    samples["LIEUX_ACCIDENT_GRAIN"] = lie.loc[duplicate_lieux_accident, ["Num_Acc", "catr", "voie", "pr", "pr1", "vma"]].head(200)

    for name, df in d.items():
        full_duplicate = df[EXPECTED_COLUMNS[name]].duplicated(keep=False) if set(EXPECTED_COLUMNS[name]).issubset(df.columns) else df.duplicated(keep=False)
        findings.append(make_rule(f"{name.upper()}_DUPLICATE_ROWS", name, "Uniqueness", "error", "No fully duplicated rows", len(df), full_duplicate.sum(), "Fully duplicated rows should be removed or justified."))
        if full_duplicate.sum():
            samples[f"{name.upper()}_DUPLICATE_ROWS"] = df.loc[full_duplicate, EXPECTED_COLUMNS[name]].head(200)

    accident_ids = set(car["Num_Acc_norm"].dropna())
    for name, df in {"lieux": lie, "usagers": usa, "vehicules": veh}.items():
        orphan = ~df["Num_Acc_norm"].isin(accident_ids)
        findings.append(make_rule(f"{name.upper()}_FK_ACCIDENT_EXISTS", name, "Referential integrity", "error", "Every Num_Acc exists in characteristics", len(df), orphan.sum(), f"Every row in {name} must reference an accident in caract."))
        if orphan.sum():
            samples[f"{name.upper()}_FK_ACCIDENT_EXISTS"] = df.loc[orphan].head(200)

    for name, df in {"lieux": lie, "usagers": usa, "vehicules": veh}.items():
        represented = set(df["Num_Acc_norm"].dropna())
        missing_child = ~car["Num_Acc_norm"].isin(represented)
        findings.append(make_rule(f"CARACT_ACCIDENT_HAS_{name.upper()}", "caract", "Referential integrity", "error", f"Every accident has at least one {name} row", len(car), missing_child.sum(), f"Every accident should be represented in {name}."))

    vehicle_key_cols = ["Num_Acc_norm", "id_vehicule_norm", "num_veh_norm"]
    vehicle_keys = veh[vehicle_key_cols].dropna().drop_duplicates().assign(_exists=True)
    user_vehicle_check = usa[vehicle_key_cols].merge(vehicle_keys, how="left", on=vehicle_key_cols, sort=False)
    orphan_user_vehicle = ~user_vehicle_check["_exists"].eq(True)
    orphan_user_vehicle.index = usa.index
    findings.append(make_rule("USAGERS_VEHICLE_FK_EXISTS", "usagers", "Referential integrity", "error", "Every user vehicle key exists in vehicles", len(usa), orphan_user_vehicle.sum(), "The tuple Num_Acc + id_vehicule + num_veh in users must exist in vehicles."))
    if orphan_user_vehicle.sum():
        samples["USAGERS_VEHICLE_FK_EXISTS"] = usa.loc[orphan_user_vehicle, ["Num_Acc", "id_usager", "id_vehicule", "num_veh"]].head(200)

    user_vehicle_keys = usa[vehicle_key_cols].dropna().drop_duplicates().assign(_exists=True)
    vehicle_user_check = veh[vehicle_key_cols].merge(user_vehicle_keys, how="left", on=vehicle_key_cols, sort=False)
    vehicle_without_user = ~vehicle_user_check["_exists"].eq(True)
    vehicle_without_user.index = veh.index
    findings.append(make_rule("VEHICLES_HAVE_AT_LEAST_ONE_USER", "vehicules", "Referential integrity", "warning", "Every vehicle has at least one associated user", len(veh), vehicle_without_user.sum(), "Vehicles without users may represent parked/unoccupied vehicles or missing user records."))
    samples["VEHICLES_HAVE_AT_LEAST_ONE_USER"] = veh.loc[vehicle_without_user, ["Num_Acc", "id_vehicule", "num_veh", "catv", "occutc"]].head(200)

    invalid_month = car["mois_num"].isna() | (car["mois_num"] < 1) | (car["mois_num"] > 12)
    findings.append(make_rule("CARACT_MONTH_1_TO_12", "caract", "Validity", "error", "Month is between 1 and 12", len(car), invalid_month.sum(), "The mois field must be a number from 1 to 12."))
    if invalid_month.sum():
        samples["CARACT_MONTH_1_TO_12"] = car.loc[invalid_month, ["Num_Acc", "mois"]].head(200)

    invalid_day = car["jour_num"].isna() | (car["jour_num"] < 1) | (car["jour_num"] > 31)
    findings.append(make_rule("CARACT_DAY_1_TO_31", "caract", "Validity", "error", "Day is between 1 and 31", len(car), invalid_day.sum(), "The jour field must be a number from 1 to 31."))
    if invalid_day.sum():
        samples["CARACT_DAY_1_TO_31"] = car.loc[invalid_day, ["Num_Acc", "jour"]].head(200)

    invalid_date = car["accident_date"].isna()
    findings.append(make_rule("CARACT_VALID_ACCIDENT_DATE", "caract", "Validity", "error", "Valid accident date", len(car), invalid_date.sum(), "jour, mois and an must form a valid calendar date, for example 31/02 is not valid."))
    invalid_time = ~car["hrmn"].astype("string").str.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", na=False)
    findings.append(make_rule("CARACT_VALID_TIME_FORMAT", "caract", "Validity", "error", "Valid HH:MM accident time", len(car), invalid_time.sum(), "hrmn must be a valid 24-hour HH:MM value."))
    invalid_year = car["an_num"].ne(accident_year)
    findings.append(make_rule("CARACT_YEAR_EQUALS_FILE_YEAR", "caract", "Validity", "error", f"Accident year equals {accident_year}", len(car), invalid_year.sum(), "Annual files should contain only records for the selected year."))

    missing_birth_year = usa["age"].isna()
    invalid_age = (~missing_birth_year) & ((usa["age"] < 0) | (usa["age"] > 110))
    findings.append(make_rule("USAGERS_AGE_BETWEEN_0_AND_110", "usagers", "Validity", "error", "Age is between 0 and 110", len(usa), invalid_age.sum(), "Derived age = accident year - birth year must be within 0..110."))
    if invalid_age.sum():
        samples["USAGERS_AGE_BETWEEN_0_AND_110"] = usa.loc[invalid_age, ["Num_Acc", "id_usager", "an_nais", "age"]].head(200)
    findings.append(make_rule("USAGERS_BIRTH_YEAR_MISSING", "usagers", "Completeness", "warning", "Birth year is populated", len(usa), missing_birth_year.sum(), "Missing birth year prevents age-based analysis."))

    invalid_latitude = car["lat_num"].isna() | (car["lat_num"] < -90) | (car["lat_num"] > 90)
    invalid_longitude = car["long_num"].isna() | (car["long_num"] < -180) | (car["long_num"] > 180)
    findings.append(make_rule("CARACT_LATITUDE_VALID", "caract", "Validity", "error", "Latitude is numeric and in [-90, 90]", len(car), invalid_latitude.sum(), "Latitude must be valid."))
    findings.append(make_rule("CARACT_LONGITUDE_VALID", "caract", "Validity", "error", "Longitude is numeric and in [-180, 180]", len(car), invalid_longitude.sum(), "Longitude must be valid."))
    outside_france = ~car["coord_in_france_bbox"]
    findings.append(make_rule("CARACT_COORDINATES_IN_FRANCE", "caract", "Geographic validity", "error", "Coordinates are within France or French overseas bounding boxes", len(car), outside_france.sum(), "Coordinates outside accepted French bounding boxes should be corrected or justified."))
    samples["CARACT_COORDINATES_IN_FRANCE"] = car.loc[outside_france, ["Num_Acc", "dep", "com", "adr", "lat", "long"]].head(200)

    dep_prefix = np.where(car["dep"].astype("string").str.len() == 3, car["com"].astype("string").str[:3], car["com"].astype("string").str[:2])
    dep_com_mismatch = car["dep"].astype("string").ne(pd.Series(dep_prefix, index=car.index).astype("string"))
    findings.append(make_rule("CARACT_DEP_COM_CONSISTENCY", "caract", "Consistency", "error", "Department code matches commune prefix", len(car), dep_com_mismatch.sum(), "The department code should match the leading part of the commune code."))
    if dep_com_mismatch.sum():
        samples["CARACT_DEP_COM_CONSISTENCY"] = car.loc[dep_com_mismatch, ["Num_Acc", "dep", "com"]].head(200)

    for name, domain_rules in DOMAINS.items():
        df = d[name]
        for col, allowed_values in domain_rules.items():
            values = df[col].astype("string").str.replace("\u00a0", "", regex=False).str.strip()
            invalid_domain = ~(values.isna() | values.isin(allowed_values))
            findings.append(make_rule(f"{name.upper()}_{col.upper()}_DOMAIN", name, "Validity", "error", f"{col} values are in authorized domain", len(df), invalid_domain.sum(), f"{col} must belong to its authorized BAAC code list."))
            if invalid_domain.sum():
                samples[f"{name.upper()}_{col.upper()}_DOMAIN"] = df.loc[invalid_domain, [col]].head(200)

    numeric_rules = [
        ("lieux", "nbv", "Number of lanes is non-negative or -1 unknown", -1, None, "warning"),
        ("lieux", "pr", "Road reference point is non-negative or -1 unknown", -1, None, "warning"),
        ("lieux", "pr1", "Distance from reference point is non-negative or -1 unknown", -1, None, "warning"),
        ("lieux", "lartpc", "Central reservation width is positive when populated", 0, None, "warning"),
        ("lieux", "larrout", "Roadway width is positive or -1 unknown", -1, None, "warning"),
        ("lieux", "vma", "Maximum authorized speed is between -1 and 130", -1, 130, "error"),
        ("vehicules", "occutc", "Public transport occupant count is non-negative when populated", 0, None, "warning"),
    ]
    for dataset, col, label, minimum, maximum, severity in numeric_rules:
        df = d[dataset]
        values = to_number(df[col])
        invalid = values.isna() & df[col].notna()
        if minimum == -1:
            invalid = invalid | (values < -1)
        elif col in {"lartpc", "larrout"}:
            invalid = invalid | ((values <= 0) & values.notna())
        else:
            invalid = invalid | ((values < minimum) & values.notna())
        if maximum is not None:
            invalid = invalid | (values > maximum)
        findings.append(make_rule(f"{dataset.upper()}_{col.upper()}_NUMERIC_RANGE", dataset, "Validity", severity, label, len(df), invalid.sum(), label))
        if invalid.sum():
            samples[f"{dataset.upper()}_{col.upper()}_NUMERIC_RANGE"] = df.loc[invalid, ["Num_Acc", col]].head(200)

    vma = to_number(lie["vma"])
    standard_vma = {-1, 5, 10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 110, 130}
    non_standard_vma = ~(vma.isna() | vma.isin(standard_vma))
    findings.append(make_rule("LIEUX_VMA_STANDARD_SPEED_SET", "lieux", "Validity", "warning", "vma belongs to a standard speed-limit set", len(lie), non_standard_vma.sum(), "Non-standard speed limits can be valid locally, but extreme values should be reviewed."))
    samples["LIEUX_VMA_STANDARD_SPEED_SET"] = lie.loc[non_standard_vma, ["Num_Acc", "catr", "voie", "vma"]].head(200)

    pedestrian_wrong_place = (usa["catu"].astype("string").str.strip() == "3") & (usa["place"].astype("string").str.strip() != "10")
    findings.append(make_rule("USAGERS_PEDESTRIAN_PLACE_10", "usagers", "Consistency", "error", "Pedestrians have place=10", len(usa), pedestrian_wrong_place.sum(), "Pedestrians should have place=10."))
    if pedestrian_wrong_place.sum():
        samples["USAGERS_PEDESTRIAN_PLACE_10"] = usa.loc[pedestrian_wrong_place, ["Num_Acc", "id_usager", "catu", "place"]].head(200)

    non_pedestrian = usa["catu"].astype("string").str.strip() != "3"
    locp = usa["locp"].astype("string").str.strip()
    actp = usa["actp"].astype("string").str.strip()
    etatp = usa["etatp"].astype("string").str.strip()
    non_pedestrian_ped_fields = non_pedestrian & (~locp.isin(["-1", "0"]) | ~actp.isin(["-1", "0"]) | ~etatp.isin(["-1", "0"]))
    findings.append(make_rule("USAGERS_NON_PEDESTRIAN_PED_FIELDS_EMPTY", "usagers", "Consistency", "warning", "Non-pedestrians have pedestrian-specific fields set to -1 or 0", len(usa), non_pedestrian_ped_fields.sum(), "Non-pedestrian records should normally not contain pedestrian location/action/status details."))
    samples["USAGERS_NON_PEDESTRIAN_PED_FIELDS_EMPTY"] = usa.loc[non_pedestrian_ped_fields, ["Num_Acc", "id_usager", "catu", "locp", "actp", "etatp"]].head(200)

    vehicle_num_format = ~veh["num_veh"].astype("string").str.match(r"^[A-Z]{1,2}[0-9]{2}$", na=False)
    user_vehicle_num_format = ~usa["num_veh"].astype("string").str.match(r"^[A-Z]{1,2}[0-9]{2}$", na=False)
    findings.append(make_rule("VEHICLES_NUM_VEH_FORMAT", "vehicules", "Validity", "error", "num_veh follows pattern A01, B01, AB01, ...", len(veh), vehicle_num_format.sum(), "Vehicle number should match one or two uppercase letters followed by two digits."))
    findings.append(make_rule("USAGERS_NUM_VEH_FORMAT", "usagers", "Validity", "error", "num_veh follows pattern A01, B01, AB01, ...", len(usa), user_vehicle_num_format.sum(), "User vehicle number should match one or two uppercase letters followed by two digits."))
    if vehicle_num_format.sum():
        samples["VEHICLES_NUM_VEH_FORMAT"] = veh.loc[vehicle_num_format, ["Num_Acc", "id_vehicule", "num_veh"]].head(200)
    if user_vehicle_num_format.sum():
        samples["USAGERS_NUM_VEH_FORMAT"] = usa.loc[user_vehicle_num_format, ["Num_Acc", "id_usager", "id_vehicule", "num_veh"]].head(200)

    users_per_accident = usa.groupby("Num_Acc_norm").size()
    vehicles_per_accident = veh.groupby("Num_Acc_norm").size()
    high_user_accidents = users_per_accident[users_per_accident > 20]
    high_vehicle_accidents = vehicles_per_accident[vehicles_per_accident > 8]
    findings.append(make_rule("ACCIDENT_USER_COUNT_OUTLIER", "usagers", "Anomaly", "warning", "No accident has more than 20 users", len(users_per_accident), len(high_user_accidents), "Very high numbers of users per accident should be reviewed."))
    findings.append(make_rule("ACCIDENT_VEHICLE_COUNT_OUTLIER", "vehicules", "Anomaly", "warning", "No accident has more than 8 vehicles", len(vehicles_per_accident), len(high_vehicle_accidents), "Very high numbers of vehicles per accident should be reviewed."))
    samples["ACCIDENT_USER_COUNT_OUTLIER"] = high_user_accidents.sort_values(ascending=False).reset_index(name="user_count").head(200)
    samples["ACCIDENT_VEHICLE_COUNT_OUTLIER"] = high_vehicle_accidents.sort_values(ascending=False).reset_index(name="vehicle_count").head(200)

    outlier_findings, outlier_samples = iqr_outlier_findings(d)
    findings.extend(outlier_findings)
    samples.update(outlier_samples)

    return pd.DataFrame(findings), samples, d


def missing_profile(data):
    records = []
    for name, df in data.items():
        base_cols = [c for c in EXPECTED_COLUMNS[name] if c in df.columns]
        for col in base_cols:
            m = missing_mask(df[col]).sum()
            unknown = df[col].astype("string").str.replace("\u00a0", "", regex=False).str.strip().eq("-1").sum()
            zero = df[col].astype("string").str.replace("\u00a0", "", regex=False).str.strip().eq("0").sum()
            records.append({
                "dataset": name,
                "column": col,
                "missing_records": int(m),
                "missing_rate_pct": pct(int(m), len(df)),
                "unknown_code_minus_1": int(unknown),
                "unknown_rate_pct": pct(int(unknown), len(df)),
                "zero_records": int(zero),
                "zero_rate_pct": pct(int(zero), len(df)),
                "unique_values": int(df[col].nunique(dropna=True)),
            })
    return pd.DataFrame(records)


def render_metric_cards(findings, data):
    total_rules = len(findings)
    passed = int((findings["status"] == "PASS").sum())
    failed = int((findings["status"] == "FAIL").sum())
    warned = int((findings["status"] == "WARN").sum())
    score = round(passed / total_rules * 100, 1) if total_rules else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quality score", f"{score}%")
    c2.metric("Rules", total_rules)
    c3.metric("Passed", passed)
    c4.metric("Failed", failed)
    c5.metric("Warnings", warned)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents", f"{len(data['caract']):,}")
    c2.metric("Location rows", f"{len(data['lieux']):,}")
    c3.metric("Users", f"{len(data['usagers']):,}")
    c4.metric("Vehicles", f"{len(data['vehicules']):,}")


def main():
    st.title("BAAC 2024 Data Quality Report")
    st.caption("Automated profiling interpretation, quality controls, anomaly detection and visual checks for the French road accident datasets.")

    st.info(
        "This report loads the four BAAC CSV files automatically from the `data` folder when they are available. "
        "You only need the upload buttons when you want to test another file quickly."
    )
    with st.expander("How to use another dataset without uploading every time", expanded=False):
        st.markdown(
            """
            The easiest option is to replace the CSV files in the `data` folder next to this app.

            The second option is to edit the small **USER DATA SETTINGS** block at the top of the Python file:

            ```python
            DEFAULT_DATA_DIR = "data"
            DATASET_FILES = {
                "caract": "caract-2024.csv",
                "lieux": "lieux-2024.csv",
                "usagers": "usagers-2024.csv",
                "vehicules": "vehicules-2024.csv",
            }
            ```

            For example, for 2023 files, you can replace the filenames by `caract-2023.csv`, `lieux-2023.csv`, `usagers-2023.csv` and `vehicules-2023.csv`. The rest of the checks will run the same way.
            """
        )

    with st.sidebar:
        st.header("Input data")
        st.write("The app first looks for files in the local `data` folder. Upload is optional and only overrides local files for the current session.")
        uploaded_files = {
            name: st.file_uploader(f"{name} CSV", type=["csv"], key=f"upload_{name}")
            for name in DATASET_FILES
        }
        accident_year = st.number_input("Accident year", min_value=2000, max_value=2100, value=2024, step=1)
        sample_size = st.slider("Map sample size", min_value=1_000, max_value=50_000, value=10_000, step=1_000)
        show_passed = st.checkbox("Show passed controls in quality table", value=False)

    raw_data, errors, loaded_from = load_datasets(uploaded_files)
    if errors or set(raw_data.keys()) != set(DATASET_FILES.keys()):
        missing = sorted(set(DATASET_FILES.keys()) - set(raw_data.keys()))
        st.error("The app cannot start because one or more required datasets are missing.")
        if missing:
            st.warning(f"Missing logical datasets: {', '.join(missing)}")
        for error in errors:
            st.error(error)
        st.info(
            "Launch the application with: `python -m streamlit run streamlit_app.py`. "
            "Then upload the four CSV files in the sidebar, or put them in the same folder as `streamlit_app.py`."
        )
        st.stop()

    with st.sidebar.expander("Loaded files", expanded=True):
        for name, source in loaded_from.items():
            st.write(f"**{name}**: {source}")

    st.success("The four required datasets were found. The report can now run the quality checks.")
    loaded_summary = pd.DataFrame([
        {"dataset": name, "rows": len(df), "columns": len(df.columns), "source": loaded_from.get(name, "unknown")}
        for name, df in raw_data.items()
    ]).sort_values("dataset")
    st.dataframe(loaded_summary, use_container_width=True, hide_index=True)

    st.info(
        "If you want to use another dataset without uploading files every time, replace the four CSV files in the `data` folder, "
        "or edit the `DATASET_FILES` block at the top of `streamlit_app.py`."
    )

    try:
        with st.spinner("Running data quality checks. This can take a few seconds on the first run..."):
            findings, samples, data = compute_findings(raw_data, accident_year)
            missing_df = missing_profile(data)
    except Exception as exc:
        st.error("The input files were found, but the quality checks failed before the dashboard could be displayed.")
        st.exception(exc)
        st.stop()

    render_metric_cards(findings, data)

    tabs = st.tabs([
        "Executive summary",
        "Checks overview",
        "Profiling highlights",
        "Quality controls",
        "Geographic checks",
        "Referential integrity",
        "Missing / unknown values",
        "Distributions",
        "Anomaly samples",
        "Exports",
    ])

    with tabs[0]:
        st.subheader("Executive summary")
        non_pass = findings[findings["status"] != "PASS"].sort_values(["status", "failure_rate_pct"], ascending=[True, False])
        if non_pass.empty:
            st.success("All controls passed.")
        else:
            st.warning(f"{len(non_pass)} controls require attention.")
            st.dataframe(non_pass, use_container_width=True, hide_index=True)

        status_counts = findings["status"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.bar(status_counts, x="status", y="count", text="count", title="Control status distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Main interpretation**

            We are not only checking whether the files open correctly. We check whether the data can be trusted for analysis.

            The main things we look for are simple: unique primary keys, valid dates, coordinates located in France, coherent data types, impossible ages, unexpected negative values, outliers, and broken links between the four tables.

            The YData Profiling reports helped us decide where to focus. They showed very sparse fields, hidden missing values coded as `-1`, some important missing information such as birth year, and numeric fields where extreme values can appear.
            """
        )

    with tabs[1]:
        st.subheader("Checks overview")
        render_explanation_box(
            "Why we run these checks",
            "The YData Profiling report helped us find where the dataset is fragile: sparse columns, hidden missing values, dates stored as separate fields, coordinates that may be valid numbers but wrong locations, and unusual numeric values. We turn these observations into clear data quality controls."
        )

        st.markdown("#### From profiling observation to data quality control")
        st.dataframe(pd.DataFrame(PROFILING_TO_CONTROLS), use_container_width=True, hide_index=True)

        st.markdown("#### Minimum controls covered in this report")
        minimum_checks = pd.DataFrame([
            {"Control family": "Primary key duplicates", "Examples": "Num_Acc, id_usager, id_vehicule", "Why we check it": "A primary key must identify one row only."},
            {"Control family": "Date validity", "Examples": "month 1-12, day 1-31, valid calendar date, HH:MM time", "Why we check it": "Wrong dates can break time analysis."},
            {"Control family": "Coordinates in France", "Examples": "lat, long", "Why we check it": "Coordinates can be numeric but still point outside France."},
            {"Control family": "Data type coherence", "Examples": "numeric codes, IDs, dates and time", "Why we check it": "A column should contain values in the expected format."},
            {"Control family": "Outliers", "Examples": "age, vma, number of users, number of vehicles", "Why we check it": "Extreme values can be real, but they need review."},
            {"Control family": "Age over 110", "Examples": "accident year - birth year", "Why we check it": "Impossible ages should not enter demographic indicators."},
            {"Control family": "Negative values", "Examples": "physical measures, counts, coded -1 values", "Why we check it": "Only documented unknown codes may be negative."},
            {"Control family": "Referential integrity", "Examples": "accident and vehicle joins", "Why we check it": "Broken links can lose or duplicate records."},
        ])
        st.dataframe(minimum_checks, use_container_width=True, hide_index=True)

        st.markdown("#### Error dashboard")
        non_pass = findings[findings["status"] != "PASS"].copy()
        if non_pass.empty:
            st.success("No failing or warning control was found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    non_pass.groupby(["dataset", "status"]).size().reset_index(name="controls"),
                    x="dataset",
                    y="controls",
                    color="status",
                    text="controls",
                    title="Non-passing controls by dataset",
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(
                    non_pass.groupby(["dimension", "status"]).size().reset_index(name="controls"),
                    x="dimension",
                    y="controls",
                    color="status",
                    text="controls",
                    title="Non-passing controls by quality dimension",
                )
                st.plotly_chart(fig, use_container_width=True)

            heat = non_pass.pivot_table(index="dataset", columns="dimension", values="failing_records", aggfunc="sum", fill_value=0)
            fig = px.imshow(heat, text_auto=True, aspect="auto", title="Failing records heatmap: dataset x quality dimension")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Profiling highlights")
        col1, col2 = st.columns(2)
        with col1:
            shapes = pd.DataFrame(
                [{"dataset": name, "rows": len(df), "columns": len(EXPECTED_COLUMNS[name])} for name, df in data.items()]
            )
            st.dataframe(shapes, use_container_width=True, hide_index=True)

            st.markdown(
                """
                **What we noticed from the profiling reports**

                - `caract` is the accident table. It is mostly complete, but address and coordinate quality still matter because they drive the map analysis.
                - `lieux` is the most sparse table. Columns such as `lartpc`, `v2` and `voie` have many missing values. We treat this as structural missingness when the field is not useful for most accidents.
                - `usagers` looks complete at first, but many fields use codes like `-1`. We should read these values as unknown or not applicable, not as real numbers.
                - `vehicules.occutc` is almost always empty. That is expected because it mainly applies to public transport vehicles.
                - Because of these observations, the report focuses on missingness, hidden unknown values, date validity, coordinates, outliers, negative values and table joins.
                """
            )
        with col2:
            missing_top = missing_df.sort_values("missing_rate_pct", ascending=False).head(15)
            fig = px.bar(missing_top, x="missing_rate_pct", y="dataset", color="column", orientation="h", title="Top missing-value rates by column")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Zero and unknown-code concentration")
        zero_unknown = missing_df.assign(
            signal=lambda x: x["unknown_rate_pct"] + x["zero_rate_pct"]
        ).sort_values("signal", ascending=False).head(20)
        fig = px.bar(
            zero_unknown,
            x="signal",
            y="column",
            color="dataset",
            orientation="h",
            title="Columns most affected by zero and -1 coded values",
            labels={"signal": "zero rate + -1 rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Quality controls")
        render_explanation_box(
            "How to read this page",
            "Each line is one rule. PASS means no issue was found. FAIL means the rule found records that should be corrected or justified. WARN means the value can be valid, but it deserves attention before using it in dashboards or models."
        )
        table = findings.copy()
        if not show_passed:
            table = table[table["status"] != "PASS"]
        selected_status = st.multiselect("Filter status", sorted(findings["status"].unique()), default=sorted(table["status"].unique()))
        selected_dataset = st.multiselect("Filter dataset", sorted(findings["dataset"].unique()), default=sorted(findings["dataset"].unique()))
        selected_dimension = st.multiselect("Filter dimension", sorted(findings["dimension"].unique()), default=sorted(findings["dimension"].unique()))
        filtered = table[
            table["status"].isin(selected_status)
            & table["dataset"].isin(selected_dataset)
            & table["dimension"].isin(selected_dimension)
        ].sort_values(["status", "failure_rate_pct"], ascending=[True, False])
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        if not filtered.empty:
            fig = px.bar(
                filtered.sort_values("failing_records", ascending=False).head(20),
                x="failing_records",
                y="check",
                color="dataset",
                orientation="h",
                title="Top controls by number of failing records",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Geographic checks")
        car = data["caract"].copy()
        invalid_geo = car[~car["coord_in_france_bbox"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows with coordinates", f"{car[['lat_num', 'long_num']].dropna().shape[0]:,}")
        c2.metric("Outside accepted France boxes", f"{len(invalid_geo):,}")
        c3.metric("Coordinate failure rate", f"{pct(len(invalid_geo), len(car))}%")

        st.markdown("We check coordinates in two steps. First, latitude and longitude must be valid numbers. Then, the point must fall inside mainland France or a French overseas territory. Rows outside the boxes may be caused by inverted latitude/longitude, wrong signs, or typing errors.")
        if not invalid_geo.empty:
            st.dataframe(invalid_geo[["Num_Acc", "dep", "com", "adr", "lat", "long"]], use_container_width=True, hide_index=True)

        map_df = car[["Num_Acc", "lat_num", "long_num", "dep", "coord_in_france_bbox"]].dropna().copy()
        if len(map_df) > sample_size:
            map_df = map_df.sample(sample_size, random_state=42)
        fig = px.scatter_geo(
            map_df,
            lat="lat_num",
            lon="long_num",
            color="coord_in_france_bbox",
            hover_name="Num_Acc",
            hover_data=["dep"],
            title="Accident coordinates sample",
        )
        fig.update_geos(showcountries=True, projection_type="natural earth")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Referential integrity")
        ri = findings[findings["dimension"] == "Referential integrity"].sort_values("status")
        st.dataframe(ri, use_container_width=True, hide_index=True)

        car_ids = set(data["caract"]["Num_Acc_norm"])
        coverage = []
        for name in ["lieux", "usagers", "vehicules"]:
            child_ids = set(data[name]["Num_Acc_norm"])
            coverage.append({
                "child_dataset": name,
                "accidents_in_child": len(child_ids),
                "accidents_in_caract": len(car_ids),
                "missing_in_child": len(car_ids - child_ids),
                "orphan_child_rows": int((~data[name]["Num_Acc_norm"].isin(car_ids)).sum()),
            })
        st.dataframe(pd.DataFrame(coverage), use_container_width=True, hide_index=True)

        users_per_acc = data["usagers"].groupby("Num_Acc_norm").size().reset_index(name="users")
        vehicles_per_acc = data["vehicules"].groupby("Num_Acc_norm").size().reset_index(name="vehicles")
        merged_counts = users_per_acc.merge(vehicles_per_acc, on="Num_Acc_norm", how="outer").fillna(0)
        fig = px.scatter(merged_counts, x="vehicles", y="users", title="Users vs vehicles per accident", hover_name="Num_Acc_norm")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.subheader("Missing and semantic unknown values")
        st.dataframe(missing_df.sort_values(["missing_rate_pct", "unknown_rate_pct"], ascending=False), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(missing_df.sort_values("missing_rate_pct", ascending=False).head(20), x="missing_rate_pct", y="column", color="dataset", orientation="h", title="Top missing rates")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(missing_df.sort_values("unknown_rate_pct", ascending=False).head(20), x="unknown_rate_pct", y="column", color="dataset", orientation="h", title="Top -1 unknown-code rates")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[7]:
        st.subheader("Key distributions")
        car = data["caract"]
        usa = data["usagers"]
        veh = data["vehicules"]
        lie = data["lieux"]

        month_counts = car["mois"].value_counts().sort_index().reset_index()
        month_counts.columns = ["month", "accidents"]
        fig = px.bar(month_counts, x="month", y="accidents", title="Accidents by month")
        st.plotly_chart(fig, use_container_width=True)

        hour_counts = car["accident_hour"].value_counts().sort_index().reset_index()
        hour_counts.columns = ["hour", "accidents"]
        fig = px.line(hour_counts, x="hour", y="accidents", markers=True, title="Accidents by hour")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            severity = usa["grav_label"].value_counts().reset_index()
            severity.columns = ["severity", "users"]
            fig = px.bar(severity, x="severity", y="users", title="User severity distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            age_data = usa["age"].dropna()
            fig = px.histogram(age_data, nbins=50, title="Age distribution")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            top_vehicle_cat = veh["catv"].value_counts().head(15).reset_index()
            top_vehicle_cat.columns = ["catv", "vehicles"]
            fig = px.bar(top_vehicle_cat, x="catv", y="vehicles", title="Top vehicle categories")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            vma = to_number(lie["vma"]).dropna()
            fig = px.histogram(vma, nbins=40, title="Speed limit distribution (vma)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[8]:
        st.subheader("Anomaly samples")
        sample_keys = sorted(samples.keys())
        selected_rule = st.selectbox("Select a rule to inspect sample rows", sample_keys)
        st.write(findings.loc[findings["rule_id"] == selected_rule, ["status", "dataset", "dimension", "check", "description"]])
        sample = samples.get(selected_rule, pd.DataFrame())
        if sample.empty:
            st.info("No sample rows for this rule.")
        else:
            st.dataframe(sample, use_container_width=True, hide_index=True)

    with tabs[9]:
        st.subheader("Export report artifacts")
        st.download_button(
            "Download quality findings as CSV",
            findings.to_csv(index=False).encode("utf-8"),
            file_name="baac_2024_quality_findings.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download missing-value profile as CSV",
            missing_df.to_csv(index=False).encode("utf-8"),
            file_name="baac_2024_missing_profile.csv",
            mime="text/csv",
        )

        markdown_summary = f"""
# BAAC 2024 Data Quality Report

## Summary
- Rules evaluated: {len(findings)}
- Passed: {(findings['status'] == 'PASS').sum()}
- Failed: {(findings['status'] == 'FAIL').sum()}
- Warnings: {(findings['status'] == 'WARN').sum()}

## Main checks covered
- Primary key duplicates
- Month 1-12, day 1-31, valid calendar date and HH:MM time
- Coordinates inside France and French overseas bounding boxes
- Data type coherence
- Outliers and extreme values
- Age between 0 and 110
- Negative-value policy
- Referential integrity between tables

## Non-passing controls
{findings[findings['status'] != 'PASS'].to_markdown(index=False)}
"""
        st.download_button(
            "Download Markdown summary",
            markdown_summary.encode("utf-8"),
            file_name="baac_2024_data_quality_summary.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
