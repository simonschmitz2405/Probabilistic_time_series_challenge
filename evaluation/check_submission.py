import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configue for saving
data_date = datetime.now().strftime('%Y%m%d') 
Path = Path(__file__).resolve().parent.parent

# Get paths
path = Path / "output" / f"{data_date}_RobbStark.csv"

# Read data
forecast = pd.read_csv(path)


def check_df(df):
    EXPECTED_COLS = ["forecast_date", "target", "horizon", "q0.025", "q0.25",
                    "q0.5", "q0.75", "q0.975"]
    LEN_EXP_COLS = len(EXPECTED_COLS)

    # if exclude_weather == True:
    #     print("Excluding weather variables!")
    #     TARGETS = ["DAX", "energy"]
    # else:
    TARGETS = ["bikes", "energy", "no2"]

    TARGET_VALS = dict(bikes = [str(i) + " day" for i in range(1,7)],
                       energy = [str(i) + " hour" for i in (36,40,44,60,64,68)],
                       no2 = [str(i) + " hour" for i in (36,40,44,60,64,68)])

    TARGET_LEN = dict(bikes = len(TARGET_VALS["bikes"]),
                      energy = len(TARGET_VALS["energy"]),
                      no2 = len(TARGET_VALS["no2"]))

    TARGET_PLAUS = dict(bikes = [0, 10000],
                        energy = [0,200],
                        no2 = [0,300])

    COLS_QUANTILES = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]

    print("Start checking...")
    print("---------------------------")
    col_names = df.columns


    print("Checking the Columns...")
    # Check column length
    if len(col_names) != LEN_EXP_COLS:
        print("Dataset contains ",len(col_names), "columns. Required are",LEN_EXP_COLS)
        print("Stopping early...")
        sys.exit()

    if set(col_names) != set(EXPECTED_COLS):
        print("Dataset does not contain the required columns (or more).")
        missing_cols = set(EXPECTED_COLS) - set(col_names)
        print("The missing columns are:", missing_cols)
        print("Stopping early...")
        sys.exit()

    for i,col in enumerate(EXPECTED_COLS):
        if col == col_names[i]:
            continue
        else:
            print("Columns not in correct order. Order should be:", EXPECTED_COLS)
            print("Your order is:", col_names.values)
            print("Stopping early...")
            sys.exit()

    # Date Col
    print("Checking type of columns...")
    try:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d",
                                            errors="raise")
    except (pd.errors.ParserError, ValueError):
        print("Could not parse Date in format YYYY-MM-DD")
        print("Stopping early...")
        sys.exit()

    try:
        df["target"] = df["target"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert target column to String.")
        print("Stopping early...")
        sys.exit()

    try:
        df["horizon"] = df["horizon"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert horizon column to String.")
        print("Stopping early...")
        sys.exit()

    for cq in COLS_QUANTILES:
        if pd.to_numeric(df[cq], errors="coerce").isna().any():
            print("----WARNING: Some elements in",cq,"column are not numeric. This may be fine if you only submit 2 out of 3 targets.")
            print("")
            # print("Stopping early...")
            # sys.exit()

    print("Checking if the Dates make sense...")

    if len(pd.unique(df["forecast_date"])) > 1:
        print("forecast_date needs to be the same in all rows.")
        print("Stopping early...")
        sys.exit()

    if df["forecast_date"][0].date() < datetime.today().date():
        print("----WARNING: Forecast date should not be in the past.")
        print("")
        # warnings.warn("Forecast date should not be in the past.")

    if df["forecast_date"][0].weekday() != 2:
        print("----WARNING: Forecast date should be a Wednesday.")
        print("")
        # warnings.warn("Forecast date should be a Wednesday")

    print("Checking targets...")

    if not df["target"].isin(TARGETS).all():
        print(f"Target column can only contain {TARGETS}. Check spelling.")
        print("Stopping early...")
        sys.exit()

    for target in TARGETS:

        if len(df[df["target"] == target]) != TARGET_LEN[target]:
            if target == "demand":
                print("Exactly 6 rows need to have target = ", target)
            else:
                print("Exactly 5 rows need to have target =", target)
            print("Stopping early...")
            sys.exit()

        if (df[df["target"] == target]["horizon"] != TARGET_VALS[target]).any():
            print("Target", target, "horizons need to be (in this order):", TARGET_VALS[target])
            print("Stopping early...")
            sys.exit()

        if (df[df["target"] == target][COLS_QUANTILES] < TARGET_PLAUS[target][0]).any(axis=None) or \
            (df[df["target"] == target][COLS_QUANTILES] > TARGET_PLAUS[target][1]).any(axis=None):
            print("----WARNING: Implausible values for",target,"detected. You may want to re-check.")
            print("")
            # warnings.warn("Implausible values for "+str(target)+" detected. You may want to re-check them.")

    print("Checking quantiles...")

    ALL_NAN_IDX = df[df.isna().any(axis=1)].index
    NAN_TARGET_IDX_LIST = []

    if len(ALL_NAN_IDX) != 0:
        NAN_TARGET = df.iloc[ALL_NAN_IDX[0]]["target"]
#        NAN_TARGET_LENS = dict(DAX = 5,
#                            energy = 6,
#                            infections = 5)

        NAN_TARGET_IDX_LIST = df[df["target"] == NAN_TARGET].index

        print("Assume that --",NAN_TARGET,"-- is your NaN-target. Please DOUBLECHECK if this is correct.")

#        if len(ALL_NAN_IDX) > NAN_TARGET_LENS[NAN_TARGET]:
        if len(ALL_NAN_IDX) > TARGET_LEN[NAN_TARGET]:
            print("Your dataframe contains more NaNs than entries for target",NAN_TARGET,".")
            print("Stopping early...")
            sys.exit()
    else:
        print("Seems like you submitted all three targets. Good job!")

    for i, row in df.iterrows():
        if i in NAN_TARGET_IDX_LIST:
            continue

        diffs = row[COLS_QUANTILES].diff()
        if diffs[1:].isna().any():
            print("Something is wrong with your quantiles.")
            print("Stopping early...")
            sys.exit()
        diffs[0] = 0
        if (diffs < 0).any():
            print("Predictive quantiles in row",i,"are not ordered correctly (need to be non-decreasing)")
            print("Stopping early...")
            sys.exit()

    print("---------------------------")
    print("Looks good!")


check_df(forecast)