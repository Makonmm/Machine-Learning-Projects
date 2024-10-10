import os
import os.path
import argparse
import sys
import traceback
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="XPG Boost to Regression in Python")
    parser.add_argument("--model", type=str, metavar="arg",
                        required=True, help="Pre trained XPG Boost model")
    parser.add_argument("--data", type=str, metavar="arg",
                        required=True, help="CSV data")
    parser.add_argument("--result", type=str, metavar="arg",
                        required=True, help="Local to save the result")
    args = parser.parse_args()

    return args


def main():

    try:

        args = parse_args()

        df = pd.read_csv(args.data, dtype=np.float32)

        regressor = xgb.XGBRegressor()
        regressor.load_model(args.model)

        preds = regressor.predict(df)

        result = pd.DataFrame({"predictions: ": preds})

        result.to_csv(args.result, float_format="%.3f",
                      index=False, header=False)

    except Exception as e:
        print(e, file=sys.stderr)
        print("====", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(-1)


if __name__ == "__main__":
    main()
    print("Model executed!")
