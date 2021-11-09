import argparse
import datetime as dt
import os
import pandas as pd
import changepoint.changepoint_detection as cpd
from moment_strategy.base_strat import BaseStrat

from settings.default import CPD_DEFAULT_LBW, USE_KM_HYP_TO_INITIALISE_KC


def main(
    input_file_path: str, output_score_path: str, output_cpd_path: str,
    start_date: dt.datetime, end_date: dt.datetime, lookback_window_length :int
):
    data = pd.read_csv(input_file_path, index_col = 0, parse_dates = True)
    ticker = os.path.basename(input_file_path)
    ticker = os.path.splitext(ticker)[0]
    data["daily_returns"] = BaseStrat.calc_returns(data["price"])
    data = data.iloc[1:]

    cpd.run_module(
        data, lookback_window_length, output_score_path, output_cpd_path, 
        start_date, end_date, USE_KM_HYP_TO_INITIALISE_KC
    )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "input_file_path",
            metavar="t",
            type=str,
            nargs="?",
            default="^FTSE",
            # choices=[],
            help="Input file location for csv.",
        )
        parser.add_argument(
            "output_score_path",
            metavar="f",
            type=str,
            nargs="?",
            default="data/cpd_score.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "output_cpd_path",
            metavar="f",
            type=str,
            nargs="?",
            default="data/cpd_detect.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "start_date",
            metavar="s",
            type=str,
            nargs="?",
            default="2005-01-01",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "end_date",
            metavar="e",
            type=str,
            nargs="?",
            default="2009-12-31",
            help="End date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )

        args = parser.parse_known_args()[0]

        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")

        return (
            args.input_file_path,
            args.output_score_path,
            args.output_cpd_path,
            start_date,
            end_date,
            args.lookback_window_length
        )

    main(*get_args())
