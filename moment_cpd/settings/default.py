import os

CPD_LBWS = [10, 21, 63, 126, 256]
CPD_DEFAULT_LBW = 63
USE_KM_HYP_TO_INITIALISE_KC = True
CPD_INPUT_FOLER_DEFAULT = 'data/prod_data'
CPD_OUTPUT_FOLDER_DEFAULT = f"data/cpd_data/stocks_cpd_{CPD_DEFAULT_LBW}lbw"

START_DATE = '2009-01-01'
END_DATE = '2021-12-31'
CPD_THRESHOLD = 0.999

FEATURES_FILE_PATH_DEFAULT = f"data/feature_data/stocks_cpd_{CPD_DEFAULT_LBW}lbw.csv"

# commodities from Yahoo Finance
COMMODITIES_TICKERS = [
    "CC=F",
    "CL=F",
    "CT=F",
    "ES=F",
    "GC=F",
    "GF=F",
    "HE=F",
    "HG=F",
    "HO=F",
    "KC=F",
    "KE=F",
    "LBS=F",
    "LE=F",
    "MGC=F",
    "NG=F",
    "NQ=F",
    "OJ=F",
    "PA=F",
    "PL=F",
    "RB=F",
    "RTY=F",
    "SB=F",
    "SI=F",
    "SIL=F",
    "YM=F",
    "ZB=F",
    "ZC=F",
    "ZF=F",
    "ZL=F",
    "ZM=F",
    "ZN=F",
    "ZO=F",
    "ZR=F",
    "ZS=F",
    "ZT=F",
]