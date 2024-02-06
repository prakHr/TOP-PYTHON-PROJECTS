import warnings

warnings.filterwarnings("ignore")
import visions
from visions import String, VisionsBaseType
from typing import Sequence

import numpy as np
import pandas as pd
import pandas.api.types as pdt

from visions.relations import IdentityRelation, TypeRelation
from visions.typesets.typeset import get_type_from_path

from visions.functional import cast_to_inferred, detect_type, infer_type
from visions.typesets import CompleteSet
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/gprak/anaconda3/envs/py38/Lib/site-packages/graphviz/'

def is_len_2(series):
    return (series.str.len() == 2).all() and not series.hasnans


def is_alpha2(series):
    iso_3166_alpha_iso_2_codes = [
        "AF",
        "AX",
        "AL",
        "DZ",
        "AS",
        "AD",
        "AO",
        "AI",
        "AQ",
        "AG",
        "AR",
        "AM",
        "AW",
        "AU",
        "AT",
        "AZ",
        "BS",
        "BH",
        "BD",
        "BB",
        "BY",
        "BE",
        "BZ",
        "BJ",
        "BM",
        "BT",
        "BO",
        "BQ",
        "BA",
        "BW",
        "BV",
        "BR",
        "IO",
        "BN",
        "BG",
        "BF",
        "BI",
        "CV",
        "KH",
        "CM",
        "CA",
        "KY",
        "CF",
        "TD",
        "CL",
        "CN",
        "CX",
        "CC",
        "CO",
        "KM",
        "CG",
        "CD",
        "CK",
        "CR",
        "CI",
        "HR",
        "CU",
        "CW",
        "CY",
        "CZ",
        "DK",
        "DJ",
        "DM",
        "DO",
        "EC",
        "EG",
        "SV",
        "GQ",
        "ER",
        "EE",
        "SZ",
        "ET",
        "FK",
        "FO",
        "FJ",
        "FI",
        "FR",
        "GF",
        "PF",
        "TF",
        "GA",
        "GM",
        "GE",
        "DE",
        "GH",
        "GI",
        "GR",
        "GL",
        "GD",
        "GP",
        "GU",
        "GT",
        "GG",
        "GN",
        "GW",
        "GY",
        "HT",
        "HM",
        "VA",
        "HN",
        "HK",
        "HU",
        "IS",
        "IN",
        "ID",
        "IR",
        "IQ",
        "IE",
        "IM",
        "IL",
        "IT",
        "JM",
        "JP",
        "JE",
        "JO",
        "KZ",
        "KE",
        "KI",
        "KP",
        "KR",
        "KW",
        "KG",
        "LA",
        "LV",
        "LB",
        "LS",
        "LR",
        "LY",
        "LI",
        "LT",
        "LU",
        "MO",
        "MG",
        "MW",
        "MY",
        "MV",
        "ML",
        "MT",
        "MH",
        "MQ",
        "MR",
        "MU",
        "YT",
        "MX",
        "FM",
        "MD",
        "MC",
        "MN",
        "ME",
        "MS",
        "MA",
        "MZ",
        "MM",
        "NA",
        "NR",
        "NP",
        "NL",
        "NC",
        "NZ",
        "NI",
        "NE",
        "NG",
        "NU",
        "NF",
        "MK",
        "MP",
        "NO",
        "OM",
        "PK",
        "PW",
        "PS",
        "PA",
        "PG",
        "PY",
        "PE",
        "PH",
        "PN",
        "PL",
        "PT",
        "PR",
        "QA",
        "RE",
        "RO",
        "RU",
        "RW",
        "BL",
        "SH",
        "KN",
        "LC",
        "MF",
        "PM",
        "VC",
        "WS",
        "SM",
        "ST",
        "SA",
        "SN",
        "RS",
        "SC",
        "SL",
        "SG",
        "SX",
        "SK",
        "SI",
        "SB",
        "SO",
        "ZA",
        "GS",
        "SS",
        "ES",
        "LK",
        "SD",
        "SR",
        "SJ",
        "SE",
        "CH",
        "SY",
        "TW",
        "TJ",
        "TZ",
        "TH",
        "TL",
        "TG",
        "TK",
        "TO",
        "TT",
        "TN",
        "TR",
        "TM",
        "TC",
        "TV",
        "UG",
        "UA",
        "AE",
        "GB",
        "US",
        "UM",
        "UY",
        "UZ",
        "VU",
        "VE",
        "VN",
        "VG",
        "VI",
        "WF",
        "EH",
        "YE",
        "ZM",
        "ZW",
    ]
    return series.isin(iso_3166_alpha_iso_2_codes).all()


class CountryCode(VisionsBaseType):
    @staticmethod
    def get_relations():
        return [IdentityRelation(String)]

    @classmethod
    def contains_op(cls, series, state):
        return is_len_2(series) and is_alpha2(series)



typeset = CompleteSet()
variable_set = typeset + CountryCode
variable_set.output_graph("variable_set.pdf")

def detectDtypes(arr):
    
    keys = [i for i in range(len(arr))]
    json_dict = {k: [v] for k, v in zip(keys, arr)}
    df = pd.DataFrame(json_dict)
    out_json_dict = {
        "input": arr,
        "detect_type(df,typeset)": list((detect_type(df, variable_set)).values()),
        "infer_type(df, typeset)": list((infer_type(df, variable_set)).values())
    }

    return out_json_dict


if __name__ == "__main__":
    arr = [
        1,
        2,
        3,
        np.nan,
        "1",
        "2",
        "3",
        "http://www.cwi.nl:80/%7Eguido/Python.html",
        "0b8a22ca-80ad-4df5-85ac-fa49c44b7ede",
        "AF"
    ]

    rv = detectDtypes(arr)
    df = pd.DataFrame(rv)
    print(df)
    from pprint import pprint
    pprint(rv)