#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 11/17/24, 11:28 PM
#     Last change by yifei
#    *****************************************************************************
from collections import OrderedDict


NATURAL_GAS_gri30 = OrderedDict(
    [
        ("methane", 0.9477),
        ("ethane", 0.042),
        ("propane", 0.002),
        ("nitrogen", 0.005),
        ("carbon dioxide", 0.003),
        ("oxygen", 0.0001),
        ("hydrogen", 0.0002),
    ]
)

NATURAL_GAS = OrderedDict(
    [
        ("methane", 0.947),
        ("ethane", 0.042),
        ("propane", 0.002),
        ("isobutane", 0.0002),
        ("butane", 0.0002),
        ("isopentane", 0.0001),
        ("pentane", 0.0001),
        ("hexane", 0.0001),
        ("nitrogen", 0.005),
        ("carbon dioxide", 0.003),
        ("oxygen", 0.0001),
        ("hydrogen", 0.0002),
    ]
)

HYDROGEN = OrderedDict([("hydrogen", 1)])

RUSSIAN_NATURAL_GAS = {
    "methane": 0.9696,
    "nitrogen": 0.0086,
    "carbon dioxide": 0.0018,
    "ethane": 0.0137,
    "propane": 0.0045,
    "butane": 0.0015,
    "pentane": 0.0002,
    "hexane": 0.0001,
    "oxygen": 0.0,
}

NORTH_SEA_NATURAL_GAS = {
    "methane": 0.8872,
    "nitrogen": 0.0082,
    "carbon dioxide": 0.0194,
    "ethane": 0.0693,
    "propane": 0.0125,
    "butane": 0.0028,
    "pentane": 0.0005,
    "hexane": 0.0001,
    "oxygen": 0.0,
}

DANISH_NATURAL_GAS = {
    "methane": 0.9007,
    "nitrogen": 0.0028,
    "carbon dioxide": 0.006,
    "ethane": 0.0568,
    "propane": 0.0219,
    "butane": 0.009,
    "pentane": 0.0022,
    "hexane": 0.0006,
    "oxygen": 0.0,
}

DUTCH_NATURAL_GAS = {
    "methane": 0.8364,
    "nitrogen": 0.1021,
    "carbon dioxide": 0.0168,
    "ethane": 0.0356,
    "propane": 0.0061,
    "butane": 0.0019,
    "pentane": 0.0004,
    "hexane": 0.0007,
    "oxygen": 0.0,
}

GERMAN_NATURAL_GAS = {
    "methane": 0.8646,
    "nitrogen": 0.1024,
    "carbon dioxide": 0.0208,
    "ethane": 0.0106,
    "propane": 0.0011,
    "butane": 0.0003,
    "pentane": 0.0001,
    "hexane": 0.0001,
    "oxygen": 0.0,
}

BIO_METHANE = {
    "methane": 0.9615,
    "nitrogen": 0.0075,
    "carbon dioxide": 0.029,
    "ethane": 0.0,
    "propane": 0.0,
    "butane": 0.0,
    "pentane": 0.0,
    "hexane": 0.0,
    "oxygen": 0.002,
}

BIO_METHANE_LPG_MIX = {
    "methane": 0.9094,
    "nitrogen": 0.0069,
    "carbon dioxide": 0.0268,
    "ethane": 0.0,
    "propane": 0.05,
    "butane": 0.005,
    "pentane": 0.0,
    "hexane": 0.0,
    "oxygen": 0.0019,
}

# Italian gas (SNAM) compositions — source: SNAM gas quality data (mole fractions)
# Volume reference: 15°C / 1.01325 bar (Sm³) per UNI EN ISO 13443
# Combustion reference: 25°C per UNI EN ISO 6976
# Mazara del Vallo (Algerian gas via Tunisia) — primary Sicilian supply point
SNAM_MAZARA_GAS = {
    "methane": 0.86909,
    "ethane": 0.08661,
    "propane": 0.01163,
    "isobutane": 0.00072,
    "butane": 0.00112,
    "isopentane": 0.00019,
    "pentane": 0.00019,
    "hexane": 0.00014,       # includes C6+ (esani+)
    "nitrogen": 0.01334,
    "carbon dioxide": 0.01672,
    "helium": 0.00025,
}

# Gela (Libyan gas, Greenstream pipeline) — secondary Sicilian supply point
SNAM_GELA_GAS = {
    "methane": 0.86645,
    "ethane": 0.05246,
    "propane": 0.02022,
    "isobutane": 0.00317,
    "butane": 0.00478,
    "isopentane": 0.00071,
    "pentane": 0.00026,
    "hexane": 0.00060,       # includes C6+ (esani+)
    "nitrogen": 0.04455,
    "carbon dioxide": 0.00602,
    "helium": 0.00078,
}


# COMMON_GAS_COMPOSITIONS: A dictionary of common gas mixture compositions
COMMON_GAS_COMPOSITIONS = {
    "NATURAL_GAS_gri30": NATURAL_GAS_gri30,
    "NATURAL_GAS": NATURAL_GAS,
    "HYDROGEN": HYDROGEN,
    "RUSSIAN_NATURAL_GAS": RUSSIAN_NATURAL_GAS,
    "NORTH_SEA_NATURAL_GAS": NORTH_SEA_NATURAL_GAS,
    "DANISH_NATURAL_GAS": DANISH_NATURAL_GAS,
    "DUTCH_NATURAL_GAS": DUTCH_NATURAL_GAS,
    "GERMAN_NATURAL_GAS": GERMAN_NATURAL_GAS,
    "BIO_METHANE": BIO_METHANE,
    "BIO_METHANE_LPG_MIX": BIO_METHANE_LPG_MIX,
    "SNAM_MAZARA_GAS": SNAM_MAZARA_GAS,
    "SNAM_GELA_GAS": SNAM_GELA_GAS,
}
