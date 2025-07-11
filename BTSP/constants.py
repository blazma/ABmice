from collections import namedtuple

ANIMALS = {
    "CA1": ["KS028",
            "KS029",
            "KS030",
            "srb128",
            "srb131",
            "srb231",
            "srb251",
            "srb402",
            "srb410",
            "srb504",
            "srb504a",
            "srb517"],
    "CA3": ["srb231",
            "srb251",
            "srb269",
            "srb270",
            "srb363",
            "srb377",
            "srb402",
            "srb410",
            "srb529"]
}

AREA_PALETTE = {
    "CA1": "#C0BAFF",
    "CA3": "#FFB0BA"
}
ANIMALS_PALETTE = {
    #"CA1": ["#FF00FF", "#005ACF", "#E6AFFF", "#A0FAFF", "#4C99FF", "#3BFFB0", "#BB8DFF", "#00C77B", "#414BFF", "#A6FF9E", "#FF6EB4"],
    "CA1": ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990"],
    #"CA3": ["#ff3700", "#ff005f", "#ee00a8", "#a046e3", "#006cfe", "#007bf2", "#74b2df", "#d0f5ff", "#567b97"]
    "CA3": ["#911EB4", "#42D4F4", "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F032E6", "#BFEF45"]
}

SESSIONS_TO_IGNORE = {
    "CA1": ["KS029_110321",  # not part of old meta
            "KS029_110521",  # too few cells
            "KS028_110621",  # post
            "KS028_110621",  # post
            "KS028_110721",  # post
            "KS028_110821",  # not part of old meta
            "KS029_111321",  # post
            "KS029_111421",  # post
            "KS029_111521",  # post
            "KS030_110521",  # post
            "KS030_110621",  # post
            "KS030_110721",  # post
            "srb131_211021"],  # post
    "CA3": ["srb410a_240607"]  # error
}
"""
SESSIONS_TO_IGNORE = {
    "CA1": ['KS029_110321',   # no p95
            #'KS029_110721',   # error
            #'KS029_110821',   # error
            #'KS029_110521',   # error
            'KS030_110721',   # error
            'srb131_211019'], # reshuffles
    "CA3": []
}
"""

DISENGAGEMENT = {
    # FORMAT: "sessionID": [DE_startLap, DE_endLap]
    # IMPORTANT: disengagement cannot be "in the middle" of the session (i.e w/ good laps before and after too)
    "CA1": {
        "KS030_103021": [25, -1],  # -1 indicates last lap
        "srb402_240319_CA1": [0, 6],
        "srb410a_240529_CA1": [48, -1],
    },
    "CA3": {
        "srb231_220809_004": [25, -1],
        "srb231_220812_001": [30, -1],
        "srb251_221027_T1": [0, -1],  # disengaged throughout -- skipping
        "srb251_221027_T2": [0, -1],  # disengaged throughout -- skipping
        "srb269_230125": [55, -1],
    }
}

CORRIDORS = [14, 15,  # random
             16, 18,  # block
             17]  # new environment

Category = namedtuple('Category', ["order", "color"])
CATEGORIES = {
    "unreliable": Category(0, "#a2a1a1"),
    "early": Category(1, "#00B0F0"),
    "transient": Category(2, "#77D600"),
    "non-btsp": Category(3, "#FF0000"),
    "btsp": Category(4, "#AA5EFF"),
}
CATEGORIES_DARK = {
    "unreliable": Category(0, "#2E2E2E"),
    "early": Category(1, "#001D91"),
    "transient": Category(2, "#056600"),
    "non-btsp": Category(3, "#702000"),
    "btsp": Category(4, "#5E276B"),
}

VSEL_NORMALIZATION = -0.45  # normalization constant for speed selectivities which generally range 0 to about 0.45
BEHAVIOR_SCORE_THRESHOLD = 4  # behavior score at which we cut off sessions (those below are discarded)
