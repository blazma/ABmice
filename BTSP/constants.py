from collections import namedtuple

ANIMALS = {
    "CA1": ["KS028",
            "KS029",
            "KS030",
            "srb131"],
    "CA3": ["srb231",
            "srb251",
            "srb269",
            "srb270",
            "srb363",
            "srb377"]
}
#ANIMALS_PALETTE = ["#796BF0", "#6EB50B", "#FF8E0F", "#FF6096"]
#ANIMALS_PALETTE_CA3 = ["#73B7FF", "#E7FF0D", "#D57AFF", "#FFB0BA"]
AREA_PALETTE = {
    "CA1": "#C0BAFF",
    "CA3": "#FFB0BA"
}
ANIMALS_PALETTE = {
    "CA1": ["#00DB55", "#00C7DB", "#274FDB", "#B25ADB"],
    "CA3": ["#D435FD", "#FF9733", "#E7FF33", "#FD3566", "#FFCD33", "#88FF4D"]
}

SESSIONS_TO_IGNORE = {
    "CA1": ['KS029_110321',   # no p95
            'KS029_110721',   # error
            'KS029_110821',   # error
            'KS029_110521',   # error
            'KS030_110721',   # error
            'srb131_211019'], # reshuffles
    "CA3": []
}

DISENGAGEMENT = {
    # FORMAT: "sessionID": [DE_startLap, DE_endLap]
    # IMPORTANT: disengagement cannot be "in the middle" of the session (i.e w/ good laps before and after too)
    "CA1": {
        "KS030_103021": [25, -1],  # -1 indicates last lap
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


