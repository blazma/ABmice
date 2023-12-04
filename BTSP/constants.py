from collections import namedtuple

ANIMALS = {
    "CA1": ["KS028",
            "KS029",
            "KS030",
            "srb131"],
    "CA3": ["srb231",
            "srb251",
            "srb269",
            "srb270"]
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
    "CA1": [
        "KS030_103021"
    ],
    "CA3": [
        "srb231_220809_004",
        "srb231_220812_001",
        "srb251_221027_T1",
        "srb251_221027_T2"
    ]
}
CORRIDORS = [14, 15,  # random
             16, 18,  # block
             17]  # new environment

Category = namedtuple('Category', ["order", "color"])
CATEGORIES = {
    "unreliable": Category(0, "#a2a1a1"),
    "early": Category(1, "#00e3ff"),
    "transient": Category(2, "#ffe000"),
    "non-btsp": Category(3, "#ff000f"),
    "btsp": Category(4, "#be70ff"),
}