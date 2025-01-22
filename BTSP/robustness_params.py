PARAMS = {
    "equalized to min": {  # equalized to smallest number of object
        "AxS": {
            "single area": {
                "max sessions CA1": 3,
                "max sessions CA3": 5
            },
            "both areas": {
                "max sessions CA1": 3,  # 3
                "max sessions CA3": 6  # 5
            }
        },
        "AxC": {
            "single area": {
                "max cells CA1": 112,  # number of cells for session with the lowest number of reliable cells
                "max cells CA3": 10,
                "runs": 1000
            },
            "both areas": {
                "max cells CA1": 112,  # number of cells for session with the lowest number of reliable cells
                "max cells CA3": 10,
                "runs": 100
            }
        },
        "AxPFs": {
            "single area": {
                "max pfs CA1": 131,  # number of rPFs for session with the lowest number of reliable PFs
                "max pfs CA3": 11,  # -||- except we ignore srb269 cuz it has 3 rPFs in total
                "runs": 1000
            },
            "both areas": {
                "max pfs CA1": 131,  # number of rPFs for session with the lowest number of reliable PFs
                "max pfs CA3": 11,  # -||- except we ignore srb269 cuz it has 3 rPFs in total
                "runs": 100
            }
        },
        "SxC": {
            "single area": {
                "max cells CA1": 55,
                "max cells CA3": 5,
                "runs": 1000
            },
            "both areas": {
                "max cells CA1": 59,
                "max cells CA3": 5,
                "runs": 100
            }
        },
        "SxPFs": {
            "single area": {
                "max pfs CA1": 59,
                "max pfs CA3": 5,
                "runs": 1000
            },
            "both areas": {
                "max pfs CA1": 59,
                "max pfs CA3": 9,
                "runs": 100
            }
        }
    },

    ###################################################################################################################
    "equalized to avg": {  # equalized to smallest number of object
        "AxS": {
            "single area": {
                "max sessions CA1": 3,
                "max sessions CA3": 5
            },
            "both areas": {
                "max sessions CA1": 3,  # 3
                "max sessions CA3": 6  # 5
            }
        },
        "AxC": {
            "single area": {
                "max cells CA1": 647,
                "max cells CA3": 39,
                "runs": 1000
            },
            "both areas": {
                "max cells CA1": 647,  # number of cells for session with the lowest number of reliable cells
                "max cells CA3": 39,
                "runs": 100
            }
        },
        "AxPFs": {
            "single area": {
                "max pfs CA1": 1561,  # number of rPFs for session with the lowest number of reliable PFs
                "max pfs CA3": 67,  # -||- except we ignore srb269 cuz it has 3 rPFs in total
                "runs": 1000
            },
            "both areas": {
                "max pfs CA1": 1561,  # number of rPFs for session with the lowest number of reliable PFs
                "max pfs CA3": 67,  # -||- except we ignore srb269 cuz it has 3 rPFs in total
                "runs": 100
            }
        },
        "SxC": {
            "single area": {
                "max cells CA1": 365,
                "max cells CA3": 11,
                "runs": 1000
            },
            "both areas": {
                "max cells CA1": 365,
                "max cells CA3": 11,
                "runs": 100
            }
        },
        "SxPFs": {
            "single area": {
                "max pfs CA1": 575,
                "max pfs CA3": 12,
                "runs": 1000
            },
            "both areas": {
                "max pfs CA1": 575,
                "max pfs CA3": 12,
                "runs": 100
            }
        }
    }
}