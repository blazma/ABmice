import csv

path_version_before = "BTSP_CA1_WrongEarly/placefield_categories.csv"
path_version_after = "BTSP_CA1_CorrectEarly/placefield_categories.csv"
output_path = "BTSP_CA1_CorrectEarly/placefield_categories_change.csv"


def read_placefield_categories_file(path):
    placefield_categories = []
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            placefield_categories.append(row)
    return placefield_categories

placefields_before = read_placefield_categories_file(path_version_before)
placefields_after = read_placefield_categories_file(path_version_after)

placefields_only_in_before = [pf for pf in placefields_before if pf not in placefields_after]
placefields_only_in_after = [pf for pf in placefields_after if pf not in placefields_before]
placefields_w_category_change = []
for pf_a in placefields_only_in_after:
    for pf_b in placefields_only_in_before:
        is_same_place_field_w_category_change = pf_a["session"] == pf_b["session"] and \
                                                pf_a["cellid"] == pf_b["cellid"] and \
                                                pf_a["corridor"] == pf_b["corridor"] and \
                                                pf_a["first bin"] == pf_b["first bin"] and \
                                                pf_a["last bin"] == pf_b["last bin"] and \
                                                pf_a["category"] != pf_b["category"]
        if is_same_place_field_w_category_change:
            category_change = f"{pf_b['category']} -> {pf_a['category']}"
            pf_a_vals = list(pf_a.values())[1:]  # [1:] -> skip category bc i only care about change
            placefields_w_category_change.append([category_change, *pf_a_vals])

with open(output_path, "w") as file:
    writer = csv.writer(file)
    writer.writerow(["BEFORE path:", path_version_before,"\n"])
    writer.writerow(["AFTER path:", path_version_after,"\n"])
    writer.writerow(["\n"])

    # category changes
    writer.writerow(["PFs with category change", "category change", *list(placefields_before[0].keys())[1:], "\n"])
    for pf in placefields_w_category_change:
        writer.writerow(["", *pf, "\n"])
    writer.writerow(["\n"])

    # only in before
    writer.writerow([f"PFs only in BEFORE", *list(placefields_before[0].keys()), "\n"])
    for pf in placefields_only_in_before:
        writer.writerow(["", *list(pf.values()), "\n"])
    writer.writerow(["\n"])

    # only in after
    writer.writerow([f"PFs only in AFTER", *list(placefields_after[0].keys()), "\n"])
    for pf in placefields_only_in_after:
        writer.writerow(["", *list(pf.values()), "\n"])
