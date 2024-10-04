import os

area = "CA1"
base_folder = rf"D:\{area}"

confirm = input(f"Are you sure you want to delete shuffle data in {base_folder}? [Y/N]\t")
if confirm != "Y":
    print("aborting...")
    exit()

animals = {
    "CA1": ["KS028",
            "KS029",
            "KS030",
            "srb131",
            "srb231",
            "srb251",
            "srb402"],
    "CA3": ["srb231",
            "srb251",
            "srb269",
            "srb270",
            "srb363",
            "srb377",
            "srb402"]
}

for animal in animals[area]:
    imaging_folder = f"{base_folder}/{animal}_imaging"
    _, subfolders, _ = next(os.walk(imaging_folder))
    sessions = [folder for folder in subfolders if animal in folder]

    for session in sessions:
        analysed_data_dir = f"{imaging_folder}/{session}/analysed_data"
        if os.path.isdir(analysed_data_dir):
            _, _, files = next(os.walk(analysed_data_dir))
            shuffle_files = [f"{analysed_data_dir}/{file}" for file in files if "shuffle" in file or "p95" in file]
            for file in shuffle_files:
                print(f"deleting {file}...")
                os.remove(file)
