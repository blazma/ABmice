import pyan
from IPython.display import HTML

file_path = r"C:\Users\martin\home\phd\repos\ABmice\BTSP\Statistics_Robustness.py"
output_root = r"C:\Users\martin\home\phd\repos\ABmice\BTSP\misc"

output = HTML(pyan.create_callgraph(filenames=file_path, format="html")).data
with open(f"{output_root}/graph.html", "w") as f:
    f.write(output)
