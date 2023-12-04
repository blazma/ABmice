import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


df = pd.DataFrame.from_dict(
        {'H1': {0: 0.55, 1: 0.56, 2: 0.46, 3: 0.93, 4: 0.74, 5: 0.35, 6: 0.75, 7: 0.86, 8: 0.81, 9: 0.88},
         'H2': {0: 0.5, 1: 0.55, 2: 0.61, 3: 0.82, 4: 0.51, 5: 0.35, 6: 0.58, 7: 0.66, 8: 0.93, 9: 0.86},
         'H3': {0: 0.42, 1: 0.51, 2: 0.86, 3: 0.59, 4: 0.46, 5: 0.71, 6: 0.58, 7: 0.72, 8: 0.53, 9: 0.92},
         'H4': {0: 0.89, 1: 0.87, 2: 0.04, 3: 0.64, 4: 0.44, 5: 0.05, 6: 0.33, 7: 0.93, 8: 0.08, 9: 0.9},
         'H5': {0: 0.92, 1: 0.75, 2: 0.13, 3: 0.85, 4: 0.51, 5: 0.15, 6: 0.38, 7: 0.92, 8: 0.36, 9: 0.76},
         'chirality': {0: 'Left', 1: 'Left', 2: 'Left', 3: 'Left', 4: 'Left', 5: 'Right', 6: 'Right', 7: 'Right', 8: 'Right', 9: 'Right'},
         'image': {0: 'image_0', 1: 'image_1', 2: 'image_2', 3: 'image_3', 4: 'image_4', 5: 'image_0', 6: 'image_1', 7: 'image_2', 8: 'image_3', 9: 'image_4'}})

df_long = df.melt(id_vars=['chirality', 'image'], value_vars=['H1', 'H2', 'H3', 'H4', 'H5'],
                  var_name='H', value_name='value')

fig, ax = plt.subplots(figsize=(15, 6))
sns.violinplot(ax=ax,
               data=df_long,
               x='H',
               y='value',
               hue='chirality',
               palette='summer',
               split=True)
plt.show()
