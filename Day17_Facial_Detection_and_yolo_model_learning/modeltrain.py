# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("sbaghbidi/human-faces-object-detection")

# print("Path to dataset files:", path)


#CSV check
import pandas as pd, pathlib
p = pathlib.Path("kagglehub/datasets/sbaghbidi/human-faces-object-detection/versions/1/faces.csv")
print("CSV exists?", p.exists())
df = pd.read_csv(p)
print("Columns:", list(df.columns))
print(df.head())
