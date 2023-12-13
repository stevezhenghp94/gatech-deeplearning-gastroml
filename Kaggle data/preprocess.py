import pandas as pd

#####  Get cleaned ingredients (no unit amount or other descriptors)  #####
df = pd.DataFrame(
    pd.read_csv(
        "Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        sep=",",
        header=0,
        index_col=False,
    )
)
df.columns = [
    "id",
    "title",
    "ingredients",
    "instructions",
    "image_name",
    "cleaned_ingredients",
]
## filter out blank instructions
df_filtered = df[~df["instructions"].isnull()].reset_index(drop=True)

import nltk
nltk.download('punkt')
import re

stop_words = [
    "tsp",
    "whole",
    "lb",
    "tbsp",
    "cup",
    "ounce",
    "oz",
    "ground",
    "room temperature",
    "unsalted",
    "salted",
    "melted",
    "about",
    "medium",
    "extra-virgin olive",
    "small",
    "large",
    "more",
    "red",
    "black",
    "white",
    "yellow",
    "green",
    "orange",
    "purple",
    "blue",
    "pink",
    "brown",
    "gray",
    "light",
    "dark",
    "thin",
    "thick",
    "into",
    "dry",
    "flakes",
    "evaporated",
    "powder",
    "baking",
    "shallow",
    "deep" "dish",
    "inch",
    "pieces",
    "heavy",
    "twist",
    "hot",
    "and",
    "dish",
    "bags",
    "sweetness",
    "garnish",
    "taste",
    "pat",
    "fresh",
    "wheel",
    "dehydrated",
    "for",
    "chopped",
    "cut",
    "can",
    "teaspoon",
    "wedges",
    "tablespoon",
    "with",
    "tablespoon",
    "sprigs",
    "slices",
]

full = []
for i in range(len(df_filtered)):
    print(i)
    s1 = df_filtered["cleaned_ingredients"][i]
    s2 = df_filtered["instructions"][i]

    ## take full ingredients list, tokenize, and try to find matching words from instructions

    # preprocess
    s1 = s1.lower()
    s1 = re.sub(r"[^\w\s]", "", s1)  # remove punctuation
    s1 = re.sub(r"[0-9]+", "", s1)  # remove numbers
    for s in stop_words:  # remove stop words
        s1 = s1.replace(s, "")

    ingr_list_cleaned = []
    tokens = nltk.word_tokenize(s1)
    for token in tokens:
        if token in s2 and len(token) > 2:
            ingr_list_cleaned.append(token)

    ingr_list_cleaned = list(set(ingr_list_cleaned))
    full.append(ingr_list_cleaned)

df_filtered["Ingredients_RawMats"] = full
df_filtered.to_csv(
    "Food Ingredients and Recipe Dataset with Image Name Mapping_CLEANED.csv",
    index=False,
)

#####  Convert to JSON  #####
# From CLEANED to final_dat, the following changes were made in Excel:
# 1. adding a 'partition' column. Partitioned 60/20/20 for train/val/test
# 2. removing rows where 'Ingredients_RawMats' is an empty list
csv_file = pd.DataFrame(
    pd.read_csv("final_data.csv", sep=",", header=0, index_col=False)
)

csv_file.columns = [
    "id",
    "title",
    "ingredients",
    "instructions",
    "image_name",
    "cleaned_ingredients",
    "ingredients_rawmats",
    "partition",
]

# filter out rows with no images
csv_file_filtered = csv_file[csv_file["image_name"] != "#NAME?"].reset_index(drop=True)

csv_file_filtered.to_json(
    "final_data.json",
    orient="records",
    date_format="epoch",
    double_precision=10,
    force_ascii=True,
    date_unit="ms",
    default_handler=None,
)

##### Save images based on partition ####
import shutil

for index, row in csv_file_filtered.iterrows():
    img_name = row["image_name"] + ".jpg"
    shutil.copy(
        "Food Images/Food Images/" + img_name,
        "images/" + row["partition"] + "/" + img_name,
    )
