import pandas as pd
import os
# df = pd.read_csv('sixray/attention_label/results2_img_arm_factual.csv')
# print(df.head(30))


filenames = next(os.walk('./sixray/train/neg/'), (None, None, []))[2]
for filename in filenames:
    print(filename)