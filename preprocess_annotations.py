import pandas
import numpy as np
import matplotlib.pyplot as plt
import json

ann = pandas.read_csv('places/attention_label/results_img_place_chunk_1_factual_20220112_005336_44.csv')

print(ann.iloc[1])



a = ann['attention'].iloc[1].replace('[', '').replace(']','').split(',')
a = [int(i) for i in a]
a = np.asarray(a).reshape(224,224)

a = np.array(json.loads(ann['attention'].iloc[1]))

plt.imshow(a)
plt.savefig('att.png')
