import pandas as pd
import os

columns = pd.read_csv('data/individual_level/raw_csv_700/April 2015 to Dec 2015.csv').columns
data = pd.DataFrame(columns=columns)

directory = 'data/individual_level/raw_csv_700'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    curr = pd.read_csv(f)
    for index, row in curr.iterrows():
        r = []
        for col in columns:
            if col == "Stat Date":
                s = row[col][6:10] + "-" + row[col][0:2] + "-" + row[col][3:5]
                r.append(s)
            else:
                r.append(row[col])
        data.loc[-1] = r
        data.index += 1
data.to_csv('out.csv', index=False) 
print("done!")