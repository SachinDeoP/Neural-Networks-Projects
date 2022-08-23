import os
import re
import pandas as pd

#Loading data
os.chdir("C:\\Users\\703294213\\Documents\\Work\\Names")
# data = pd.read_csv("Indian-Female-Names.csv")
data = pd.read_excel('India_Baby_Names.xls', sheet_name=0)

#Getting names in a list
name_list_1 = data['Name_1'].to_list()
name_list_2 = data['Name_2'].to_list()
name_list = name_list_1 + name_list_2

#Removing any numerical part from names
name_list = [x for x in name_list if not isinstance(x, int)]
name_list = [x for x in name_list if not isinstance(x, float)]

#Getting only string names
only_alpha = []
for string in range(len(name_list)):
    if name_list[string].isalpha():
        only_alpha.append(name_list[string])
    else:
        pass

#Using regular expression to select names
r = re.compile(".*[s].*[p].*")
selected_list = list(filter(r.match, only_alpha))


print("Done")