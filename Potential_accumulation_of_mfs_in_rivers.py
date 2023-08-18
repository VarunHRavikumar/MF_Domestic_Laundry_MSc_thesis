'''
*** Please Note: ***
'''
'''
The code uses the shapefile converted into csv data 
to minimise the run time and compile the code efficiently.
Once the output is saved as as csv this file is read in and 
merged with the HydroRIVER Europe and Middle East shapefile.
'''
'''
The complete HydroRIVER Europe and Middle East shapefile
is required to run this code. 
Once the shapefile is clipped to European boundaries, the code will start showing errors.
This occurs as a result of the NEXT_DOWN information for a river being 
removed during the clipping operation, thereby leaving missing data.
'''


import pandas as pd
import geopandas as gpd
import numpy as np

# Class for the node of the tree
class Node:
    def __init__(self, data, hv_id):
        self.data = data
        self.hv_id = hv_id
        self.children = []


# Function to build an N-ary tree from the dataset
def buildNaryTree1(data):
    node_dict = {}  # Dictionary to store nodes by HYRIV_ID
    root = None

    for index, row in data.iterrows():
        hyriv_id = row['HYRIV_ID']
        effl_mf = row['EFFL_MF']

        node = Node(effl_mf, hyriv_id)
        node_dict[hyriv_id] = node

       

    for index, row in data.iterrows():
         hyriv_id = row['HYRIV_ID']
         next_down = row['NEXT_DOWN']
      
         node = node_dict[hyriv_id]

         if next_down == 0:
             root = node
         else:
             parent_node = node_dict[next_down]
             parent_node.children.append(node)
             
    return root

# Function to perform sum replacement on the N-ary tree
def sumReplacementNary(node):
    if node is None:
        return 0

    total = len(node.children)
    subtree_sum = 0

    for i in range(total):
        subtree_sum += sumReplacementNary(node.children[i])
    
    node.data += subtree_sum
    new_df.at[node.hv_id, 'Accumulation'] = node.data
    return node.data

# Function to perform preorder traversal on the N-ary tree
def preorderTraversal(node):
    if node is None:
        return

    total = len(node.children)
    for i in range(total):
        preorderTraversal(node.children[i])

# Read data from CSV file
csv_path = 'Path_to_csv ' # # <--- replace this with the HydroRIVER Europe and Middle East csv 
df = pd.read_csv(csv_path)
new_df = pd.DataFrame(columns = ['Accumulation'], index = df['HYRIV_ID'])
roots = df['MAIN_RIV'].unique()
print(roots.size)
for root in roots:
  print(root)
  
  river_segments = df.loc[df['MAIN_RIV'] == root]
  # river_segments =  river_segments.sort_values('NEXT_DOWN')
  root_node = buildNaryTree1(river_segments)
  sumReplacementNary(root_node)
  print('Done')

#exporting data as CSV
new_df.to_csv('Path_to_output_csv') # <--- specify the location where you want to save the output CSV

#reading the river shapefile
river_data = gpd.read_file('Path_to_shapefile') # <--- replace this with the HydroRIVER Europe and Middle East shape file
river_data = river_data.merge(new_df, left_on= 'HYRIV_ID', right_on= 'HYRIV_ID', how= 'outer')
river_data.to_file('Path_to_output_shapefile') # <--- specify the location where you want to save the output shapefile
