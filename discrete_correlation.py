import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.svm import SVR
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestRegressor 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from statistics import mode
import statistics
import pickle
import sys
import random
from collections import Counter
import scipy.stats as ss
import copy

import math
import csv
from tqdm import tqdm

from collections import Counter
from itertools import repeat, chain

from functools import cmp_to_key

from pybloomfilter import BloomFilter


import seaborn, time
seaborn.set_style('whitegrid')

# from sklearn.linear_model import LinearRegression

# from pomegranate import BayesianNetwork

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# def return_size(start,end,matrix,cola

class discrete_correl:
  exception_list_0=[]
  exception_list_1=[]
  exception_list_not_one=[]
  factor_0_to_1=0

def encode_discrete(matrix,df,i,j):


  a=np.array(matrix)
  temp_matrix=unique_rows(a)

  print("Sanity Check unique rows before",len(temp_matrix))

  col_name=list(df.columns)
  correl_data_struct=discrete_correl()

  print("\n")  
  print("columns",col_name[i],col_name[j])

  prim_sec_map={}
  sec_prim_map={}

  for t in range(0,len(matrix)):
    prim_sec_map[matrix[t][i]]=set([])
    sec_prim_map[matrix[t][j]]=set([])

  for t in range(0,len(matrix)):
    prim_sec_map[matrix[t][i]].add(matrix[t][j])
    sec_prim_map[matrix[t][j]].add(matrix[t][i])

  factor=0
  factor_list=[]
  temp_exception_list_0=set([])
  temp_exception_list_not_one=set([])
  
  for t in range(0,len(matrix)):
    if not(len(prim_sec_map[matrix[t][i]])==1 and len(sec_prim_map[matrix[t][j]])==1):
      temp_exception_list_not_one.add(matrix[t][i])

    if len(prim_sec_map[matrix[t][i]])==1:
      temp=0
      val=next(iter(prim_sec_map[matrix[t][i]]))
      factor=max(len(sec_prim_map[val]),factor) 
      factor_list.append(len(sec_prim_map[val]))
    else:
      temp_exception_list_0.add(matrix[t][i])  

  factor=100    
  correl_data_struct.factor_0_to_1=factor


  encoding_map={}    
  count_map={}
  count_exception=0
  max_col_1=0

  for t in range(0,len(matrix)):
    max_col_1=max(max_col_1,matrix[t][j])

  for t in range(0,len(matrix)):

    if matrix[t][i] in encoding_map:
      continue

    if matrix[t][i] in temp_exception_list_0:
      encoding_map[matrix[t][i]]=math.floor(factor*(max_col_1+4)+count_exception)
      count_exception+=1
    else:
      if matrix[t][j] in count_map:
        encoding_map[matrix[t][i]]=math.floor(matrix[t][j]*factor+count_map[matrix[t][j]])
        count_map[matrix[t][j]]+=1
      else:
        count_map[matrix[t][j]]=0
        encoding_map[matrix[t][i]]=math.floor(matrix[t][j]*factor+count_map[matrix[t][j]])
        count_map[matrix[t][j]]+=1

  one_one_val=0
  for key in count_map.keys():      
    if len(sec_prim_map[key])==1:
      one_one_val+=1

      

  for t in range(0,len(matrix)):
    matrix[t][i]=encoding_map[matrix[t][i]]

  for t in temp_exception_list_0:
    correl_data_struct.exception_list_0.append(encoding_map[t])  

  for t in temp_exception_list_not_one:
    correl_data_struct.exception_list_not_one.append(encoding_map[t])   

  print("one one mappings are:",one_one_val,"proportion:",one_one_val*1.00/len(prim_sec_map.keys()))
  
  print("many to many vals:",len(correl_data_struct.exception_list_0)*1.00/len(prim_sec_map.keys()))
  print("one to many vals:",(len(correl_data_struct.exception_list_not_one)-len(correl_data_struct.exception_list_0))*1.00/len(prim_sec_map.keys()))  
  print("one to one vals:",1.00-(len(correl_data_struct.exception_list_not_one)*1.00/len(prim_sec_map.keys())))   

  a=np.array(matrix)
  temp_matrix=unique_rows(a)

  print("Sanity Check unique rows after",len(temp_matrix))  
  print("\n\n")

  return correl_data_struct  

def analyse_fpr(matrix,df,i,j,correl_data_struct,target_fpr,block_size):

  num_blocks=math.floor(len(matrix)/block_size)

  print("num blocks:",num_blocks)


  many_many_elements=set(correl_data_struct.exception_list_0)
  one_many_elements=set(correl_data_struct.exception_list_not_one)


  size_correl=0.0
  size_normal=0.0

  block_bloom_list_0_normal=[]
  block_bloom_list_0_correl=[]
  block_bloom_list_1=[]

  block_set_0=[]
  block_set_1=[]


  for t in range(0,num_blocks):
    block_set_0.append(set([]))
    block_set_1.append(set([]))

  for t in range(0,int(block_size*num_blocks)):
    ind=math.floor(t/block_size)
    block_set_0[ind].add(matrix[t][i])
    block_set_1[ind].add(matrix[t][j])

    
  for t in range(0,num_blocks):

    count_to_add=0

    for item in block_set_0[t]:
      if item in one_many_elements:
        count_to_add+=1

    block_bloom_list_0_correl.append(BloomFilter(count_to_add, target_fpr))
    block_bloom_list_0_normal.append(BloomFilter(len(block_set_0[t]), target_fpr))
    block_bloom_list_1.append(BloomFilter(len(block_set_1[t]), target_fpr)) 

    for item in block_set_0[t]:
      block_bloom_list_0_normal[-1].add(item)
      if item in one_many_elements:
        block_bloom_list_0_correl[-1].add(item)

    # print("perecentage used:",count_to_add*1.00/len(block_set_0[t]))    

    for item in block_set_1[t]:
      block_bloom_list_1[-1].add(item)  

    size_normal+=1.44*math.log(1.00/target_fpr,2)*len(block_set_0[t])  
    size_correl+=1.44*math.log(1.00/target_fpr,2)*count_to_add

  print("Size Ratio:",size_correl*1.00/size_normal)
  # correl_bf=BloomFilter(len(correl_data_struct.exception_list_0), 0.01)
  # for item in correl_data_struct.exception_list_0:
  #   correl_bf.add(item)
  #   # print(item)

  # correl_bf_not_one=BloomFilter(len(correl_data_struct.exception_list_not_one), 0.01)
  # for item in correl_data_struct.exception_list_not_one:
  #   correl_bf_not_one.add(item)  

  # size_correl=size_normal
  # size_correl+=1.44*math.log(1.00/0.01,2)*len(correl_data_struct.exception_list_0)
  # size_correl+=1.44*math.log(1.00/0.01,2)*len(correl_data_struct.exception_list_not_one)

  num_queries_per_block=1000

  total_negatives=0
  total_false_positives_normal=0
  total_false_positives_correl=0



  


  for curr_block in tqdm(range(0,num_blocks)):
    rand_list=np.random.uniform(0,1.0,num_queries_per_block)

    for t in range(0,num_queries_per_block):
      ind=math.floor(rand_list[t]*num_blocks*block_size)

      if matrix[ind][i] in block_set_0[curr_block]:
        if matrix[ind][i] not in many_many_elements:
          val=math.floor(matrix[ind][i]/correl_data_struct.factor_0_to_1)
          if val not in block_bloom_list_1[curr_block] or val not in block_set_1[curr_block]:
            while(True):
              print("ERROR",val,matrix[ind][i],matrix[ind][j])
        continue

      total_negatives+=1


      
      if matrix[ind][i] in block_bloom_list_0_normal[curr_block]:
        total_false_positives_normal+=1
        

      if matrix[ind][i] in many_many_elements:
        if matrix[ind][i] in block_bloom_list_0_correl[curr_block]:
          total_false_positives_correl+=1
      else:
        val=math.floor(matrix[ind][i]/correl_data_struct.factor_0_to_1)
        if matrix[ind][i] in one_many_elements:
          if matrix[ind][i] in block_bloom_list_0_correl[curr_block] and val in block_bloom_list_1[curr_block]:
            total_false_positives_correl+=1
        else:
          if val in block_bloom_list_1[curr_block]:
            total_false_positives_correl+=1      
        

  fpr_correl=total_false_positives_correl*1.00/total_negatives
  fpr_normal=total_false_positives_normal*1.00/total_negatives
  print("Normal False positive rate:",fpr_normal)
  print("Correl False positive rate:",fpr_correl)       



  print("\n\n")


  return  fpr_correl,size_correl,fpr_normal,size_normal        


def benchmark_vortex(file_name,acceptance_list):

  col_name=[]
  if 'dmv' in file_name:
    df=pd.read_csv(file_name)
  else:
    df=pd.read_csv(file_name,header=None,sep='|')  
    for i in range(0,len(df.columns)):
      col_name.append(str(i))

    df.columns=col_name 
  
  col_list=list(df.columns)
  for i in range(0,len(col_list)):
    # if "VIN" not in col_list[i]:
    #   continue
    # if "Date" in col_list[i]:
    #   continue
    # continue
    if col_list[i] in acceptance_list:
      continue
    df=df.drop(col_list[i], 1)  

  columns_list=list(df.columns)
  
  print(col_list)
  print('df stuff',df.columns)
  matrix=df.values.tolist()

  card_list={}

  for i in range(0,len(matrix[0])):

    if i>=0:
      continue

    print("col type",df.columns[i],type(matrix[0][i]),matrix[0][i])
    if type(1.00)==type(matrix[0][i]):
      df[df.columns[i]] = df[df.columns[i]].fillna(0-1)

      for j in range(0,len(matrix)):
        if np.isnan(matrix[j][i]):
          matrix[j][i]=0-1

    if True or type("string")==type(matrix[0][i]):
    # if True:  
      map_dict={}
      # print(matrix[i])
      temp_col=[]
      for t in range(0,len(matrix)):
        map_dict[matrix[t][i]]=0
        temp_col.append(matrix[t][i])

      print("sorting by freq")  
      # new_list = sorted(temp_col, key = temp_col.count, reverse=True)
      # new_list= a.sort(key=Counter(a).get, reverse=True)
      new_list= list(chain.from_iterable(repeat(i, c) for i,c in Counter(temp_col).most_common()))
      print("sorting by freq done")
  
      count=0  
      map_dict[new_list[0]]=0
      for t in range(0,len(new_list)):
        if t==0:
          continue
        if new_list[t]!=new_list[t-1]:
          count+=1
        map_dict[new_list[t]]=count  

      print("cardinality of col",i,"is:",count) 
      
      card_list[i]=count 
      card_order.append(count)
      df[df.columns[i]] = df[df.columns[i]].map(map_dict)  
      for j in range(0,len(matrix)):
        # print(matrix[j][j],map_dict[matrix[i][j]])
        matrix[j][i]=map_dict[matrix[j][i]]

  # for i in tqdm(range(0,len(matrix[0]))):
  #   for j in tqdm(range(0,len(matrix[0]))):
  #     if i<=j:
  #       continue
  #     check_map_val(matrix,df,i,j)
  #     check_map_val(matrix,df,j,i)

  random.shuffle(matrix)

  correl_data_struct=encode_discrete(matrix,df,0,1)  


  correl_size_list=[]
  correl_fpr_list=[]
  normal_size_list=[]
  normal_fpr_list=[]

  # fpr_acheive_list=[0.5,0.2,0.1,0.01,0.001]

  fpr_acheive_list=[0.001,0.004,0.01,0.1]

  for t in tqdm(range(0,len(fpr_acheive_list))):
    fpr_correl,size_correl,fpr_normal,size_normal=analyse_fpr(matrix,df,0,1,correl_data_struct,fpr_acheive_list[t],math.pow(2,13))

    correl_size_list.append(size_correl/1000.0)
    correl_fpr_list.append(fpr_correl)
    normal_size_list.append(size_normal/1000.0)
    normal_fpr_list.append(fpr_normal)


  print(correl_fpr_list)
  print(correl_size_list)
  print(normal_fpr_list)
  print(normal_size_list)
    
  plt.plot(correl_size_list,correl_fpr_list, marker='x', markerfacecolor='black', markersize=8, color='red',linestyle='-', linewidth=2,label='Correl')
  plt.plot(normal_size_list,normal_fpr_list, marker='x', markerfacecolor='black', markersize=8, color='orange',linestyle='--', linewidth=2,label='Normal')
  plt.yscale('log')
  plt.legend(loc='upper right')
  plt.xlabel('Size')
  plt.ylabel('False Positive Rate')
  plt.xlim(left=0)
  plt.tight_layout()
  # if only_last:
  #   plt.savefig("plbf_gauss_low_last_3.png")
  # else:
  #   plt.savefig("plbf_gauss_low_3.png")
  plt.savefig("discrete_correl.png")

  plt.show()
  plt.clf()
  
  return   








#DMV
file_name="subset_data/dmv_small.csv"
# file_name="dmv_tiny.csv"
# acceptance_list=["City","State"]
# acceptance_list=["State","Zip"]
# acceptance_list=["Record Type","Registration Class"]
# acceptance_list=["City","Zip","County","State","Color","Model Year"]
acceptance_list=["City","Zip"]
# acceptance_list=["City","Zip","County","State","Color"]
# query_cols=[["City","Color"],["City"],["Color"],["Body Type"],["State","Color"],["City","Color","State"],["Color","Body Type"]]
# target_fpr=0.01

# benchamrk_bloom_filter_real(file_name,acceptance_list)
benchmark_vortex(file_name,acceptance_list)