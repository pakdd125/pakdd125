import os
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import time
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def Sort(sub_li):finallyoutput = []; sort_sub_li = sorted(sub_li.items(), key=lambda x: x[1], reverse=True);return sort_sub_li

datasets = ["ecommerce.txt","chainstore.txt","fruithut_utility.txt","liquor_15.txt"] #datasets
folders = ['Liquor','Fruithut','Chainstor','Ecommerce'] #folders for files to be stored

def getprice(itemset,prices): #get price of itemset
	p = 0
	for each in itemset:
		p = p + prices[each]
	return p

def makesets(): #make datasets
	try:os.mkdir("Ecommerce");os.mkdir("Fruithut");os.mkdir("Liquor");os.mkdir("Chainstore")
	except:print("Folder exists")
	d1 = open("fruithut_utility.txt","r");d2 = open("liquor_15.txt","r");d3=open("chainstore.txt","r");d4=open("ecommerce.txt","r")
	d1trans = d1.readlines(); d1len = len(d1trans); d1train = int(d1len * 0.8);d2trans = d2.readlines(); d2len = len(d2trans); d2train = int(d2len * 0.8);d3trans = d3.readlines(); d3len = len(d3trans); d3train = int(d3len * 0.8);d4trans = d4.readlines(); d4len = len(d4trans); d4train = int(d4len * 0.8)
	traind1 = open("Fruithut/train.txt","w"); testd1 = open("Fruithut/test.txt","w");traind2 = open("Liquor/train.txt","w"); testd2 = open("Liquor/test.txt","w");traind3 = open("Chainstore/train.txt","w"); testd3 = open("Chainstore/test.txt","w");traind4 = open("Ecommerce/train.txt","w"); testd4 = open("Ecommerce/test.txt","w")
	traind1.writelines("%s" % item for item in d1trans[:d1train]);testd1.writelines("%s" % item for item in d1trans[d1train:]);traind2.writelines("%s" % item for item in d2trans[:d2train]);testd2.writelines("%s" % item for item in d2trans[d2train:]);traind3.writelines("%s" % item for item in d3trans[:d3train]);testd3.writelines("%s" % item for item in d3trans[d3train:]);traind4.writelines("%s" % item for item in d4trans[:d4train]);testd4.writelines("%s" % item for item in d4trans[d4train:])
makesets()
def generateprice(link): #get price values from dataset

	dictionaryofprice = {};file = open(link,"r"); retaildata = file.readlines()
	for each in retaildata:
		each = each.strip();each = each.split(":");temp = each[0];temp = temp.split(" ");temp1 = each[-1];temp1 = temp1.split(" ");temp = [int(item) for item in temp]
		for item in temp:
			try:
				price = dictionaryofprice[item];ind = temp.index(item);newprice = float(temp1[ind])
				if newprice < price:dictionaryofprice[item] = newprice
			except:
				ind = temp.index(item);dictionaryofprice[item] = float(temp1[ind])
	return dictionaryofprice

def generatefreq(link): #get frequency values from dataset

	dictionary = {}
	file = open(link,"r");retaildata = file.readlines()
	for each in retaildata:
		each = each.strip();each = each.split(":");each=each[0];each = each.split(" ")
		each = [int(item) for item in each]
		for item in each:
			try:frequency = dictionary[item];frequency += 1;dictionary[item] = frequency
			except:dictionary[item] = 1
	return (dictionary)

#to create itemsets, we follow the following approach:

#Itemsets are mined in a level-wise manner. At the onset, we scan the transactional dataset D. 
#The first level contains the top-lambda high-utility items. To build level 2, we scan the transactional
#dataset D. Note that each item in D has a unique identifier, designated as IID, which is an integer. If item
# i occurs in a transaction t, we use the remaining items (of t) occurring at level 1 such that they have higher
# IID than that of i. For example, while iterating through the items in level 1, consider item 5 as the input item
# for creating itemsets for level 2, and consider a sample transaction {3,5,6,7}. Assume that items 3,5,6 and 7
# occur at level 1. Now we consider the remaining items with a higher IID than 5 i.e., {6,7} and use them to create itemsets {5,6}, {5,7}.

#Intuitively, {3,5} should also qualify for level 2; here, {3,5} will be considered when 3 is the input item for creating itemsets of size 2.
# If {5,6} and {5,7} already exist in the itemset linked list of level 2, we simply increment their frequencies by 1, thereby saving
# time in computing itemset frequencies. Observe how by following the above approach, we avoid generating duplicate itemsets
# e.g., {A,B} and {B,A}. Once itemsets have been generated for level 2 as discussed above, we sort the itemsets in
# descending order of utility. Then, we progressively add itemsets at level 2 of the index, if and only if they belong to the top-lambda sorted itemsets.

#For creating higher levels, say level n, we essentially follow the same process, while considering itemsets of level (n-1) and items at level 1.
# We do so in the following manner. For items in transactions that have lower IIDs than the highest item IID in the input itemset, we create a temporary itemset TI
# with the remaining items that excludes the item with the highest IID, and check if TI occurs at the previous level of UPI. For example, assuming that items 3,5,6 and 7
# occur at level 1, for input itemset {6,7} and transaction {3,5,6,7}, we have two TIs, namely {5,6} and {3,6}, as 7 is the item with the highest IID.
# We proceed to check if {5,6} and {3,6} occur at level 2. If not, then we proceed to create the new itemsets, along with the item with the highest IID,
# for the next level, i.e., {5,6,7} and {3,6,7}. We increment their frequencies by 1, thereby saving time in computing itemset frequencies. However,
# if it does occur at the previous level, then we discard the corresponding itemsets, as they would automatically become a part of this level during
# future iterations, i.e., when {5,6} and {3,6} are input itemsets. For items in transactions that have higher IIDs than the highest item IID in the input itemset,
# we adhere to the same process as described earlier.

def make_itemsets(level1,leveln_1,train,prices,frequencies,num_itemsets_at_each_level):

	level1items = [item[0][0] for item in level1];input_itemsets = [item[0] for item in leveln_1];output = {}
	for itemset in tqdm(input_itemsets):
		for trans in train:
			if set(itemset).issubset(trans):
				items = set(trans) - set(itemset)
				for item in items:
					if (item > max(itemset)) and (item in level1items):
						newitemset = tuple(itemset + [item])
						try:
							freq = output[newitemset];freq = freq + 1;output[newitemset] = freq
						except:
							output[newitemset] = 1
					elif (item in level1items):
						newitemset = sorted([item]+itemset)
						if newitemset[:-1] not in input_itemsets:
							newitemset = tuple(newitemset)
							try:
								freq = output[newitemset];freq += 1;output[newitemset] = freq
							except:
								output[newitemset] = 1
	output = {key: value * getprice(key,prices) for key, value in output.items()}
	output = sorted(output.items(), key=lambda x: x[0]);output = output[:num_itemsets_at_each_level]; output = [[list(sublist), number] for (sublist, number) in output];output = [item + [getprice(item[0],prices)] for item in output]
	return output

for folder in folders:

	print(folder)

	num_itemsets_at_each_level = 3000
	utility_threshold = 0; support_threshold = 0

	train = open(str(folder)+"/train.txt","r");	test = open(str(folder)+"/test.txt","r");train = train.readlines();test = test.readlines()
	train = [item.strip() for item in train];train = [item.split(':') for item in train];train=[item[0] for item in train];train = [item.split(' ') for item in train]
	test = [item.strip() for item in test];test = [item.split(':') for item in test];test=[item[0] for item in test];test = [item.split(' ') for item in test]
	train = [[int(item) for item in inner_list] for inner_list in train];test = [[int(item) for item in inner_list] for inner_list in test]
	prices = generateprice(str(folder)+"/train.txt")
	frequencies = generatefreq(str(folder)+"/train.txt")
	utility = {item: prices[item] * frequencies[item] for item in prices}
	train = [[num for num in inner_list if num in prices] for inner_list in train];test = [[num for num in inner_list if num in prices] for inner_list in test]

	kui_l1 = Sort(utility);kui_l1 = kui_l1[:num_itemsets_at_each_level];kui_l1 = [list(t) for t in kui_l1];kui_l1 = [[[item[0]],item[1]] for item in kui_l1];kui_l1 = [item + [getprice(item[0],prices)] for item in kui_l1]
	level2 = make_itemsets(kui_l1,kui_l1,train,prices,frequencies,num_itemsets_at_each_level)
	level3 = make_itemsets(kui_l1,level2,train,prices,frequencies,num_itemsets_at_each_level)
	level4 = make_itemsets(kui_l1,level3,train,prices,frequencies,num_itemsets_at_each_level)
	level5 = make_itemsets(kui_l1,level4,train,prices,frequencies,num_itemsets_at_each_level)
	level6 = make_itemsets(kui_l1,level5,train,prices,frequencies,num_itemsets_at_each_level)
	total = kui_l1+level2+level3+level4+level5+level6
	total = sorted(total, key=lambda x: x[1], reverse=True)

	#each entry is of the format: {itemset, revenue, price}
	#here we end the itemset creation part

	#---------------------------------------------------------------------------------------------------------------------------
	
	variations_in_s = [200,400,600,800,1000];temp_var = 10000000000000;num_substitutes = 20;num_isix = 10 #variables

	for run in range(5): #number of times experiment is run
		print("RUN:"+str(run))
		for x in variations_in_s:
			if x ==600:variations_in_ti = [500,1000,1500,2000,2500]
			else:variations_in_ti = [1500]
			isix = {};enriched_test = [];updated_prices = [];random_s = random.sample(total, x);all_s_itemsets = [];dict_enriched_subs = {}
			#creation of stix index (referred to as isix, which was the original name)
			for entries in random_s:
				alpha_dict = {}
				itemset = entries[0]
				all_s_itemsets.append(itemset)
				new_items = [temp_var + i for i in range(1, num_substitutes+1)]
				dict_enriched_subs[tuple(itemset)] = tuple(new_items)
				for new_item in new_items:alpha = random.uniform(0.75, 1);alpha_dict[new_item] = alpha;prices[new_item] = alpha * entries[2]
				isix_temp = dict(sorted(alpha_dict.items(), key=lambda item: item[1], reverse=True)[:num_isix])
				list_of_tuples = [(key, value) for key, value in isix_temp.items()]
				isix[tuple(itemset)] = tuple(list_of_tuples);t_isix = [key for key,value in isix_temp.items()];temp_var += (num_substitutes)
			sorted_s = sorted(all_s_itemsets, key=len)
			#we now enrich datasets and proceed to experimental results
			for trans in tqdm(test):
				temp_lis = []
				for itemset in sorted_s:
					if set(itemset).issubset(set(trans)):
						trans += dict_enriched_subs[tuple(itemset)];temp_lis += itemset
				trans = list(set(trans) - set(temp_lis));enriched_test.append(trans)
			#enriched_test = list(enriched_test for enriched_test,_ in itertools.groupby(enriched_test))
			for ti in variations_in_ti:
				our_itemsets = [];our_ti = ti;ourtrr= 0;reftrr=0;time1 = time.time() - time.time()
				for each in total:
					itemset = each[0]
					if our_ti >0:
						start_time1 = time.time()
						for sub in sorted_s:
							if set(sub).issubset(set(itemset)):
								subs = isix[tuple(sub)];subs = [val[0] for val in subs];t_itemset = list(set(itemset) - set(sub))
								for t_each in subs:
									if t_itemset+[t_each] not in our_itemsets:our_itemsets.append(t_itemset+[t_each])
							else:
								if itemset not in our_itemsets:our_itemsets.append(itemset)	
						our_ti -= 1;time1 += time.time() - start_time1
				start_time2 = time.time()
				ref_itemsets = [xx[0] for xx in total]
				#random.shuffle(ref_itemsets)
				final_ref = []
				ti_ref = ti
				for each in ref_itemsets:
					a = 0
					for soob in sorted_s:
						if set(soob).issubset(set(each)):a = 1
					if a ==0:
						if ti_ref>0:final_ref.append(each);ti_ref -= 1
					else:ti_ref -= 1
				time2 = time.time() - start_time2

				for trans in tqdm(enriched_test):
					for ite in our_itemsets:
						if set(ite).issubset(set(trans)):ourtrr += getprice(ite,prices)
					for ite in final_ref:
						if set(ite).issubset(set(trans)):reftrr += getprice(ite,prices)
				print(str(ourtrr),str(reftrr),str(time1),str(time2))