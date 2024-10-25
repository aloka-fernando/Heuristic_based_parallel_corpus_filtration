################################################################################################
#v5 - run all ngram segments during similarity matching
#v6- 07.06.2024 reducing dataset size by removing  lines with zero ngram_segments, duplicates
#v7 - fixed reindexing issue
#v8 - split
#v12 - indexing substrings
#v13 - reduce search space by removing lines with single occurance of ngram segment - FINAL
################################################################################################

#imports
import torch
import csv
import numpy as np
import mapply
import gc
import pickle
import time
import os
import string
import pandas as pd
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util

#debug
debug = False
startTime=time.time()

#parameters
min_ngram = 6

#inputs
data_dir="/home/aloka/p4_filtration/data/ccmatrix_sentSim_7gram_v13/csv_files"
input_file = "en-si-labse-sorted-src-sentSim_7gram.train.csv"
df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print("Original Dataset size : {}".format(len(df)))

#outputs
sub_dir = "csv_files"
exp_file_name="sentSim_{}gram".format(min_ngram)
src_lang="en"
tgt_lang="si"
output_dir="/home/aloka/p4_filtration/data/ccmatrix_{}_v13/".format(exp_file_name)
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory exists...")

#output files
laser3_src_sorted_csv_file="{}-{}-laser3-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)
xlmr_src_sorted_csv_file="{}-{}-xlmr-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)
labse_src_sorted_csv_file="{}-{}-labse-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)

#remove punctuations from string - creates a translation table that maps punctuations to None
translator = str.maketrans('', '', string.punctuation +'–“’…')

#remove punctuations & special characters
def get_punct_rm_sentence(text):
    return str(text).translate(translator)

#returns ngram chunnks
def get_substring_ngram_segments(text):
    ngram_segments = []
    sentTkns = [str(token) for token in str(text).strip().split()]
    if len(sentTkns) >= min_ngram:
        ngram_segments = [" ".join(sentTkns[i:i+min_ngram]) for i in range(len(sentTkns) - min_ngram+1)]
    return ngram_segments



#################################################################################################3
#Add Columns
##################################################################################################

#add columns
df["src_sent_length"] = df["src_sents"].str.split().str.len()
df["src_mod_sents"] = df["src_sents"].apply(get_punct_rm_sentence)
print("Punctuation removal completed.....................")
df["src_substring_ngram_segments"] = df["src_mod_sents"].apply(get_substring_ngram_segments)


#create substring:freq dictionary
substring_frequencies = {}
for substring_list in df["src_substring_ngram_segments"].tolist():
    for substring in substring_list:
        if substring not in substring_frequencies.keys():
            substring_frequencies[substring]=1
        else:
            substring_frequencies[substring]=substring_frequencies[substring]+1
print("Total number of substring_frequencies.keys() : {}".format(len(substring_frequencies)))

#Unique substrings
print("Total number of Unique substrings : {}".format(len(substring_frequencies.keys())))

#create indexes for substring
indexed_substrings = {}
for index, substring in enumerate(substring_frequencies.keys()):
            indexed_substrings[substring]= index
assert  len(substring_frequencies.keys()) == len(indexed_substrings)

#convert substrings to indexes
def substring2index(substrings):
    return set([int(indexed_substrings[str(substring)]) for substring in substrings])

def get_non_ovelaping_sentences(substring_list):
    if len(substring_list) > 0:
        overlaps = [True for substring in substring_list if substring_frequencies[substring] > 1]
        if any(overlaps):
            return "Overlapping"
        else:
            return "Non-overlapping"
    return "empty"

df["src_substring_ngram_segments_indexes"]=df["src_substring_ngram_segments"].apply(substring2index)
df["non_overlapping"]= df["src_substring_ngram_segments"].apply(get_non_ovelaping_sentences)
# print(df[:5])
# print("Final Columns : {}".format(df.columns.tolist()))

####################################################
##preprocessing - setp1 - #drop duplicates
####################################################
print("\nDeduplicating:")
dedup_df = df.drop_duplicates("src_mod_sents", keep="first")
print("Size of the de-duplicated/duplicate dataset : {}/{}".format(len(dedup_df), len(df[~df.isin(dedup_df)].dropna())))

############################################################
###preprocessing - setp2 - #drop non-overlapping sentences
############################################################
dedup_overlapping_df = dedup_df[dedup_df["non_overlapping"] == "Overlapping"]
print("Dataframe size after valid/short(empty)/unique_sentences : {}/{}/{}".format(len(dedup_overlapping_df), len(dedup_df[dedup_df["non_overlapping"] == "empty"]), len(dedup_df[dedup_df["non_overlapping"] == "Non-overlapping"])))
endTime = time.time()
print('Time taken to complete tgt ngram_segments: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))


#prepare dataset to apply ngram_filtration
preprocessed_df = dedup_overlapping_df.iloc[:,:]
#preprocessed_df["reindexed_ids"] = reindexed_preprocessed_df.index.tolist()
preprocessed_df_copy = preprocessed_df.copy()
dedup_overlapping_df.drop(columns=["src_sents", "tgt_sents", "laser3_scores", "xlmr_scores", "labse_scores", "src_sent_length", "src_mod_sents", "src_substring_ngram_segments", "non_overlapping"])

collected = gc.collect()
print("Garbage collector: collected %d objects.\n" % (collected))

#returns indexes of simSents
def return_sentSim_indexes(row, preprocessed_df_copy):
    print("Index [{}]-----------------------------------------------------------------------".format(row.name))
    preprocessed_df_copy.drop(index=row.name, axis=1, inplace=True)
    preprocessed_df_copy["ref_src_sents_indexes"] = [row["src_substring_ngram_segments_indexes"]]*len(preprocessed_df_copy)
    preprocessed_df_copy["has_overlapping_indexes"] = preprocessed_df_copy.apply(lambda current_row: len((current_row["src_substring_ngram_segments_indexes"]).intersection(current_row["ref_src_sents_indexes"]))>0, axis = 1)
    return preprocessed_df_copy[preprocessed_df_copy["has_overlapping_indexes"] == True].index.tolist()


################################################################
##main logic
#################################################################
mapply.init(
    n_workers=3,
    chunk_size=1,
    max_chunks_per_worker=0,
    progressbar=False,
)

#print("Preprocessed df columns : {}".format(preprocessed_df.columns.tolist()))
preprocessed_df["sentSim_indexes"] = preprocessed_df.mapply(lambda row: return_sentSim_indexes(row, preprocessed_df_copy.copy()), axis=1)
src_sentSim_indexes_to_filter = preprocessed_df[preprocessed_df["sentSim_indexes"].apply(lambda x: len(x) > 0)].index.tolist()
print("Total filtered_org_indexes for filtering : {}".format(len(src_sentSim_indexes_to_filter)))

#########################################################################
##Post-PRocessing
#########################################################################

#remove ignore_indexes from org
print("\nSummary:")
print("Original Dataframe size : {}".format(len(df)))
df.drop(index=df[~df.isin(dedup_df)].dropna().index.tolist(), axis=1, inplace=True)
print("After dropping duplicates from org dataset : {}".format(len(df)))

#remove filtered indexes
#print("Filtered org indexes : {}".format(filtered_org_indexes))
df.drop(index=src_sentSim_indexes_to_filter, axis=1, inplace=True)
print("Total filtered_org_indexes for filtering : {}".format(len(src_sentSim_indexes_to_filter)))
df.drop(columns=["src_sent_length", "src_mod_sents", "src_substring_ngram_segments", "non_overlapping", "src_substring_ngram_segments_indexes"], axis=1, inplace=True)
print(df.columns.tolist())
print("Final df size after removing filtered indexes : {}".format(len(df)))


endTime = time.time()
print('Time taken to complete filtration+merging: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))

#laser3_scores sorted
laser3_src_ngram_filtered_sorted_df= df.sort_values("laser3_scores", ascending=False)
laser3_src_ngram_filtered_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_src_sorted_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_ngram_filtered_sorted_df= df.sort_values("xlmr_scores", ascending=False)
xlmr_src_ngram_filtered_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_src_sorted_csv_file), index=False)

#labse_scores sorted
labse_src_ngram_filtered_sorted_df= df.sort_values("labse_scores", ascending=False)
labse_src_ngram_filtered_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_src_sorted_csv_file), index=False)
#
endTime = time.time()
print('Time taken to complete Script: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")
