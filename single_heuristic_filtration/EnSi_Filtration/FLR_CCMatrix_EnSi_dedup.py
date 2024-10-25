# Created by  aloka on 23/03/2024

#imports
import torch
import csv
import time
import pandas as pd
from datasets import load_dataset
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util


startTime=time.time()

#inputs
data_dir="/userdirs/aloka/datasets/opus/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores.en-si.csv"
df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")

#outputs
corpus="CCMatrix"
src_lang="en"
tgt_lang="si"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_dedup/"

#batch
dataset_size = len(df)
batch_size=100000
no_of_batches = dataset_size // batch_size

count=0
output_df = pd.DataFrame()

#no of batches
if dataset_size % batch_size != 0:
    no_of_batches += 1
print("Dataset size : {}".format(dataset_size))
print('No of batches : {}'.format(no_of_batches))


#output files
sub_dir = "csv_files"
laser3_sorted_src_dedup_csv_file="{}-{}-laser3-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang)
xlmr_sorted_src_dedup_csv_file="{}-{}-xlmr-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang)
labse_sorted_src_dedup_csv_file="{}-{}-labse-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang)

laser3_sorted_tgt_dedup_csv_file="{}-{}-laser3-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, tgt_lang)
xlmr_sorted_tgt_dedup_csv_file="{}-{}-xlmr-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, tgt_lang)
labse_sorted_tgt_dedup_csv_file="{}-{}-labse-sorted-{}-dedup.train.csv".format(src_lang, tgt_lang, tgt_lang)

laser3_sorted_src_tgt_dedup_csv_file="{}-{}-laser3-sorted-{}-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)
xlmr_sorted_src_tgt_dedup_csv_file="{}-{}-xlmr-sorted-{}-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)
labse_sorted_src_tgt_dedup_csv_file="{}-{}-labse-sorted-{}-{}-dedup.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)

###########
#Read data
###########

# src_lines=[]
# tgt_lines=[]
# reader = csv.reader(input_file)
# next(reader)  # skip header

###############################################################
#main logic
###############################################################


# deduplicate within the batch
print('Batch dataset size original : {}'.format(len(df)))
src_dedup_dataset = df.drop_duplicates("src_sents", keep="first")
print('Dataset size after src deduplication : {}'.format(len(src_dedup_dataset)))
tgt_dedup_dataset = df.drop_duplicates("tgt_sents", keep="first")
print('Dataset size after tgt deduplication : {}'.format(len(tgt_dedup_dataset)))
src_tgt_dedup_dataset = src_dedup_dataset.drop_duplicates("tgt_sents", keep="first")
print('Dataset size after src & tgt deduplication : {}'.format(len(src_tgt_dedup_dataset)))

endTime = time.time()
print('Time taken to complete Deduplication: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))


#laser3_scores sorted
laser3_src_dedup_sorted_df= src_dedup_dataset.sort_values("laser3_scores", ascending=False)
laser3_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_dedup_csv_file), index=False)
laser3_tgt_dedup_sorted_df= tgt_dedup_dataset.sort_values("laser3_scores", ascending=False)
laser3_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_tgt_dedup_csv_file), index=False)
laser3_src_tgt_dedup_sorted_df= src_tgt_dedup_dataset.sort_values("laser3_scores", ascending=False)
laser3_src_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_tgt_dedup_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_dedup_sorted_df= src_dedup_dataset.sort_values("xlmr_scores", ascending=False)
xlmr_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_dedup_csv_file), index=False)
xlmr_tgt_dedup_sorted_df= tgt_dedup_dataset.sort_values("xlmr_scores", ascending=False)
xlmr_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_tgt_dedup_csv_file), index=False)
xlmr_src_tgt_dedup_sorted_df= src_tgt_dedup_dataset.sort_values("xlmr_scores", ascending=False)
xlmr_src_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_tgt_dedup_csv_file), index=False)

#labse_scores sorted
labse_src_dedup_sorted_df= src_dedup_dataset.sort_values("labse_scores", ascending=False)
labse_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_dedup_csv_file), index=False)
labse_tgt_dedup_sorted_df= tgt_dedup_dataset.sort_values("labse_scores", ascending=False)
labse_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_tgt_dedup_csv_file), index=False)
labse_src_tgt_dedup_sorted_df= src_tgt_dedup_dataset.sort_values("labse_scores", ascending=False)
labse_src_tgt_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_tgt_dedup_csv_file), index=False)

endTime = time.time()
print('Time taken to complete Script: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")