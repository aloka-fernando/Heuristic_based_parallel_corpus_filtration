# Created by  aloka on 23/03/2024

#imports
import torch
import time
import os
import string
import re
import pandas as pd
from datasets import load_dataset
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util


startTime=time.time()
#inputs
data_dir="/userdirs/aloka/datasets/opus/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores.en-si.csv"
#input_file= open("{}/{}".format(data_dir, input_file), "r", encoding="utf8")

#parameters
sentCharRatio_threshold=0.6
special_symbol_pattern = re.compile(r'[!@#$%^&*()\-_=+\[\]{}|;:,.<>?/~`]')

#outputs
filter="sentCharRatio"
src_lang="en"
tgt_lang="si"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_{}".format(filter)

#output files
sub_dir = "csv_files"
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory Exists...........")

laser3_sorted_src_sentCharRatio_csv_file="{}-{}-laser3-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
laser3_sorted_tgt_sentCharRatio_csv_file="{}-{}-laser3-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
laser3_sorted_src_tgt_sentCharRatio_csv_file="{}-{}-laser3-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_src_sentCharRatio_csv_file="{}-{}-xlmr-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_tgt_sentCharRatio_csv_file="{}-{}-xlmr-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_src_tgt_sentCharRatio_csv_file="{}-{}-xlmr-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_src_sentCharRatio_csv_file="{}-{}-labse-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_tgt_sentCharRatio_csv_file="{}-{}-labse-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_src_tgt_sentCharRatio_csv_file="{}-{}-labse-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)

#word/sentence length ratio
def get_sentCharRatio(sent):
    #chars = [char for char in "".join(str(sent).split())]
    char_list = [not(char.isdigit() or char in string.punctuation) for char in str(sent)]
    return round(sum(char_list) / len(char_list), 2)


###############################################################
#main logic
###############################################################

#read data
org_dataset_df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print('Dataset size original : {}'.format(len(org_dataset_df)))


#sentTextRatio filtration
org_dataset_df["src_sentCharRatio"] = org_dataset_df['src_sents'].apply(get_sentCharRatio)
org_dataset_df["tgt_sentCharRatio"] = org_dataset_df['tgt_sents'].apply(get_sentCharRatio)
print(org_dataset_df.head(5))

src_sentCharRatio_df = org_dataset_df.loc[(org_dataset_df['src_sentCharRatio'] >= sentCharRatio_threshold)]
print('Dataset size after src_sentCharRatio filtering : {}'.format(len(src_sentCharRatio_df)))
tgt_sentCharRatio_df = org_dataset_df.loc[(org_dataset_df['tgt_sentCharRatio'] >= sentCharRatio_threshold)]
print('Dataset size after src_sentCharRatio filtering : {}'.format(len(tgt_sentCharRatio_df)))
src_tgt_sentCharRatio_df = src_sentCharRatio_df.loc[(org_dataset_df['tgt_sentCharRatio'] >= sentCharRatio_threshold)]
print('Dataset size after src_tgt_sentCharRatio_df filtering : {}'.format(len(src_tgt_sentCharRatio_df)))

#laser3_scores sorted
laser3_src_sentCharRatio_sorted_df= src_sentCharRatio_df.sort_values("laser3_scores", ascending=False)
laser3_src_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_sentCharRatio_csv_file), index=False)
laser3_tgt_sentCharRatio_sorted_df= tgt_sentCharRatio_df.sort_values("laser3_scores", ascending=False)
laser3_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_tgt_sentCharRatio_csv_file), index=False)
laser3_src_tgt_sentCharRatio_sorted_df= src_tgt_sentCharRatio_df.sort_values("laser3_scores", ascending=False)
laser3_src_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_tgt_sentCharRatio_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_sentCharRatio_sorted_df= src_sentCharRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_sentCharRatio_csv_file), index=False)
xlmr_tgt_sentCharRatio_sorted_df= tgt_sentCharRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_tgt_sentCharRatio_csv_file), index=False)
xlmr_src_tgt_sentCharRatio_sorted_df= src_tgt_sentCharRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_tgt_sentCharRatio_csv_file), index=False)

#labse_scores sorted
labse_src_sentCharRatio_sorted_df= src_sentCharRatio_df.sort_values("labse_scores", ascending=False)
labse_src_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_sentCharRatio_csv_file), index=False)
labse_tgt_sentCharRatio_sorted_df= tgt_sentCharRatio_df.sort_values("labse_scores", ascending=False)
labse_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_tgt_sentCharRatio_csv_file), index=False)
labse_src_tgt_sentCharRatio_sorted_df= src_tgt_sentCharRatio_df.sort_values("labse_scores", ascending=False)
labse_src_tgt_sentCharRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_tgt_sentCharRatio_csv_file), index=False)


endTime = time.time()
print('Time taken to complete sentTextRatio filtration: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")