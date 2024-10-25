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
sentWordRatio_threshold=0.6
special_symbol_pattern = re.compile(r'[!@#$%^&*()\-_=+\[\]{}|;:,.<>?/~`]')

#outputs
filter="sentWordRatio"
src_lang="en"
tgt_lang="si"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_{}".format(filter)

#output files
sub_dir = "csv_files"
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory Exists...........")

laser3_sorted_src_sentWordRatio_csv_file="{}-{}-laser3-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
laser3_sorted_tgt_sentWordRatio_csv_file="{}-{}-laser3-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
laser3_sorted_src_tgt_sentWordRatio_csv_file="{}-{}-laser3-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_src_sentWordRatio_csv_file="{}-{}-xlmr-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_tgt_sentWordRatio_csv_file="{}-{}-xlmr-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_src_tgt_sentWordRatio_csv_file="{}-{}-xlmr-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_src_sentWordRatio_csv_file="{}-{}-labse-sorted-src-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_tgt_sentWordRatio_csv_file="{}-{}-labse-sorted-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_src_tgt_sentWordRatio_csv_file="{}-{}-labse-sorted-src-tgt-{}.train.csv".format(src_lang, tgt_lang, filter)

#word/sentence length ratio
def get_sentWordRatio(sent):
    word_list = [word.strip(string.punctuation) for word in str(sent).strip().split()]
    isWord_list = [not (any(char.isdigit() for char in word) or all(char in string.punctuation for char in word) or bool(special_symbol_pattern.search(word))) for word in word_list]
    return round(sum(isWord_list) / len(isWord_list), 2)


###############################################################
#main logic
###############################################################

#read data
org_dataset_df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print('Dataset size original : {}'.format(len(org_dataset_df)))


#sentTextRatio filtration
org_dataset_df["src_sentWordRatio"] = org_dataset_df['src_sents'].apply(get_sentWordRatio)
org_dataset_df["tgt_sentWordRatio"] = org_dataset_df['tgt_sents'].apply(get_sentWordRatio)
print(org_dataset_df.head(5))

src_sentWordRatio_df = org_dataset_df.loc[(org_dataset_df['src_sentWordRatio'] >= sentWordRatio_threshold)]
print('Dataset size after src_sentWordRatio filtering : {}'.format(len(src_sentWordRatio_df)))
tgt_sentWordRatio_df = org_dataset_df.loc[(org_dataset_df['tgt_sentWordRatio'] >= sentWordRatio_threshold)]
print('Dataset size after tgt_sentWordRatio_df filtering : {}'.format(len(tgt_sentWordRatio_df)))
src_tgt_sentWordRatio_df = src_sentWordRatio_df.loc[(org_dataset_df['tgt_sentWordRatio'] >= sentWordRatio_threshold)]
print('Dataset size after src_tgt_sentWordRatio_df filtering : {}'.format(len(src_tgt_sentWordRatio_df)))

#laser3_scores sorted
laser3_src_sentWordRatio_sorted_df= src_sentWordRatio_df.sort_values("laser3_scores", ascending=False)
laser3_src_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_sentWordRatio_csv_file), index=False)
laser3_tgt_sentWordRatio_sorted_df= tgt_sentWordRatio_df.sort_values("laser3_scores", ascending=False)
laser3_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_tgt_sentWordRatio_csv_file), index=False)
laser3_src_tgt_sentWordRatio_sorted_df= src_tgt_sentWordRatio_df.sort_values("laser3_scores", ascending=False)
laser3_src_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_tgt_sentWordRatio_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_sentWordRatio_sorted_df= src_sentWordRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_sentWordRatio_csv_file), index=False)
xlmr_tgt_sentWordRatio_sorted_df= tgt_sentWordRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_tgt_sentWordRatio_csv_file), index=False)
xlmr_src_tgt_sentWordRatio_sorted_df= src_tgt_sentWordRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_tgt_sentWordRatio_csv_file), index=False)

#labse_scores sorted
labse_src_sentWordRatio_sorted_df= src_sentWordRatio_df.sort_values("labse_scores", ascending=False)
labse_src_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_sentWordRatio_csv_file), index=False)
labse_tgt_sentWordRatio_sorted_df= tgt_sentWordRatio_df.sort_values("labse_scores", ascending=False)
labse_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_tgt_sentWordRatio_csv_file), index=False)
labse_src_tgt_sentWordRatio_sorted_df= src_tgt_sentWordRatio_df.sort_values("labse_scores", ascending=False)
labse_src_tgt_sentWordRatio_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_tgt_sentWordRatio_csv_file), index=False)


endTime = time.time()
print('Time taken to complete sentTextRatio filtration: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")