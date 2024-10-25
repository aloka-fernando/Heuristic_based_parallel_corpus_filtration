# Created by  aloka on 23/03/2024

#imports
import torch
import time
import os
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
min_sent_ratio=0.79
max_sent_ratio=1.31


#outputs
filter="sentRatio"
src_lang="en"
tgt_lang="si"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_sentRatio"

#output files
sub_dir = "csv_files"
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
laser3_sorted_src_slength_csv_file="{}-{}-laser3-sorted-{}.train.csv".format(src_lang, tgt_lang, filter)
xlmr_sorted_src_slength_csv_file="{}-{}-xlmr-sorted-{}.train.csv".format(src_lang, tgt_lang, filter)
labse_sorted_src_slength_csv_file="{}-{}-labse-sorted-{}.train.csv".format(src_lang, tgt_lang, filter)

###############################################################
#main logic
###############################################################

#read data
org_dataset_df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print('Dataset size original : {}'.format(len(org_dataset_df)))


#slength filtration
org_dataset_df["src_sent_length"] = org_dataset_df['src_sents'].apply(lambda x: len(str(x).split()))
org_dataset_df["tgt_sent_length"] = org_dataset_df['tgt_sents'].apply(lambda x: len(str(x).split()))
org_dataset_df["sentRatio"] = org_dataset_df['src_sent_length'] / org_dataset_df['tgt_sent_length']
print(org_dataset_df.head(5))

sentRatio_df = org_dataset_df.loc[(org_dataset_df['sentRatio'] >= min_sent_ratio) & (org_dataset_df['sentRatio'] <= max_sent_ratio)]
print('Dataset size after sentRatio filtering : {}'.format(len(sentRatio_df)))

#laser3_scores sorted
laser3_src_slength_sorted_df= sentRatio_df.sort_values("laser3_scores", ascending=False)
laser3_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_slength_csv_file), index=False)
#xlmr_scores sorted
xlmr_src_slength_sorted_df= sentRatio_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_slength_csv_file), index=False)
#labse_scores sorted
labse_src_slength_sorted_df= sentRatio_df.sort_values("labse_scores", ascending=False)
labse_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_slength_csv_file), index=False)


endTime = time.time()
print('Time taken to complete slength filtration: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")