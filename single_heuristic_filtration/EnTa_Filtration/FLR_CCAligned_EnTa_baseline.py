# Created by  aloka on 23/03/2024

#imports
import torch
import os
import csv
import time
import pandas as pd
from datasets import load_dataset
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util


startTime=time.time()

#inputs
data_dir="/userdirs/aloka/datasets/opus/CCAligned/en-ta.txt"
input_file = "CCAligned-scores.en-ta.csv"
df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")

#outputs
corpus="CCAligned"
src_lang="en"
tgt_lang="ta"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccaligned_baseline/"
sub_dir = "csv_files"

if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory Exists...........")

laser3_baseline_csv_file="{}-{}-laser3-sorted-baseline.train.csv".format(src_lang, tgt_lang)
xlmr_baseline_csv_file="{}-{}-xlmr-sorted-baseline.train.csv".format(src_lang, tgt_lang)
labse_baseline_csv_file="{}-{}-labse-sorted-baseline.train.csv".format(src_lang, tgt_lang)

#batch
dataset_size = len(df)
print("Dataset size : {}".format(dataset_size))


#################################################################
#main logic
###############################################################

#laser3_scores sorted
laser3_sorted_baseline_df= df.sort_values("laser3_scores", ascending=False)
laser3_sorted_baseline_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_baseline_csv_file), index=False)
xlmr_sorted_baseline_df= df.sort_values("xlmr_scores", ascending=False)
xlmr_sorted_baseline_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_baseline_csv_file), index=False)
labse_sorted_baseline_df= df.sort_values("labse_scores", ascending=False)
labse_sorted_baseline_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_baseline_csv_file), index=False)

endTime = time.time()
print('Time taken to complete Script: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")