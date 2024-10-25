# Created by  aloka on 23/03/2024

#imports
import torch
import time
import pandas as pd
from datasets import load_dataset
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util


startTime=time.time()
#inputs
data_dir="/userdirs/aloka/datasets/opus/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores.en-si.csv"
#input_file= open("{}/{}".format(data_dir, input_file), "r", encoding="utf8")

#outputs
corpus="CCMatrix"
src_lang="en"
tgt_lang="si"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_slength"

#output files
sub_dir = "csv_files"
laser3_sorted_src_slength_csv_file="{}-{}-laser3-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang)
xlmr_sorted_src_slength_csv_file="{}-{}-xlmr-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang)
labse_sorted_src_slength_csv_file="{}-{}-labse-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang)

laser3_sorted_tgt_slength_csv_file="{}-{}-laser3-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, tgt_lang)
xlmr_sorted_tgt_slength_csv_file="{}-{}-xlmr-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, tgt_lang)
labse_sorted_tgt_slength_csv_file="{}-{}-labse-sorted-{}-slength.train.csv".format(src_lang, tgt_lang, tgt_lang)

laser3_sorted_src_tgt_slength_csv_file="{}-{}-laser3-sorted-{}-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)
xlmr_sorted_src_tgt_slength_csv_file="{}-{}-xlmr-sorted-{}-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)
labse_sorted_src_tgt_slength_csv_file="{}-{}-labse-sorted-{}-{}-slength.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang)


###############################################################
#main logic
###############################################################

#read data
org_dataset_df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print('Dataset size original : {}'.format(len(org_dataset_df)))

#slength filtration
src_slenth_df = org_dataset_df.loc[org_dataset_df['src_sents'].str.split().str.len() >= 5]
print('Dataset size after src slength >5 filtering : {}'.format(len(src_slenth_df)))
# short_src_slenth_df = org_dataset_df.loc[org_dataset_df['src_sents'].str.split().str.len() < 5]
# print('Dataset size after src slength < 5 filtering : {}'.format(len(short_src_slenth_df)))
tgt_slenth_df = org_dataset_df.loc[org_dataset_df['tgt_sents'].str.split().str.len() > 5]
print('Dataset size after tgt slength >5 filtering : {}'.format(len(tgt_slenth_df)))
src_tgt_slenth_df = org_dataset_df.loc[(org_dataset_df['src_sents'].str.split().str.len() > 5) & (org_dataset_df['tgt_sents'].str.split().str.len() > 5)]
print('Dataset size after src & tgt slength >5 filtering : {}'.format(len(src_tgt_slenth_df)))

endTime = time.time()
print('Time taken to complete slength filtration : {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print('----------------------------------------------------------------------------------------------------------------')


#laser3_scores sorted
laser3_src_slength_sorted_df= src_slenth_df.sort_values("laser3_scores", ascending=False)
laser3_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_slength_csv_file), index=False)
laser3_tgt_slength_sorted_df= tgt_slenth_df.sort_values("laser3_scores", ascending=False)
laser3_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_tgt_slength_csv_file), index=False)
laser3_src_tgt_slength_sorted_df= src_tgt_slenth_df.sort_values("laser3_scores", ascending=False)
laser3_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_tgt_slength_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_slength_sorted_df= src_slenth_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_slength_csv_file), index=False)
xlmr_tgt_slength_sorted_df= tgt_slenth_df.sort_values("xlmr_scores", ascending=False)
xlmr_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_tgt_slength_csv_file), index=False)
xlmr_src_tgt_slength_sorted_df= src_tgt_slenth_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_tgt_slength_csv_file), index=False)

#labse_scores sorted
labse_src_slength_sorted_df= src_slenth_df.sort_values("labse_scores", ascending=False)
labse_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_slength_csv_file), index=False)
labse_tgt_slength_sorted_df= tgt_slenth_df.sort_values("labse_scores", ascending=False)
labse_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_tgt_slength_csv_file), index=False)
labse_src_tgt_slength_sorted_df= src_tgt_slenth_df.sort_values("labse_scores", ascending=False)
labse_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_tgt_slength_csv_file), index=False)

#write short sentences
# laser3_short_src_slength_sorted_df= short_src_slenth_df.sort_values("laser3_scores", ascending=False)
# laser3_short_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, "short_src_slength_filtered.en-si.csv"), index=False)

endTime = time.time()
print('Time taken to complete slength filtration: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print('----------------------------------------------------------------------------------------------------------------\n\n')
print("Script Completed.....")