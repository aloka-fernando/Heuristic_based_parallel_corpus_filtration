# Created by  aloka on 23/03/2024

#imports
import time
import os
import pandas as pd



startTime=time.time()

#inputs
data_dir="/userdirs/aloka/datasets/opus/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores.en-si.csv"
df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")

#outputs
corpus="ccmatrix"
exp_file_name="baseline"
src_lang="en"
tgt_lang="si"
sub_dir = "csv_files"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/{}_{}/".format(corpus, exp_file_name)
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory exists...")




laser3_baseline_csv_file="{}-{}-{}-laser3-sorted-{}.train.csv".format(src_lang, tgt_lang, corpus, exp_file_name)
xlmr_baseline_csv_file="{}-{}-{}-xlmr-sorted-{}.train.csv".format(src_lang, tgt_lang, corpus, exp_file_name)
labse_baseline_csv_file="{}-{}-{}-labse-sorted-{}.train.csv".format(src_lang, tgt_lang, corpus, exp_file_name)

#batch
dataset_size = len(df)
print("Dataset size : {}".format(dataset_size))


##############################################################
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