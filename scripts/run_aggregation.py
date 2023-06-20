#!/usr/bin/env python 
#----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import re
import itertools
import time
#import sklearn.metrics
#----------------------------------------------------------
def lsd(path):
	""" list directories """
	return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
#----------------------------------------------------------
def lsf(path):
	""" list files """
	return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
#----------------------------------------------------------
def parse(filename):
	""" parse filename """
#	filename=.split("_")
#	#"ApprovalVoting_accuracy_sr05_ns4_final"
#	model_name=filename[0]
#	spammer_ratio=filename[2]
#	ns=filename[3]
#----------------------------------------------------------
def run_agg(regex, out_filename):
	run_paths=lsd("./")
	data=None
	for run in run_paths:
		iter=int(run.replace("Run", ""))

		path=os.path.join(os.getcwd(), run)
#		os.chdir(path)

		files=lsf(path)
		for f in files:
			if not regex.search(f):
				continue

			full_path=os.path.join(path, f)
			temp_df=pd.read_csv(full_path, sep=",")
			temp_df["NumIter"]=iter #add iteration field
			if data is None:
				data=temp_df
			else:
				data=data.append(temp_df)

#	data.replace("", "NA", inplace=True)

	os.chdir("../")
	data.to_csv("%s"%out_filename, index=False, sep=",")
#----------------------------------------------------------
if __name__=='__main__':
	
	if len(sys.argv)<4:
		print "Usage: run_aggregation.py input_dir filter output_filename_prefix"
		print "Example: run_aggregation.py /tmp/ResultsMBCC \"\" \"\""
		sys.exit(1)
	
	path=sys.argv[1]
	filter=sys.argv[2]
	prefix=sys.argv[3]
	#--------------------------------------------------
	#path=os.path.dirname(os.path.realpath(__file__))
	#path=os.path.dirname("../bin/Release/")
	#path=os.path.join(path, directory)
	print("Path: %s"%path)
	#--------------------------------------------------
	os.chdir(path)
	#--------------------------------------------------
	CURR_TIME=time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime(int(time.time())))
	#--------------------------------------------------
	regex=re.compile(".*%s.*\.csv$"%filter)
	out_filename="%s/out_%s_%s.csv"%(path, prefix, CURR_TIME)
	run_agg(regex=regex, out_filename=out_filename)
	print("Output: %s"%out_filename)
	#--------------------------------------------------
#	bcc=find(path, re.compile("^BCC.*\.csv$"))
#	print "\n".join(bcc)
