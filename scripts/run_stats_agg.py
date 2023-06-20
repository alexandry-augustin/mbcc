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
#----------------------------------------------------------
def lsd(path):
	""" list directories """
	return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
#----------------------------------------------------------
def lsf(path):
	""" list files """
	return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
#----------------------------------------------------------
def run_agg(regex, out_filename):
	run_paths=lsd("./")
	data=None
	for run in run_paths:
		iter=int(run.replace("Run", ""))

		path=os.path.join(os.getcwd(), run)

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
			
	data.to_csv("../%s"%out_filename, index=False, sep=",")
#----------------------------------------------------------
if __name__=='__main__':
	path=os.path.dirname("../bin/Release/")
	path=os.path.join(path, "stats")
	os.chdir(path)
	#--------------------------------------------------
	#postfix="TrueLabelDistribution"
	#postfix="TrueLabel"
	postfix="ConfusionMatrixPosterior"
	regex=re.compile(".*%s\.csv$"%postfix)
	if len(sys.argv)==2:
		out_filename=sys.argv[1]
	else:
		CURR_TIME=time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime(int(time.time())))
		out_filename="out_stats_%s_%s.csv"%(postfix, CURR_TIME)
	print "Output: %s"%out_filename
	run_agg(regex=regex, out_filename=out_filename)
