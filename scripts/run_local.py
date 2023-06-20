#!/usr/bin/env python 
#----------------------------------------------------------
import os
import sys
import subprocess
import time
import numpy as np
import shlex
from collections import deque
#----------------------------------------------------------
def build_cmd_str(
    filename, 
    START_CLUSTER_RUN, 
    END_CLUSTER_RUN, 
    RS, 
    SM, 
    NBITER):
	return "mono %s %s %s %s %s %s"%(
		filename, 
        START_CLUSTER_RUN, 
        END_CLUSTER_RUN, 
        RS, 
        SM, 
        NBITER)
#----------------------------------------------------------
if __name__=="__main__":
	
	error=os.system("sh ./build_release.sh")
	if error:
		sys.exit(1)

	currdir=os.path.dirname(os.path.realpath(__file__))
	#os.getcwd() #current working directory

	path="../bin/Release/"
	os.chdir(path)
	filename="infer.net.exe"

	IS_PARALLEL=False
	NB_CORES=2 #CPU cores

	START_CLUSTER_RUN=1
	END_CLUSTER_RUN=1
	RS=1 #ratio spammers
	SM=180 #nb samples
	NBITER=40	#nb iterations of the inference engine
	#---------------------------------------------
	# Build command queue
	#---------------------------------------------
	cmd=deque()

	for rs in np.linspace(0, 5, 10+1, dtype=float):	#ratio of spammers
		for cluster_run in range(START_CLUSTER_RUN, END_CLUSTER_RUN+1):
			cmd.append(build_cmd_str(
				filename, 
		        cluster_run, 
                cluster_run, 
                rs, 
                SM, 
                NBITER))
	#nb samples
#	for sm in np.linspace(6, 204, 10+1, dtype=int):
#	#for sm in np.linspace(6, 204, 10):
#	#	sm=int(sm)
#		for cluster_run in range(START_CLUSTER_RUN, END_CLUSTER_RUN+1):
#			cmd.append(build_cmd_str(filename, cluster_run, cluster_run, RS, sm, NBITER))
	#---------------------------------------------
	# Run command queue
	#---------------------------------------------
	start_time=time.time()

	while len(cmd)>0:
		if IS_PARALLEL:
			proc=list()
			for i in range(NB_CORES):
				c=cmd.pop()
				print "Running: %s"%c
				print "%s remaining tasks."%len(cmd)
				#proc.append(subprocess.Popen(["mono", filename, "%s"%START_CLUSTER_RUN, "%s"%END_CLUSTER_RUN, "%s"%RS, "%s"%SM]))
				args=shlex.split(c)
				proc.append(subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
				assert(len(proc)<=NB_CORES)
	#		exit_codes=[p.wait() for p in proc]
			for p in proc:
				exit_code=p.wait()
				out, err=p.communicate()
				if out:
					pass
					#stderr=c.replace(" ", "_")
					#save to file
				if err:
					pass
		else:
			c=cmd.pop()
			print "Running: %s"%c
			print "%s remaining tasks."%len(cmd)
			error=os.system(c)
			#subprocess.call(c)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)




#	cmd=list()
##	for RS in np.arange(0, .45, .05): #ratio spammers
#	for SM in np.arange(1, 50): #sample multiplier -- 15 7
#		cmd.append("mono %s %s %s %s %s"%(filename, START_CLUSTER_RUN, END_CLUSTER_RUN, RS, SM))
#	
#	start_time=time.time()

#	proc=list()
#	for c in cmd:
#		print "Running: %s"%c
#
#		if not IS_PARALLEL:
#			error=os.system(c)
#		else:
#			args=shlex.split(c)
#			proc.append(subprocess.Popen(args))
#	exit_codes=[p.wait() for p in proc]
	#---------------------------------------------
	# Print execution time
	#---------------------------------------------
	duration=time.time()-start_time
	m, s=divmod(duration, 60)
	h, m=divmod(m, 60)
	print("------------------------------")
	print("Script %s terminated."%os.path.basename(__file__))
	print("Total duration: %d:%02d:%02d"%(h, m, s))
