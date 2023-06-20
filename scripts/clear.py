#!/usr/bin/env python 
#----------------------------------------------------------
import os
#----------------------------------------------------------
if __name__=="__main__":
	
	path="../bin/Release/"
	dirname="ResultsMIBCC"
	os.chdir(path)
	error=os.system("rm -r ./%s"%dirname)
	print "%s%s deleted"%(path, dirname)
