PUT THIS IN FairShip/python/detectArch
#!/usr/bin/env /software/miniconda/bin/python
import os,sys
sys.path.append(os.environ['SHIPBUILD']+'/alibuild')
from alibuild_helpers.utilities import getVersion, detectArch
print detectArch()
