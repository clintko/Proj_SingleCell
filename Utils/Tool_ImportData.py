import numpy as np

######################################################

PATH_EQAPOL = "/data/clintko/flow_EQAPOL/"
PATH_EQAPOL_TMP = "/data/deep-learning-group/test_data/flow_EQAPOL/"

COSTIM  = "Costim"
CMV     = "CMV"
SEB     = "SEB"
MARKERS = "Markers"

FNAME_COSTIM  = "data_Costim.np"
FNAME_CMV     = "data_CMV.np"
FNAME_SEB     = "data_SEB.np"
FNAME_MARKERS = "markers.np"

######################################################

def read_EQAPOL(fnames = None, data_dir = PATH_EQAPOL_TMP):
    """read in the EQAPOL data"""
    # initialization
    data = dict()
    flag = False
    
    if fnames is None:
        print("Nothing Import")
        return data
    
    if COSTIM in fnames:
        print("Read Costim")
        file_object = open(data_dir + FNAME_COSTIM, 'rb')
        data["Costim"] = np.load(file_object)
        file_object.close()
        flag = True
        
        
    if CMV in fnames:
        print("Read CMV")
        file_object = open(data_dir + FNAME_CMV, 'rb')
        data["CMV"] = np.load(file_object)
        file_object.close()
        flag = True
        
    if SEB in fnames:
        print("Read SEB")
        file_object = open(data_dir + FNAME_SEB, 'rb')
        data["SEB"] = np.load(file_object)
        file_object.close()
        flag = True

    if MARKERS in fnames:
        print("Read Markers")
        file_object = open(data_dir + FNAME_MARKERS, 'rb')
        markers = np.load(file_object)
        data["Markers"] = {items[1]: idx for idx, items in enumerate(markers)}
        file_object.close()
        flag = True
        
    if flag == False:
        print("Input fname is not found.")
    else:
        print("The data " + " ".join(fnames) + " are input.")
        
    return data