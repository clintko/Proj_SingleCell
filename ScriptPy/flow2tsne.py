print("Run python script flow2tsne.py")
print("===== Setup Environment ======")
import numpy as np
import numpy.random as rand
from collections import Counter
from scipy.interpolate import Rbf
from sklearn.manifold import TSNE

print("===== Import Data       ======")
# input the value
tmp_dir = "/data/deep-learning-group/test_data/flow_EQAPOL/"
fname_Costim = "data_Costim.np"
fname_CMV    = "data_CMV.np"
fname_SEB    = "data_SEB.np"

# open and write the file
print("Import Costim...")
file_object = open(tmp_dir + fname_Costim, 'rb')
data_Costim = np.load(file_object)
file_object.close()

print("Import CMV...")
file_object = open(tmp_dir + fname_CMV, 'rb')
data_CMV = np.load(file_object)
file_object.close()

print("Import SEB...")
file_object = open(tmp_dir + fname_SEB, 'rb')
data_SEB = np.load(file_object)
file_object.close()

print("Import markers...")
file_object = open(tmp_dir + "markers.np", 'rb')
markers = np.load(file_object)
markers_dict = {items[1]: idx for idx, items in enumerate(markers)}
file_object.close()

print("...The data are input.")

print("===== Arrange Data ======")
samples      = np.array(list(data_Costim) + list(data_CMV) + list(data_SEB))
label_groups = np.array(                [0] * len(list(data_Costim)) +      [1] * len(list(data_CMV)) +         [2] * len(list(data_SEB)))
    
print("check dimension")
print("++++++++++++++++")
print("Labels:  ", label_groups.shape)
print(Counter(label_groups))
print("++++++++++++++++")
print("Samples: ", samples.shape)
print("Samples: ", samples[0].shape)
print("Samples: ", samples[1].shape)
print("++++++++++++++++")
print("Costim: ", data_Costim.shape)
print("Costim: ", data_Costim[0].shape)
print("Costim: ", data_Costim[1].shape)
print("++++++++++++++++")
print("CMV:    ", data_CMV.shape)
print("CMV:    ", data_CMV[0].shape)
print("CMV:    ", data_CMV[1].shape)
print("++++++++++++++++")
print("SEB:    ", data_SEB.shape)
print("SEB:    ", data_SEB[0].shape)
print("SEB:    ", data_SEB[1].shape)

##############################################################################

print("===== Declare functions ======")
def check_dimension(samples):
    """ Check the dimension of an numpy array, the function 
    allows different number of the second dimention
    
    For example:
        len(shapes.shape) = 2 && shapes.shape[1] = 2
        => result: 2 - 1 + 2 = 3
    
    >>> import numpy as np
    >>> arr = np.array([[[1], [2]], [[3], [4]]])
    >>> arr.shape
    (2, 2, 1)
    >>> check_dimension(arr)
    (3, 'Wrong input dimension; Expected 3 but 3 given; the samples should contain (samples, events, markers)')
    
    >>> x = np.array([[11, 12, 13], [14, 15, 16]])
    >>> y = np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]])
    >>> z = np.array([[31, 32, 33]])
    >>> arr = np.array([x, y, z])
    >>> check_dimension(arr)
    (3, 'Wrong input dimension; Expected 3 but 3 given; the samples should contain (samples, events, markers)')
    """
    shapes = np.array([sample.shape for sample in samples])
    checked_value = len(shapes.shape) - 1 + shapes.shape[1] 
    
    error_message =         "Wrong input dimension; Expected 3 but " + str(checked_value) +         " given; the samples should contain (samples, events, markers)"
        
    return checked_value, error_message

################################################################################

def data_standardization(samples, idx_sample = 0, verbose = False):
    """standardized features of all samples using one group of samples
    
    Position arguments:
    samples    --- numpy array; (num_samples_tot, num_cells_tot, num_features)
                   note: the num_cells_tot is not fixed for each samples
    idx_sample --- int; which sample is used for standardization; default to 0
    
    Keyword arguments:
    verbose --- print out the running process
    """
    
    if (verbose):
        print("Data Standardization...")
    
    # num_samples_tot, num_cells_tot, num_features
    # => correct value: len(shapes.shape) = 2 && shapes.shape[1] = 2
    checked_value, error_message = check_dimension(samples)
    assert (checked_value == 3), error_message
    assert samples.shape[0] > idx_sample, "Incorrect input of idx_sample"
    
    # calculate mu and sd
    res = samples[idx_sample] 
    mu  = np.mean(res, axis=0)
    sd  = np.std( res, axis=0)
    
    # standardize
    samples_stdard = np.array( [(sample - mu) / sd for sample in samples] )    
    
    if (verbose):
        print("...Finish")
        
    return(samples_stdard)

##############################################################################

def data_subsetting(samples, label_groups, k = 1000, num_subsets = 10, rand_seed = 0, verbose = False):
    """create subsets for each sample
    
    Position arguments:
    samples      --- numpy array; (num_samples_tot, num_cells_tot, num_features)
    label_groups --- list or numpy array; label of each sample in samples
    
    Keyword arguments:
    k           --- number of cells get from the each sample
    num_subsets --- number of subsets; each subset contain k cells
    rand_seed   --- random seed
    verbose     --- print out the running process
    """
    
    if (verbose):
        print("Data Subsetting...")
        
    # check the dimension
    checked_value, error_message = check_dimension(samples)
    assert (checked_value == 3), error_message
    
    # initialization
    #num_samples_tot, num_cells_tot, num_features = samples.shape
    num_samples_tot = samples.shape[0]
    rand.seed(rand_seed)
    result_samples = []
    result_labels  = []
    
    # iterate through all samples
    for idx_sample in range(num_samples_tot):
        
        # initialization in each loop
        sample = samples[idx_sample]
        num_cells_tot = sample.shape[0]
        num_features  = sample.shape[1]
        
        # record the corresponding label
        group  = label_groups[idx_sample]
        result_labels += ([group] * num_subsets)
        
        # generate subsets in each sample
        for _ in range(num_subsets):
            
            # choose k cells randomly
            idx = rand.permutation(num_cells_tot)[:k]
            result_samples.append(sample[idx])
              
    # convert results from list to numpy array
    result_samples = np.array(result_samples) # (num_samples_tot * num_subsets, k, num_genes)
    result_labels  = np.array(result_labels)  # (num_samples_tot * num_subsets,)
    
    if (verbose):
        print("...Finish")
    
    return result_labels, result_samples

##############################################################################

def create_tsne(samples, verbose = False,
                tsne_dimension  = 2,
                tsne_perplexity = 40, 
                tsne_iter       = 300, 
                tsne_verbose    = 0,
                tsne_rand_seed  = 0):
    """create t-SNE plot for each sample
    
    Position arguments:
    samples --- numpy array; (num_samples_tot, num_cells_tot, num_features)
    
    Keyword arguments:
    rand_seed --- random seed
    verbose   --- print out the running process
    """
    
    if (verbose):
        print("Create t-SNE plots...")
        
    # check the dimension
    checked_value, error_message = check_dimension(samples)
    assert (checked_value == 3), error_message
    
    # initialization
    num_samples_tot = samples.shape[0]
    result_tsne = []
    
    # generate tsne plot for each sample
    for idx_sample in range(num_samples_tot):
        if (verbose):
            print("\tPrepare t-SNE plot of the", idx_sample, "sample")
        
        # initialization in each loop
        sample = samples[idx_sample]
        num_cells_tot = sample.shape[0]
        num_features  = sample.shape[1]
        
        # for each sample, generate a t-SNE plot
        tsne = TSNE(n_components = tsne_dimension, 
                    verbose      = tsne_verbose, 
                    perplexity   = tsne_perplexity, 
                    n_iter       = tsne_iter, 
                    random_state = tsne_rand_seed)
        res = tsne.fit_transform(sample)
        result_tsne.append(res)
    
    # convert the result from list to numpy array
    result_tsne = np.array(result_tsne)
    
    if (verbose):
        print("...Finish")
        
    return result_tsne

##############################################################################

def create_img(tsne_plots, samples, n_grid = 128, 
               verbose = False, 
               verbose_sample = False, 
               verbose_marker = False):
    """create image from t-SNE plot
    
    Position arguments:
    tsne_plots --- numpy array; (num_samples_tot, num_cells_tot, 2)
    samples    --- numpy array; (num_samples_tot, num_cells_tot, num_features)
    
    Keyword arguments:
    n_grid  --- the dimension of image (n_grid x n_grid)
    verbose --- print out the running process
    """
    
    if (verbose):
        print("Create images from t-SNE plot...")
    
    # check the dimension
    checked_value, error_message = check_dimension(samples)
    assert (checked_value == 3), error_message
    
    # initialization
    num_samples_tot = samples.shape[0]
    result_img = []
    
    # iterate though each samples
    for idx_sample in range(num_samples_tot):
        if (verbose_sample):
            print("\tPrepare image of the", idx_sample, "sample")
        
        # initialization in each loop
        sample = samples[idx_sample]
        num_cells_tot = sample.shape[0]
        num_features  = sample.shape[1]
        
        # get x, y coordinate of a plot
        tsne_plot = tsne_plots[idx_sample]
        x = tsne_plot[:, 0]
        y = tsne_plot[:, 1]
        
        # generate a grid
        x_c = np.linspace(min(x), max(x), n_grid)
        y_c = np.linspace(min(y), max(y), n_grid)
        x_c, y_c = np.meshgrid(x_c, y_c)
        
        # each feature is a layer/channel for the plot
        # to get each layer, perform interpolation to convert tSNE plot in a image
        img = []
        for idx_feature in range(num_features):
            
            if (verbose_marker):
                print("\t\tinterpolating the", idx_feature, "feature")
            
            # interpolation
            z = sample[:, idx_feature]
            rbfi = Rbf(x, y, z, function='multiquadric', smooth=1)
            
            # store into a list "img"
            z_c = rbfi(x_c, y_c)
            img.append(z_c)
            
        # normalize & arrange the interpolated feature values    
        img = np.array(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # append each interpolated result to the result
        result_img.append(img)
        
    # standardize images
    result_img = [(img - np.min(img)) / (np.max(img) - np.min(img)) for img in result_img]
    result_img = np.array(result_img)
    
    if (verbose):
        print("...Finish")
        
    return result_img

##############################################################################

print("===== Data Preprocess   =====")
tmp = samples
samples_std = data_standardization(tmp)

print("===== 10000  (E4) cells ======")
data_labels_E4sub10, data_samples_E4sub10 = data_subsetting(samples_std, label_groups, k = 10000, num_subsets = 10, verbose = True)
data_tsne_E4sub10 = create_tsne(data_samples_E4sub10, verbose = True)

print("===== Store the results ======")
# output the value
tmp_dir = "/data/deep-learning-group/test_data/flow_EQAPOL/"
fname = "tsne_E4sub10.npz"

# open the file for writing
file_object = open(tmp_dir + fname, 'wb') # wb --- write binary

# write data to the file
np.savez(
    file_object, 
    data_samples = data_samples_E4sub10, 
    data_labels  = data_labels_E4sub10, 
    data_tsne    = data_tsne_E4sub10)

# close the file
file_object.close()
print("The results are stored. They are stored in " + tmp_dir + fname)

