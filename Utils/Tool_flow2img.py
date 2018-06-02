import numpy as np
import numpy.random as rand
import umap

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
    
    # calculate mu and sd
    res = samples[idx_sample] 
    #mu  = np.mean(res, axis=0)
    #sd  = np.std( res, axis=0)
    X_col_max = np.max(X, axis = 0)
    X_col_min = np.min(X, axis = 0)
    
    # standardize
    #samples_stdard = np.array( [(sample - mu) / sd for sample in samples] )    
    samples_stdard = np.array([(X - X_col_min) / (X_col_max - X_col_min) for sample in samples])
    
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
    
    # check 
    assert (samples.shape[0] == len(label_groups)), "The dimension of samples and labels are not consistent."
    
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
#umap.UMAP(
#    n_neighbors=15, n_components=2, metric='euclidean',  n_epochs=None, 
#    alpha=1.0, init='spectral', spread=1.0, min_dist=0.1, 
#    set_op_mix_ratio=1.0,local_connectivity=1.0, 
#    bandwidth=1.0, gamma=1.0, negative_sample_rate=5, 
#    a=None, b=None, random_state=None, metric_kwds={}, 
#    angular_rp_forest=False, verbose=False)

def create_umap(samples, verbose = False,
                dimension  = 2,
                perplexity = 40, 
                num_iter      = 300, 
                inner_verbose = 0,
                rand_seed  = 0):
    """create UMAP plot for each sample
    
    Position arguments:
    samples --- numpy array; (num_samples_tot, num_cells_tot, num_features)
    
    Keyword arguments:
    rand_seed --- random seed
    verbose   --- print out the running process
    """
    
    if (verbose):
        print("Create UMAP plots...")
        
    # initialization
    num_samples_tot = samples.shape[0]
    results = []
    
    # generate umap plot for each sample
    for idx_sample in range(num_samples_tot):
        if (verbose):
            print("\tPrepare UMAP plot of the", idx_sample, "sample")
        
        # initialization in each loop
        sample = samples[idx_sample]
        num_cells_tot = sample.shape[0]
        num_features  = sample.shape[1]
        
        # for each sample, generate a UMAP plot
        #tsne = TSNE(n_components = tsne_dimension, 
        #            verbose      = tsne_verbose, 
        #            perplexity   = tsne_perplexity, 
        #            n_iter       = tsne_iter, 
        #            random_state = tsne_rand_seed)
        #res = tsne.fit_transform(sample)
        res = umap.UMAP(random_state = rand_seed).fit_transform(sample)
        results.append(res)
    
    # convert the result from list to numpy array
    results = np.array(results)
    
    if (verbose):
        print("...Finish")
        
    return results

def create_umap_pool(samples, verbose = False, rand_seed = 0):
    if (verbose):
        print("Create UMAP plots...")
        
    tmp = np.vstack(samples)
    tmp = umap.UMAP(random_state = rand_seed).fit_transform(tmp)
    
    if (verbose):
        print("...Finish")
        
    return tmp