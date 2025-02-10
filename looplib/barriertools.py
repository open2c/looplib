def peak_positions(boundary_list, window_sizes=[1]):
    """
    Calculate peak positions based on a boundary_list within window_sizes.

    Args:
        boundary_list (list): List of boundary values.
        window_sizes (list, optional): List of window sizes. Defaults to [1].

    Returns:
        np.ndarray: Array containing peak positions.
    """
    peak_monomers = np.array([])

    for i in window_sizes:
        inds_to_add = [boundary + i for boundary in boundary_list]
        peak_monomers = np.hstack((peak_monomers, inds_to_add))

    return peak_monomers.astype(int)


def FRiP(num_sites_t, lef_positions, peak_positions):
    """
    Calculate the Fraction of Reads in Peaks (FRiP) score.

    The FRiP score is a measure of how many loop-extruding factor (LEF) positions 
    fall within predefined peak regions, relative to the total number of LEF positions.

    Parameters:
    -----------
    num_sites_t : int
        Total number of genomic sites.
    lef_positions : array-like
        Positions of loop extruding factors (LEFs) along the genome.
    peak_positions : array-like
        Indices corresponding to peak regions (CTCFs).

    Returns:
    --------
    float
        The fraction of LEF positions that fall within peak regions.
    """
    hist, edges = np.histogram(lef_positions, np.arange(num_sites_t + 1))
    return np.sum(hist[peak_positions]) / len(lef_positions)