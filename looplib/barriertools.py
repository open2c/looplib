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


def find_convergent_pairs(ctcf_right, ctcf_left, elements_between_pairs):
    """
    Finds pairs of convergent CTCF binding sites with exactly `elements_between_pairs` barrier elements between them.

    Parameters:
    ----------
    ctcf_right : list of int
        List of positions for CTCF binding sites on the right (right-facing CTCF sites).
    ctcf_left : list of int
        List of positions for CTCF binding sites on the left (left-facing CTCF sites).
    elements_between_pairs : int
        The number of barrier elements between the convergent CTCF pairs. 
        For example, `elements_between_pairs=1` finds directly connected CTCF sites, 
        `elements_between_pairs=2` finds CTCF sites with one intervening barrier element between them, etc.

    Returns:
    -------
    list of list of int
        A list of pairs of convergent CTCF sites, where each pair consists of one 
        left-facing CTCF site and one right-facing CTCF site, separated by exactly `elements_between_pairs` barriers.
    """
    # Combine and sort CTCF positions from both directions
    ctot = np.sort(np.array(ctcf_right + ctcf_left))
    
    # Find pairs of convergent CTCF elements with exactly `elements_between_pairs` barriers between them
    convergent_pairs = [
        [ctot[i], ctot[i + elements_between_pairs]] 
        for i in range(len(ctot) - elements_between_pairs) 
        if ctot[i] in ctcf_left and ctot[i + elements_between_pairs] in ctcf_right
    ]
    
    return convergent_pairs