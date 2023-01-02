import pyBigWig
import pandas as pd
from pybedtools import BedTool
import numpy as np
import math
#First make a balanced bedfile for the positive and negative training class.
#Positive = center basepair of window overlaps with Unibind database.
#Negative = center basepair of window does not overlap with Unibind database.

#Get x random positions from ATAC-peaks.
#in BASH, using bedtools package:
#   for i in {1..500000}; do echo -e "chr1\t1\t2" >> 500000_positions.bed; done
#   bedtools shuffle -i 500000_positions.bed -incl ENCFF913MQB.bed -g chrom.sizes > 500k_random_ATAC.bed

#Split random positions into positive and negative class using bedtools intersect.
positions = pd.read_csv('500k_random_ATAC.bed', sep='\t',header=None)
unibind = pd.read_csv('mergedhepG2_ATAC_peaks.bed', sep='\t',header=None)
a = BedTool('500k_random_ATAC.bed')
b = BedTool('mergedhepG2_ATAC_peaks.bed') #all Unibind entries for this celltype (HEPG2)
a_intersect = a.intersect(b,wa=True).moveto('positive_class.bed')
b_intersect = a.intersect(b,v=True).moveto('negative_class.bed')

#Make the dataset balanced. As many positions in the positive as in the negative class
unibind_positions = pd.read_csv('positive_class.bed', sep='\t',header=None)
no_unibind_positions = pd.read_csv('negative_class.bed', sep='\t',header=None)

if len(unibind_positions) < len(no_unibind_positions):
    drop_indices = np.random.choice(no_unibind_positions.index, (len(no_unibind_positions) - len(unibind_positions)), replace=False)
    no_unibind_positions = no_unibind_positions.drop(drop_indices)
    no_unibind_positions = no_unibind_positions.reset_index(drop=True)
elif len(no_unibind_positions) < len(unibind_positions):
    drop_indices = np.random.choice(unibind_positions.index, (len(unibind_positions) - len(no_unibind_positions)), replace=False)
    unibind_positions = unibind_positions.drop(drop_indices)
    unibind_positions = unibind_positions.reset_index(drop=True)

#Make windows of size 500 out of the random positions

windowsize = 500
unibind_positions[1] = (unibind_positions[1] - np.floor(windowsize/2)).astype(int)
unibind_positions[2] = (unibind_positions[2] + np.ceil(windowsize/2)).astype(int)
unibind_positions.to_csv("500k_random_windows_unibind_100bp.bed", sep='\t', header=False, index=False)

no_unibind_positions[1] = (no_unibind_positions[1] - np.floor(windowsize/2)).astype(int)
no_unibind_positions[2] = (no_unibind_positions[2] + np.ceil(windowsize/2)).astype(int)
no_unibind_positions.to_csv("500k_random_windows_no_unibind_100bp.bed", sep='\t', header=False, index=False)

#append data to a befile to create a file used for the training of classifiers. extracts data from bigiwig file to a bedfile

#BigWig files containing per position data
bw = pyBigWig.open("ENCFfootprints.bw")
bindetect = pyBigWig.open("ENCFBindetect.bw")
shap = pyBigWig.open("ENCFshap.bw")
#Bed file containing regions of interest
profile_bed = BedTool("Region_of_interest.bed")
all_scores = []
for interval in profile_bed:
    scores = bw.values(interval.chrom, interval.start, interval.stop)
    scores = [0 if math.isnan(x) else x for x in scores]
    scoresbind = bindetect.values(interval.chrom, interval.start, interval.stop)
    scoresbind = [0 if math.isnan(x) else x for x in scoresbind]
    scoresshap = shap.values(interval.chrom, interval.start, interval.stop)
    scoresshap = [0 if math.isnan(x) else x for x in scoresshap]
    all_scores.append([interval.chrom, interval.start, interval.stop, scores,scoresbind,scoresshap])

all_scores = pd.DataFrame(all_scores)
all_scores.to_csv("Region_of_interest.bed", sep='\t', header=False, index=False)
