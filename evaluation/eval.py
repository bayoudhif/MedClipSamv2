import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth masks')
parser.add_argument('--seg_path', type=str, required=True, help='Path to predicted masks')
parser.add_argument('--output', type=str, default='results.json', help='Path to save JSON results')

args = parser.parse_args()

# Initialize metrics dictionary
seg_metrics = OrderedDict(
    Name=[],
    DSC=[],
    NSD=[],
)

# Compute metrics for each file
filenames = [f for f in os.listdir(args.seg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
filenames.sort()

for name in tqdm(filenames, desc="Evaluating masks"):
    try:
        seg_metrics['Name'].append(name)
        
        # Load and process masks
        gt_mask = cv2.imread(join(args.gt_path, name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.imread(join(args.seg_path, name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize masks
        gt_mask = cv2.threshold(gt_mask, 200, 255, cv2.THRESH_BINARY)[1]
        seg_mask = cv2.threshold(seg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        gt_data = np.uint8(gt_mask)
        seg_data = np.uint8(seg_mask)

        # Calculate metrics
        labels = np.union1d(np.unique(gt_data)[1:], np.unique(seg_data)[1:])
        assert len(labels) > 0, f'No labels found in {name}'

        DSC_arr, NSD_arr = [], []
        for i in labels:
            i_gt, i_seg = (gt_data == i), (seg_data == i)
            
            # Handle edge cases
            if np.sum(i_gt) == 0 and np.sum(i_seg) == 0:
                DSC_arr.append(1.0)
                NSD_arr.append(1.0)
                continue
                
            if np.sum(i_gt) == 0 or np.sum(i_seg) == 0:
                DSC_arr.append(0.0)
                NSD_arr.append(0.0)
                continue

            # Compute metrics
            DSC_arr.append(compute_dice_coefficient(i_gt, i_seg))
            surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], [1, 1, 1])
            NSD_arr.append(compute_surface_dice_at_tolerance(surface_distances, 2))

        seg_metrics['DSC'].append(float(np.mean(DSC_arr)))
        seg_metrics['NSD'].append(float(np.mean(NSD_arr)))

    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        seg_metrics['DSC'].append(np.nan)
        seg_metrics['NSD'].append(np.nan)

# Save results to JSON
results = {
    'per_image': seg_metrics,
    'average': {
        'DSC': float(np.nanmean(seg_metrics['DSC'])),
        'NSD': float(np.nanmean(seg_metrics['NSD']))
    }
}

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAverage DSC: {results['average']['DSC']:.4f}")
print(f"Average NSD: {results['average']['NSD']:.4f}")