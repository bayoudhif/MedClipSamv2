import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import os
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm

np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess_crf(args):
    files = os.listdir(args.sal_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for file in tqdm(files):
        img = cv2.imread(os.path.join(args.input_path, file), 1)
        annos = cv2.imread(os.path.join(args.sal_path, file), 0)
        annos = cv2.resize(annos, (img.shape[1], img.shape[0]))
        output = os.path.join(args.output_path, file)

        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], args.m)

        anno_norm = annos / 255.
        n_energy = -np.log((1.0 - anno_norm + args.epsilon)) / (args.tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + args.epsilon) / (args.tau * sigmoid(anno_norm))

        U = np.zeros((args.m, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=args.gaussian_sxy, compat=3)
        d.addPairwiseBilateral(sxy=args.bilateral_sxy, srgb=args.bilateral_srgb, rgbim=img, compat=5)

        # Inference
        Q = d.inference(1)
        map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

        # Save output
        segmented_image = map.astype('uint8') * 255

        # Connected Component Analysis (same as before)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        sorted_sizes = sorted(sizes[1:], reverse=True)
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        cv2.imwrite(output, im_result)

# ========== Modified Function: Otsu's Thresholding ========== #
def postprocess_thresholding(args):
    files = os.listdir(args.sal_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for file in tqdm(files):
        # Read saliency map (0-255)
        annos = cv2.imread(os.path.join(args.sal_path, file), 0)
        output = os.path.join(args.output_path, file)

        # Apply Otsu's Thresholding (automatically computes optimal threshold)
        _, otsu_mask = cv2.threshold(
            src=annos.astype(np.uint8), 
            thresh=0, 
            maxval=255, 
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Otsu's method
        )

        # Connected Component Analysis (keep top-K largest regions)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(otsu_mask)
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        sorted_sizes = sorted(sizes[1:], reverse=True)
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        cv2.imwrite(output, im_result)

def postprocess_kmeans(args):
    files = os.listdir(args.sal_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for file in tqdm(files):
        kmeans = KMeans(n_clusters=2, random_state=10)
        attn_weights = cv2.imread(os.path.join(args.sal_path, file), 0) / 255
        h, w = attn_weights.shape
        image = cv2.resize(attn_weights, (256, 256), interpolation=cv2.INTER_NEAREST)
        flat_image = image.reshape(-1, 1)

        labels = kmeans.fit_predict(flat_image)
        segmented_image = labels.reshape(256, 256)

        centroids = kmeans.cluster_centers_.flatten()
        background_cluster = np.argmin(centroids)
        segmented_image = np.where(segmented_image == background_cluster, 0, 1)

        segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_NEAREST)
        segmented_image = segmented_image.astype(np.uint8) * 255

        # Connected Component Analysis (same as before)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        sorted_sizes = sorted(sizes[1:], reverse=True)
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        cv2.imwrite(os.path.join(args.output_path, file), im_result)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian-sxy', type=int, default=5, help="Gaussian sxy for CRF")
    parser.add_argument('--bilateral-sxy', type=int, default=5, help="Bilateral sxy for CRF")
    parser.add_argument('--bilateral-srgb', type=int, default=3, help="Bilateral srgb for CRF")
    parser.add_argument('--epsilon', type=float, default=1e-8, help="Epsilon for CRF")
    parser.add_argument('--m', type=int, default=2, help="Number of classes")
    parser.add_argument('--tau', type=float, default=1.05, help="Tau for CRF")
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help="[Ignored for Otsu] Threshold for non-Otsu methods")
    parser.add_argument('--input-path', type=str, default='images', help="Path to input images")
    parser.add_argument('--sal-path', type=str, default='cams', help="Path to saliency maps")
    parser.add_argument('--output-path', type=str, default='output', help="Output path")
    parser.add_argument('--postprocess', type=str, default='thresholding', 
                        choices=['crf', 'thresholding', 'kmeans'], 
                        help="Postprocessing method (crf/thresholding/kmeans)")
    parser.add_argument('--num-contours', type=int, default=1, help="Number of contours to keep")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    print("Postprocessing started...")
    if args.postprocess == 'crf':
        postprocess_crf(args)
    elif args.postprocess == 'thresholding':
        postprocess_thresholding(args)  # Otsu's method
    elif args.postprocess == 'kmeans':
        postprocess_kmeans(args)
    print("Postprocessing done!")