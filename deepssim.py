
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import numpy as np
from tqdm import tqdm
import random

# First, ensure you have the necessary packages installed:
# pip install torch torchvision iqadataset scipy tqdm

# --- Step 1: Define the Backbone-Agnostic DeepSSIM Class ---
class DeepSSIM(nn.Module):
    """
    A backbone-agnostic implementation of the DeepSSIM metric.
    It accepts a pre-initialized feature extractor and its corresponding
    normalization function.
    MODIFIED: This version selects a percentage of feature maps based on
    the reference image's channel-wise variance before calculation.
    """
    def __init__(self, feature_extractor, normalize, feature_node_key='features', lite_version=False, device=None, percent_features_to_keep=0.5, window_size=2):
        """
        Initializes the DeepSSIM model.

        Args:
            feature_extractor (nn.Module): A pre-initialized feature extraction model.
            normalize (callable): A function or transform to normalize input images for the feature extractor.
            feature_node_key (str): The key to access the feature tensor if the extractor returns a dict.
            lite_version (bool): If True, uses the faster DeepSSIM-Lite version.
            device (torch.device, optional): The device to run the model on.
            percent_features_to_keep (float): The percentage of feature maps to keep based on variance (0.0 to 1.0).
            window_size (int): The size of the window for SSIM calculation in the non-lite version.
        """
        super(DeepSSIM, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.lite_version = lite_version
        self.window_size = window_size
        self.xi = 1e-8
        self.percent_features_to_keep = percent_features_to_keep

        # Store the supplied feature extractor and normalization function
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key

    def _get_deep_features(self, img):
        img_norm = self.normalize(img)
        features = self.feature_extractor(img_norm)
        # Handle dictionary output from create_feature_extractor
        if isinstance(features, dict):
            return features[self.feature_node_key]
        return features

    def _compute_gram_matrix(self, feature_map):
        n, c, h, w = feature_map.size()
        features_reshaped = feature_map.view(n, c, h * w)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        return gram / (h * w)

    def forward(self, ref_img, dist_img):
        # 1. Extract all features first
        features_ref = self._get_deep_features(ref_img.to(self.device))
        features_dist = self._get_deep_features(dist_img.to(self.device))

        # 2. Select top K% of features based on reference image variance
        n, c, h_ref, w_ref = features_ref.shape

        # Calculate variance for each channel across spatial dimensions
        ref_variances = torch.var(features_ref, dim=(2, 3), unbiased=False)

        # Determine the number of channels to keep based on the percentage
        num_channels_to_keep = int(c * self.percent_features_to_keep)
        if num_channels_to_keep == 0 and c > 0:
             num_channels_to_keep = 1 # Ensure at least one channel if features exist


        # Get the indices of the channels with the highest variance
        _, top_indices = torch.topk(ref_variances, num_channels_to_keep, dim=1)

        # 3. Filter both reference and distorted features using these indices
        # Expand indices to match the feature map dimensions for gathering
        top_indices_ref = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_ref, w_ref)
        selected_features_ref = torch.gather(features_ref, 1, top_indices_ref)

        _, _, h_dist, w_dist = features_dist.shape
        top_indices_dist = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_dist, w_dist)
        selected_features_dist = torch.gather(features_dist, 1, top_indices_dist)


        # 4. Proceed with DeepSSIM calculation on the selected features
        gram_ref = self._compute_gram_matrix(selected_features_ref)
        gram_dist = self._compute_gram_matrix(selected_features_dist)

        if self.lite_version:
            var_ref = torch.var(gram_ref, dim=(1, 2), unbiased=False)
            var_dist = torch.var(gram_dist, dim=(1, 2), unbiased=False)
            mean_ref = torch.mean(gram_ref, dim=(1, 2), keepdim=True)
            mean_dist = torch.mean(gram_dist, dim=(1, 2), keepdim=True)
            covar = torch.mean((gram_ref - mean_ref) * (gram_dist - mean_dist), dim=(1, 2))
            score = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
        else:
            gram_ref_unf = F.unfold(gram_ref.unsqueeze(1), kernel_size=self.window_size, stride=1, padding=0)
            gram_dist_unf = F.unfold(gram_dist.unsqueeze(1), kernel_size=self.window_size, stride=1, padding=0)
            gram_ref_unf = gram_ref_unf.transpose(1, 2)
            gram_dist_unf = gram_dist_unf.transpose(1, 2)

            var_ref = torch.var(gram_ref_unf, dim=2, unbiased=False)
            var_dist = torch.var(gram_dist_unf, dim=2, unbiased=False)
            mean_ref = torch.mean(gram_ref_unf, dim=2, keepdim=True)
            mean_dist = torch.mean(gram_dist_unf, dim=2, keepdim=True)
            covar = torch.mean((gram_ref_unf - mean_ref) * (gram_dist_unf - mean_dist), dim=2)

            local_scores = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
            score = torch.mean(local_scores, dim=1)

        return score

# --- Step 2: Define Evaluation Logic ---

def logistic_func(x, beta1, beta2, beta3, beta4, beta5):
    logistic_part = beta2 * (x - beta3)
    clipped_logistic_part = np.clip(logistic_part, -100, 100)
    logistic = beta1 * (0.5 - 1 / (1 + np.exp(clipped_logistic_part))) + beta4 * x + beta5
    return logistic

def get_feature_extractor(backbone_name):
    """Creates a feature extractor and its normalization function based on the backbone name."""
    if backbone_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        # Use 'features.23' for conv5_1 layer in VGG16
        return_nodes = {'features.23': 'features'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize, 'features'

    elif backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        return_nodes = {'features.5.5.add': 'features'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize, 'features'

    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported.")


def evaluate_on_dataset(dataset_name="TID2013", backbone_name="vgg16", percent_features_to_keep=0.5, window_size=4):
    """Main function to run the evaluation on a specified IQA dataset with a specified backbone."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create the selected feature extractor and normalization ---
    print(f"Creating feature extractor for: {backbone_name}")
    feature_extractor, normalize, feature_node_key = get_feature_extractor(backbone_name)

    # --- Initialize the DeepSSIM model with the chosen backbone ---
    model = DeepSSIM(
        feature_extractor=feature_extractor,
        normalize=normalize,
        feature_node_key=feature_node_key,
        lite_version=False,
        device=device,
        window_size=window_size,
        percent_features_to_keep=percent_features_to_keep # Pass the new argument here
    )

    # --- Load Dataset ---
    try:
        from iqadataset import load_dataset_pytorch
        print(f"Loading {dataset_name} dataset via iqadataset...")
        full_dataset = load_dataset_pytorch(dataset_name)
        print("Dataset loaded successfully.")
    except ImportError:
        print("Error: 'iqadataset' package not found. Please install it: pip install iqadataset")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # --- Handle large datasets with random sampling ---
    if dataset_name in ["KADID-10k", "PIPAL"] and len(full_dataset) > 1000:
        print(f"Dataset {dataset_name} is large. Performing random sampling 5 times.")
        all_srcc = []
        all_plcc = []
        num_samples = 1000
        num_runs = 5

        for run in range(num_runs):
            print(f"\n--- Running sample {run + 1}/{num_runs} ---")
            # Randomly select 1000 indices
            indices = random.sample(range(len(full_dataset)), num_samples)
            subset_dataset = Subset(full_dataset, indices)
            data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=2)

            # --- Run Inference on Subset ---
            model.eval()
            subset_preds = []
            subset_scores = []

            print(f"Running inference on a random sample of {num_samples} images...")
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name} sample {run+1}"):
                    ref_img = batch['ref_img'].to(device)
                    dist_img = batch['dis_img'].to(device)
                    scores = batch['score'].to(device)
                    preds = model(ref_img, dist_img)
                    subset_preds.append(preds.cpu())
                    subset_scores.append(scores.cpu())

            subset_preds = torch.cat(subset_preds).numpy()
            subset_scores = torch.cat(subset_scores).numpy()

            # --- Calculate Metrics for Subset ---
            srcc, _ = spearmanr(subset_preds, subset_scores)
            try:
                initial_params = [np.max(subset_scores), 10, np.mean(subset_preds), 0.1, 0.1]
                popt, _ = curve_fit(logistic_func, subset_preds, subset_scores, p0=initial_params, maxfev=10000)
                preds_mapped = logistic_func(subset_preds, *popt)
                plcc, _ = pearsonr(preds_mapped, subset_scores)
            except Exception as e:
                print(f"\nCould not fit the logistic function for PLCC calculation on sample {run+1}: {e}")
                plcc, _ = pearsonr(subset_preds, subset_scores)

            all_srcc.append(srcc)
            all_plcc.append(plcc)

        # --- Calculate and Print Average Results ---
        avg_srcc = np.mean(all_srcc)
        avg_plcc = np.mean(all_plcc)

        print("\n" + "="*50)
        print(f"Average Evaluation Results on {dataset_name} Dataset (using {backbone_name})")
        print(f"Averaged over {num_runs} random samples of {num_samples} images")
        print(f"Keeping {percent_features_to_keep*100}% of features based on variance")
        print("="*50)
        print(f"Average SRCC: {avg_srcc:.4f}")
        print(f"Average PLCC: {avg_plcc:.4f}")
        print("="*50)
        return (avg_srcc, avg_plcc)

    else:
        # --- Original Logic for Smaller Datasets ---
        data_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=2)

        # --- Run Inference ---
        model.eval()
        all_preds = []
        all_scores = []

        print(f"Running inference on {dataset_name} with {backbone_name} and keeping {percent_features_to_keep*100}% features...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                ref_img = batch['ref_img'].to(device)
                dist_img = batch['dis_img'].to(device)
                scores = batch['score'].to(device)
                preds = model(ref_img, dist_img)
                all_preds.append(preds.cpu())
                all_scores.append(scores.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_scores = torch.cat(all_scores).numpy()

        # --- Calculate Metrics ---
        srcc, _ = spearmanr(all_preds, all_scores)
        try:
            initial_params = [np.max(all_scores), 10, np.mean(all_preds), 0.1, 0.1]
            popt, _ = curve_fit(logistic_func, all_preds, all_scores, p0=initial_params, maxfev=10000)
            preds_mapped = logistic_func(all_preds, *popt)
            plcc, _ = pearsonr(preds_mapped, all_scores)
        except Exception as e:
            print(f"\nCould not fit the logistic function for PLCC calculation: {e}")
            plcc, _ = pearsonr(all_preds, all_scores)

        # --- Print Results ---
        print("\n" + "="*50)
        print(f"Evaluation Results on {dataset_name} Dataset (using {backbone_name})")
        print(f"Keeping {percent_features_to_keep*100}% of features based on variance")
        print("="*50)
        print(f"SRCC: {srcc:.4f}")
        print(f"PLCC: {plcc:.4f}")
        print("="*50)
        return (srcc, plcc)

import matplotlib.pyplot as plt
import json

datasets = ["LIVE", "CSIQ", "TID2013", "KADID-10k", "PIPAL"]
percentages = np.arange(0.1, 1.1, 0.1)
window_size = 4
results = {}

for dataset_name in datasets:
    print(f"\nEvaluating on dataset: {dataset_name}")
    srcc_results = []
    plcc_results = []
    for percent in percentages:
        print(f"Running with {percent*100:.0f}% features kept and window size {window_size}")
        srcc, plcc = evaluate_on_dataset(dataset_name=dataset_name, backbone_name="vgg16", percent_features_to_keep=percent, window_size=window_size)
        srcc_results.append(srcc)
        plcc_results.append(plcc)
    results[dataset_name] = {"srcc": srcc_results, "plcc": plcc_results}
    plt.figure(figsize=(10, 6))
    plt.plot(percentages * 100, results[dataset_name]["srcc"], marker='o')
    plt.xlabel("Percentage of Features Kept (%)")
    plt.ylabel("SRCC")
    plt.title(f"SRCC vs. Percentage of Features Kept (Window Size = {window_size}) on {dataset_name}")
    plt.grid(True)
    plt.savefig(f"{dataset_name}_SRCC_w{window_size}.png")
    plt.close() # Close the plot to free up memory

    # Plot PLCC
    plt.figure(figsize=(10, 6))
    plt.plot(percentages * 100, results[dataset_name]["plcc"], marker='o', color='orange')
    plt.xlabel("Percentage of Features Kept (%)")
    plt.ylabel("PLCC")
    plt.title(f"PLCC vs. Percentage of Features Kept (Window Size = {window_size}) on {dataset_name}")
    plt.grid(True)
    plt.savefig(f"{dataset_name}_PLCC_w{window_size}.png")
    plt.close() # Close the plot to free up memory
    with open("deepssim_evaluation_results.json", "w") as f:
        json.dump(results, f)


# Save results to a JSON file
with open("deepssim_evaluation_results.json", "w") as f:
    json.dump(results, f)
