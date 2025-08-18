import pie_toolbox.workflows.ssm_pca as ssm_pca
import pie_toolbox.workflows.classification as classification
import pie_toolbox.workflows.feature_extraction as feature_extraction
import pie_toolbox.workflows.image_dataset as image_dataset
import pie_toolbox.workflows.export as export
import subprocess
import json
import sys
import os

# To run this example
# python -m examples.example_data_pipeline

# All data is stored in example_data directory.
# You can generate example data using the script generate_example_data.py in example_data directory.


config_path = r'config'
example_data_config_path = os.path.join(config_path, 'example_data_config.json')
with open(example_data_config_path) as f:
    config = json.load(f)

# ----------- Step 0. Create directory structure and generate example data -----------
import os
example_data_path = config["input_dir"]
results_path = config["output_dir"]

os.makedirs(example_data_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

try:
    result = subprocess.run([sys.executable, "example_generate_data.py"],
                            check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    raise Exception(f"Error calling: {e.stderr}")

# ----------- Step 1. Make Image Dataset -----------
subj_images = image_dataset.ImageDataset()

# Load images
for group in config["ssm-pca"]["groups"]:
    subj_images.add_group (os.path.join(example_data_path, config["ssm-pca"]["groups"][group]), label=group)

# Load mask
mask_options = config["ssm-pca"]["mask_options"]
subj_images.load_mask (os.path.join(example_data_path, mask_options["loaded_mask"]["mask_path"]))

# Apply mask
subj_images.apply_mask(threshold=mask_options["threshold_mask"],
                       loaded_mask_threshold=mask_options["loaded_mask"]["threshold"],
                       loaded_mask_is_threshold_relative=mask_options["loaded_mask"]["is_threshold_relative"])


# ----------- Step 2. Make VoxelPCA object and fit Image Dataset -----------
vox_pca = ssm_pca.VoxelPCA()
vox_pca.fit(subj_images)

# ----------- Step 3.1 Get patterns from VoxelPCA object -----------

select_options = config["select_patterns"]

# Get patterns that explain 50% of variance (creates another VoxelPCA object with selected patterns)
patterns_50_variance = vox_pca.get_patterns(explained_variance = select_options["explained_variance"],
                                            cumulative_explained_variance = select_options["cumulative_explained_variance"],
                                            n_patterns=select_options["n_patterns"])

# Get scores for Image Dataset
scores = patterns_50_variance.get_scores(subj_images)

# Draw plots that show scores
plot = export.plot_from_voxelpca(voxelpca=patterns_50_variance,
                                 figsize=config["plot"]["figsize"],
                                 plot_type=config["plot"]["plot_type"],
                                 sorted=config["plot"]["sorted"])
export.save_plot(plot, output_path=results_path+'/plot.png')

# Visualize patterns
export.show_patterns_in_browser(patterns_50_variance, title='Pattern', threshold=0.1)


# ----------- Step 3.2 Get combined pattern from patterns 50% variance -----------
# Get combined pattern (creates another VoxelPCA object)
combined_pattern = patterns_50_variance.get_combined_pattern(scores, subj_images.labels)
# Get scores for Image Dataset
scores_combined = combined_pattern.get_scores(subj_images)
# Visualize the combined pattern
export.show_patterns_in_browser(combined_pattern, title='Combined Pattern', threshold=0)


# ----------- Step 4. Feature extraction from combined pattern -----------
# Create AtlasVOI object
atlas_obj = feature_extraction.AtlasVOI()

# Load combined pattern (first pattern in the VoxelPCA object)
atlas_obj.get_pattern(combined_pattern, index=0)
# Transform combined pattern to atlas with given parameters
atlas_obj.get_atlas(threshold_percentage=0.1, delta=0.1, dust=200, connectivity=6)
# Save atlas as NIFTI
export.save_images(atlas_obj.atlas, filepath=results_path, filename='atlas_image', image_parameters=subj_images._image_params)


# ----------- Step 5. Classification -----------
# Get scores for Image Dataset using atlas
scores_atlas = combined_pattern.get_scores(images=subj_images, atlases=atlas_obj)

classifier_options = config["classification"]

# Create SVM_Classifier object
svm = classification.SVM_Classifier()

# Fit model and perform cross-validation
svm.fit(scores_atlas, subj_images.labels)
svm.cross_validation(split_type=classifier_options["split_type"], folds=classifier_options["folds"])
# Save metrics from cross-validation
export.export_metrics_xlsx(svm.metrics[0].get_metrics(), output_path=results_path+'/test.xlsx', information_text='Test Information')


# ----------- Step 6. Validation -----------
# Create ImageDataset for validation
val_images = image_dataset.ImageDataset()

# Add validation images
for group in classifier_options["validation_groups"]:
    val_images.add_group (os.path.join(example_data_path, classifier_options["validation_groups"][group]), label=group)

# Adjust to dataset (applies mask and log-normalization using parameters from subj_images)
val_images.adjust_to_dataset(subj_images)

# Get scores for validation Image Dataset using atlas and combined pattern
scores_val = combined_pattern.get_scores(images=val_images, atlases=atlas_obj)
# Perform validation
metrics_val = svm.validation(scores_val[0], val_images.labels)
# Save validation metrics
export.export_metrics_xlsx(metrics_val, output_path=results_path+'/validation.xlsx', information_text='Validation Information')
