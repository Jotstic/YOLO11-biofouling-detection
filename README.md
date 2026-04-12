# YOLO11-biofouling-detection
The scripts i used when writing my master thesis about using YOLO11 as a model for automatic biofouling detection on salmon net pens. 

CIclass.py is a script that calculate the confidence intervals for each class when evaluating the model on the test part of the datasets. 

Inference_test.py is a script which run the YOLO11 models on videos and calculate the inference speed for each model. 

Automaticannotationtobiigle.py is a script that automatically runs your trained YOLO11 model on every image in a specified BIIGLE volume, detects biofouling organisms and fish, and uploads the predicted bounding boxes back to BIIGLE as labeled annotations. It handles authentication, image downloading, confidence thresholding, and retries, with an optional dry-run mode that previews detections without actually posting anything.

Colabtrainingscript.py trains all five YOLO11 model sizes on my datasets, and for each model it automatically saves the best weights to Google Drive, runs a validation pass, and stores the evaluation plots and metrics in organised folders so I can compare the models afterwards.

dataset.py is a script that creates a stacked bar chart showing how annotated instances are distributed across training, validation, and test splits for each class in completedataset1.

Export_dataset_frombiigle.py connects to BIIGLE, downloads all images and their rectangle annotations from one or more volumes, converts the bounding boxes to YOLO format, and saves everything into a ready to train dataset with a global train/val/test split, a dataset.yaml file, and logs for any problems encountered.

 Frameextractor.py processes underwater video files by scanning each frame with Laplacian sharpness and DCT-based blockiness filters to select the best quality frames, then extracts them as JPEG images using ffmpeg, with a second command available to find and remove exact and near-duplicate images from the output folder.

 Inference_graph.py plots inference latency against detection performance for all five YOLO11 model sizes, with horizontal error bars showing latency variation, to visualise the speed-accuracy trade-off.

 Merge_algae.py takes the test split from an existing dataset, remaps the original seven classes to five by merging green and brown algae into a single algae class and dropping bivalves, and saves the result as a new YOLO-compatible dataset ready for evaluation.

Plot_algae_comparison.py reads per-class evaluation CSVs from three different dataset configurations, extracts the algae-related classes from each, and plots a grouped bar chart comparing precision, recall, AP50, and AP50-95 across the different algae class definitions with 95% confidence interval error bars.

Plot_metrics_with_ci.py reads a CSV of per-class detection metrics with confidence intervals and plots a grouped bar chart comparing precision, recall, mAP50, and mAP50-95 across either different image sizes or model sizes, with score values labelled inside each bar.

Separate_test_run.py runs a trained YOLO model on a test dataset, saves the predicted bounding boxes, evaluates overall and per-class detection metrics, counts ground truth versus predicted instances per class, and generates side-by-side comparison images showing ground truth boxes in green and predictions in red.
