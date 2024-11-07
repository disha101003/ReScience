#!/bin/bash

# Define the base path for the models
MODELS_DIR="models/pretrained_models_CiFAR10"

# Loop through all the 'fw' files in the directory
for model_fw in $MODELS_DIR/*_fw.pth; do
    # Extract the index part from the 'fw' file name (e.g., "ft0" from "cifar100ft0_fw.pth")
    base_name=$(basename "$model_fw" "_fw.pth")

    # Look for the matching 'g' file with the same base name
    model_g="$MODELS_DIR/${base_name}_g.pth"

    # Check if the matching 'g' file exists
    if [ -f "$model_g" ]; then
        echo "Running with model_fw: $model_fw and model_g: $model_g"

        # Run the Python script with the model paths as arguments
        python -m src.models.test_pretrained_models --model_fw_path "$model_fw" --model_g_path "$model_g"
    else
        echo "No matching 'g' model found for $model_fw, skipping."
    fi

done
