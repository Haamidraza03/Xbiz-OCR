import os
from ultralytics import YOLO
from PIL import Image

# --- Configuration ---
# 1. Model Selection:
# 'yolov8n.pt' (nano) is the fastest model, great for inference.
# Replace this with 'path/to/your/finetuned/best.pt' once you complete training.
YOLO_WEIGHTS = 'yolov8n.pt' 

# Define the source image. You must replace this with a real image path.
# For now, it uses a sample image URL, but you can change it to a local file path.
INPUT_SOURCE = 'https://ultralytics.com/images/bus.jpg' 
OUTPUT_DIR = 'yolo_output'

def run_yolo_detection(source_path: str, model_weights: str, output_dir: str):
    """
    Loads a YOLO model and runs object detection on a specified image or URL.

    Args:
        source_path (str): Path to the image file or a URL.
        model_weights (str): Path to the YOLO model weights (.pt file).
        output_dir (str): Directory where the results will be saved.
    """
    try:
        # 1. Load the model (downloads weights automatically on first run)
        print(f"Loading YOLO model: {model_weights}...")
        model = YOLO(model_weights)

        # 2. Run prediction
        # 'save=True' automatically saves the resulting image with bounding boxes
        # to the 'runs/detect/' directory (or the custom 'project' dir if specified).
        print(f"Running detection on source: {source_path}")
        
        # NOTE: We set 'project' and 'name' to control where the output is saved.
        results = model.predict(
            source=source_path,
            conf=0.25,      # Minimum confidence threshold (0.25 is default)
            iou=0.7,        # IoU threshold for Non-Maximum Suppression (NMS)
            save=True,      # Save the annotated image
            project=output_dir, # Save results to this directory
            name='detection_run', # Create a subfolder under the project directory
            exist_ok=True   # Overwrite the 'detection_run' folder if it exists
        )

        # 3. Process results and print summary
        for result in results:
            print("\n--- Detection Summary ---")
            
            # Check for specific classes like 'person' for face/people detection
            person_count = result.boxes.cls.tolist().count(0.0) # Class 0 is 'person' in COCO dataset
            
            # Print bounding box information for all detected objects
            print(f"Total objects detected: {len(result.boxes)}")
            print(f"Total 'person' objects (potential faces/people) detected: {person_count}")
            
            # Display class names and number of detections
            counts = result.boxes.cls.tolist()
            class_names = result.names # A dictionary mapping class ID to name
            
            # Calculate and print counts for all detected classes
            class_counts = {class_names[int(cls)]: counts.count(cls) for cls in set(counts)}
            print("Detected Classes and Counts:")
            for name, count in class_counts.items():
                print(f"  - {name}: {count}")
                
        print(f"\n✅ Detection completed. Result image saved in: {os.path.join(output_dir, 'detection_run')}")


    except Exception as e:
        print(f"An error occurred during detection: {e}")
        print("Please ensure you have run 'pip install ultralytics opencv-python Pillow' and your INPUT_SOURCE path is correct.")

# -----------------------------------------------------------
# --- THE FINE-TUNING CODE (HOW TO TRAIN YOUR OWN MODEL) ---
# -----------------------------------------------------------

def run_yolo_finetuning():
    """
    This function demonstrates the steps needed to fine-tune the YOLO model.
    NOTE: This section is commented out because actual training requires a labeled dataset,
    significant time, and computing resources (preferably a GPU).
    """
    
    # Prerequisites for fine-tuning:
    # 1. Prepare your custom dataset (e.g., for face detection) in the YOLO format.
    # 2. Create a YAML configuration file (e.g., 'face_data.yaml') pointing to 
    #    your training/validation directories and defining class names (e.g., 'face').

    # model = YOLO(YOLO_WEIGHTS) # Start with a pre-trained model (transfer learning)

    # Example command to fine-tune the model:
    # The result will be a 'best.pt' file in the 'runs/detect/train' directory.
    # results = model.train(
    #     data='face_data.yaml',   # Path to your dataset configuration file
    #     epochs=50,               # Number of epochs (iterations) to train
    #     imgsz=640,               # Training image size
    #     batch=16,                # Batch size (adjust based on GPU memory)
    #     name='yolov8_face_finetune' # Name of the training run directory
    # )
    
    print("\n[Fine-Tuning Information]")
    print("To get a 'fine-tuned' model, you must run the commented-out model.train(...) command.")
    print("Once complete, replace YOLO_WEIGHTS = 'yolov8n.pt' with the path to your new weights (e.g., 'runs/detect/yolov8_face_finetune/weights/best.pt').")
    
# -----------------------------------------------------------

if __name__ == '__main__':
    # 1. Run the live detection example
    run_yolo_detection(INPUT_SOURCE, YOLO_WEIGHTS, OUTPUT_DIR)
    
    # 2. Print information about fine-tuning
    run_yolo_finetuning()