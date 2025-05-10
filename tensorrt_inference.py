import os
import argparse
import numpy as np
import cv2
from tensorrt_utils import MASt3RTensorRTInference

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess image for MASt3R model input
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension and transpose to NCHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Run inference using TensorRT MASt3R model')
    parser.add_argument('--engine_path', type=str, required=True,
                      help='Path to the TensorRT engine')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save output')
    
    args = parser.parse_args()
    
    # Initialize TensorRT inference
    inference = MASt3RTensorRTInference(args.engine_path)
    
    # Preprocess image
    input_tensor = preprocess_image(args.image_path)
    
    # Run inference
    output = inference.infer(input_tensor)
    
    # Process output (this will depend on your specific model's output format)
    # For example, if output is depth map:
    depth_map = output[0, 0]  # Assuming first channel is depth
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Save output
    cv2.imwrite(args.output_path, depth_map)
    print(f"Output saved to: {args.output_path}")

if __name__ == '__main__':
    main() 