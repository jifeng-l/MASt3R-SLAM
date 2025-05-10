import os
import argparse
from mast3r_slam.tensorrt_utils import MASt3RTensorRTConverter, MASt3RTensorRTInference

def main():
    parser = argparse.ArgumentParser(description='Convert MASt3R models to TensorRT')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the PyTorch model checkpoint')
    parser.add_argument('--engine_path', type=str, required=True,
                      help='Path to save the TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16',
                      choices=['fp32', 'fp16', 'int8'],
                      help='Precision mode for TensorRT')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--height', type=int, default=512,
                      help='Input image height')
    parser.add_argument('--width', type=int, default=512,
                      help='Input image width')
    parser.add_argument('--channels', type=int, default=3,
                      help='Input image channels')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.engine_path), exist_ok=True)
    
    # Initialize converter
    converter = MASt3RTensorRTConverter(
        model_path=args.model_path,
        engine_path=args.engine_path,
        precision=args.precision,
        max_batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        channels=args.channels
    )
    
    # Convert model
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    converter.convert_mast3r_model(input_shape)
    
    print("Model conversion completed successfully!")
    print(f"TensorRT engine saved to: {args.engine_path}")

if __name__ == '__main__':
    main() 