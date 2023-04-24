from PIL import Image
import cv2
import os
import argparse
import logging
from tqdm import tqdm
import multiprocessing


def watermark_image(image_path, watermark_path, output_dir):
    """Adds a watermark to an image"""
    try:
        # Load the image and watermark
        img = tqdm(Image.open(image_path), desc=f"Processing {image_path}")
        watermark = Image.open(watermark_path)

        # Set the watermark position
        position = (img.width - watermark.width, img.height - watermark.height)

        # Add the watermark to the image
        img.paste(watermark, position, watermark)

        # Save the image with the watermark
        output_path = os.path.join(output_dir, 'watermarked_' + os.path.basename(image_path))
        img.save(output_path)
    except Exception as e:
        logging.error(f"Error processing image: {image_path}. {str(e)}")


def watermark_video(video_path, watermark_path, output_dir):
    """Adds a watermark to a video"""
    try:
        # Load the video
        video = cv2.VideoCapture(video_path)

        # Get the video width and height
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load the watermark image
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

        # Create an output video file
        output_path = os.path.join(output_dir, 'watermarked_' + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        # Loop through each frame in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(total_frames), desc=f"Processing {video_path}", total=total_frames):
            ret, frame = video.read()

            # Add the watermark image to the frame
            watermark_resized = cv2.resize(watermark, (frame.shape[1], frame.shape[0]))
            alpha_s = watermark_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[:, :, c] = (alpha_s * watermark_resized[:, :, c] +
                                  alpha_l * frame[:, :, c])

            # Write the watermarked frame to the output video file
            output.write(frame)

        # Release the video capture and output objects
        video.release()
        output.release()
    except Exception as e:
        logging.error(f"Error processing video: {video_path}. {str(e)}")


if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Add a watermark to images and videos.')
    parser.add_argument('input_dir', help='the directory containing the input files')
    parser.add_argument('watermark_path', help='the path to the watermark image')
    parser.add_argument('--output-dir', dest='output_dir', help='the directory to save the output files')
    parser.add_argument('--log', dest='log_path', help='the path to the log file')
    parser.add_argument('--processes', dest='num_processes', type=int, default=1,
                        help='the number of processes to use for multiprocessing')

    # Parse command-line arguments
    args = parser.parse_args()

    # Set up logging
    if args.log_path:
        logging.basicConfig(filename=args.log_path, level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)
     
    # Create the output directory if it doesn't exist   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get a list of all files in the input directory
    file_list = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]

    # Split the file list into chunks for multiprocessing
    chunk_size = int(len(file_list) / args.num_processes) + 1
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    # Set up a multiprocessing pool
    pool = multiprocessing.Pool(processes=args.num_processes)

    # Process each file in parallel
    for file_chunk in file_chunks:
        for filename in tqdm(file_chunk):
            # Check if the file is an image or video
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                input_path = os.path.join(args.input_dir, filename)
                pool.apply_async(watermark_image, args=(input_path, args.watermark_path, args.output_dir),
                                 error_callback=lambda e: logging.error(f"Error processing image: {input_path}. {str(e)}"))
            elif filename.endswith('.mp4') or filename.endswith('.avi'):
                input_path = os.path.join(args.input_dir, filename)
                pool.apply_async(watermark_video, args=(input_path, args.watermark_path, args.output_dir),
                             error_callback=lambda e: logging.error(f"Error processing video: {input_path}. {str(e)}"))

    # Clean up the multiprocessing pool
    pool.close()
    pool.join()

    logging.info("Done!")
