import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def remove_bg(frame, depth_map,threshold=15,color_mask = [0,0,125]):
   
   print(depth_map)
   bg_mask = depth_map<threshold
   mod_frame = frame.copy()
   mod_frame[bg_mask] = color_mask
   return mod_frame

#Check if cuda is available 
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")


#import the midas model
model_type = "DPT_Large" # or "DPT_Hybrid" or "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)

#Move model to cude if available
device = torch.device("cuda" if torch.cuda.is_available( ) else "cpu")
midas.to(device)
midas.eval( )

#Load transforms for the model
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform

#Load the image and apply the transform
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Camera not opened")
    exit(0)
while True:
    ret,frame_1 = cam.read()
    if not ret:
        print("Frame not read")
        break
    
    frame = cv.cvtColor(frame_1, cv.COLOR_BGR2RGB)
    input_batch = transform(frame).to(device)

    #Predict the depth map
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().numpy()

        # Remove the background using the depth map
        mod_frame = remove_bg(frame_1, output, threshold=15, color_mask=[0, 125, 0])

        # Normalize the output for visualization
        output_normalized = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # Apply a colormap (e.g., COLORMAP_PLASMA)
        output_colormap = cv.applyColorMap(output_normalized, cv.COLORMAP_PLASMA)

        # Show the depth map with the colormap
        cv.imshow("Depth Map", output_colormap)

        #show the original image and the modified image
        cv.imshow("Frame", frame_1)
        cv.imshow("Modified Frame", mod_frame)

        key = cv.waitKey(1)
        if key == 'q' :
            break

cam.release()
cv.destroyAllWindows()


