import torch
import plotly.graph_objects as go
import numpy as np
import cv2 as cv


def plot_3d_map_plotly(frame, depth_map):
    """
    Plots the image in 3D space using Plotly where:
    - x-axis represents depth values
    - y-z plane represents the color values of the image
    """
    # Normalize depth map to a range suitable for plotting
    depth_map_normalized = cv.normalize(depth_map, None, 0, 1, cv.NORM_MINMAX)

    # Get the dimensions of the frame
    h, w, _ = frame.shape

    # Create a grid of y and z coordinates (image plane)
    z,y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten the arrays for 3D plotting
    x = depth_map_normalized.flatten()  # Depth values (x-axis)
    y = w-y.flatten()  # y-coordinates (image height)
    z = h-z.flatten()  # z-coordinates (image width)
    colors = frame.reshape(-1, 3) / 255.0  # Normalize RGB values to [0, 1]

    # Convert RGB colors to hex for Plotly
    colors_hex = ['rgb({}, {}, {})'.format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors_hex,  # Set marker colors
            opacity=0.8
        )
    )])

    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title="Depth (x-axis)",
            yaxis_title="Height (y-axis)",
            zaxis_title="Width (z-axis)"
        ),
        title="3D Depth Map Visualization"
    )

    # Show the plot
    fig.show()

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
frame_1 = cv.imread('samples\dog.jpg')

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


# Normalize the output for visualization
output_normalized = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# Apply a colormap (e.g., COLORMAP_PLASMA)
output_colormap = cv.applyColorMap(output_normalized, cv.COLORMAP_PLASMA)

# Show the depth map with the colormap
cv.imshow("Depth Map", output_colormap)

#show the original image and the modified image
cv.imshow("Frame", frame_1)
plot_3d_map_plotly(frame, output)


cv.waitKey(0)
cv.destroyAllWindows()

