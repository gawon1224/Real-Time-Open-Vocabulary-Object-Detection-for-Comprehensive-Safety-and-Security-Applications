from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Initialize the YOLO model
model = YOLO("yolov8m-world.pt")
model.set_classes(["student"])

# Track objects in the video with stream=True
results = model.track(source="/home/device03/ultra/classroom.mp4", stream=True)

for r in results:
    # Annotate the frame with tracking results
    annotated_frame = r.plot()

    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Matplotlib
    plt.imshow(rgb_frame)
    plt.axis("off")  # Hide axis
    plt.pause(0.01)  # Pause for a brief moment to allow updates
    plt.clf()  # Clear the figure for the next frame
