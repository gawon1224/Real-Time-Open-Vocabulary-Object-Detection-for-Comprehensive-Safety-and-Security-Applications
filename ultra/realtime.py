from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Initialize the YOLO model
model = YOLO("yolov8x-world.pt")  # Ensure the model file is in the correct path

# 사용자로부터 YOLO 모델의 클래스 입력 받기
input_classes = input("Enter the classes to detect, separated by commas (e.g., chair,desk,monitor): ")
# 쉼표로 나누고 양쪽 공백만 제거
class_list = [cls.strip() for cls in input_classes.split(",") if cls.strip()]
if not class_list:
    print("Error: No classes provided. Exiting.")
    exit()

# YOLO 모델에 사용자 지정 클래스 설정
model.set_classes(class_list)
print(f"Detecting the following classes: {class_list}")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

while True:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Predict objects in the frame
    results = model.predict(image)

    # Annotate the image with predictions
    annotated_frame = results[0].plot()  # YOLO provides `plot` to annotate images

    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the annotated frame using Matplotlib
    plt.imshow(rgb_frame)
    plt.axis("off")  # Hide axis
    plt.pause(0.01)  # Pause for a brief moment to allow updates
    plt.clf()  # Clear the figure for the next frame

    # # Break loop on 'q' key press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
plt.close()
