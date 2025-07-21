import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def initialize():
    """Initialize the YOLO-World model."""
    model = YOLO("yolov8x-worldv2.pt")
    return model

def object_detection(model, image_path, objects):
    """Perform object detection using YOLO-World and filter by specified classes."""
    model.set_classes(objects)
    results = model.predict(image_path, verbose=False)

    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label_index = int(box.cls)
            label = objects[label_index]
            score = round(float(box.conf), 2)
            xyxy = box.xyxy[0].tolist()
            box_coords = [round(coord, 2) for coord in xyxy]
            detected_objects.append({"label": label, "score": score, "box": box_coords})
    return detected_objects

def visualize_objects(image_path, objects_to_visualize):
    """Visualize detected objects on the image."""
    img = cv2.imread(image_path)
    for obj in objects_to_visualize:
        box = obj["box"]
        label = obj["label"]
        score = obj["score"]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, f"{label} {score}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the image with detected objects
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_t, y1_t, x2_t, y2_t = box2

    # Calculate the area of intersection
    inter_x1 = max(x1, x1_t)
    inter_y1 = max(y1, y1_t)
    inter_x2 = min(x2, x2_t)
    inter_y2 = min(y2, y2_t)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate the area of both bounding boxes
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_t - x1_t) * (y2_t - y1_t)

    # Calculate the Intersection over Union (IoU)
    union_area = area1 + area2 - inter_area
    iou_value = inter_area / union_area if union_area != 0 else 0
    return iou_value

def filter_candidates_by_iou(new_candidates, previous_candidates, iou_threshold=0.3):
    """Filter new candidates based on IoU with the previous candidates."""
    filtered_candidates = []
    
    for new_candidate in new_candidates:
        is_valid = False
        for prev_candidate in previous_candidates:
            iou_value = iou(new_candidate["box"], prev_candidate["box"])
            if iou_value >= iou_threshold:
                is_valid = True
                break
        
        if is_valid:
            filtered_candidates.append(new_candidate)
    
    return filtered_candidates

def get_class_from_question(question):
    """Extract class name from the user question."""
    # Simple example: assume classes are separated by spaces
    return question.lower().replace("is it", "").strip()

def game_flow(image_path):
    model = initialize()

    # Step 1: Perform initial object detection on the image
    objects = ["person", "blue bus", "car", "bottle", "black car", "red car", "place to sit down"]
    initial_detection = object_detection(model, image_path, objects)
    print(initial_detection)

    # Step 2: Find the object where the label is 'black car'
    correct_answer = None

    for obj in initial_detection:
        if obj['label'] == 'blue bus':
            correct_answer = obj
            break  # 'black car'를 찾으면 루프 종료
    print(correct_answer)
    
    # Visualize the first round of detected objects
    visualize_objects(image_path, initial_detection)

    # Game loop
    candidate_objects = initial_detection

    for round_num in range(10):
        print(f"\nRound {round_num + 1}: Please ask a question!")
        question = input("Enter question: ")

        # Step 3: Extract the class from the question
        class_name = get_class_from_question(question)
        print(f"Extracted class from question: {class_name}")

        # Step 4: Perform object detection with the updated class list
        new_candidates = object_detection(model, image_path, [class_name])
        print(f"Objects detected for class '{class_name}': {new_candidates}")

        # Step 5: Filter candidates by IoU with the previous candidates
        candidate_objects = filter_candidates_by_iou(new_candidates, candidate_objects)
        print(f"Candidates after IoU filtering: {candidate_objects}")

        # Step 6: Visualize the detected objects
        visualize_objects(image_path, candidate_objects)

        # Step 7: If there is only one candidate, user wins!
        if len(candidate_objects) == 1:
            print("You win! The correct answer is:", candidate_objects[0]['label'])
            return
        
        # Step 8: Proceed to next round
        print(f"Candidates after round {round_num + 1}: {len(candidate_objects)} objects remaining.")
    
    print("You lose! The correct answer could not be determined.")
    
if __name__ == "__main__":
    image_path = "/home/device03/ultra/bus.png"  # Update with the correct image path
    game_flow(image_path)
