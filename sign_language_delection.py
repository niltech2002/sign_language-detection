import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the trained model and label map
MODEL_PATH = 'ssd_mobilenet_sign_language/saved_model'
LABEL_MAP_PATH = 'label_map.pbtxt'

# Load the TensorFlow model
detect_fn = tf.saved_model.load(MODEL_PATH)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Preprocess frame for model input
    input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_tensor)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Perform detection
    detections = detect_fn(input_tensor)
    
    # Extract detection info
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # Visualize results
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5
    )
    
    # Display the output
    cv2.imshow('Sign Language Detection', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
