
  
import cv2
import mediapipe as mp
import time

faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils


# For webcam input:
cap = cv2.VideoCapture(1)

with faceDetector.FaceDetection(

    min_detection_confidence=0.5) as face_detection:

  while cap.isOpened():

    success, image = cap.read()

    start = time.time()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    if results.detections:
      for id, detection in enumerate(results.detections):
        drawing.draw_detection(image, detection)
        print(id, detection)

        bBox = detection.location_data.relative_bounding_box

        h, w, c = image.shape

        boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

        cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 2)

    cv2.imshow('MediaPipe Face Detection', image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()