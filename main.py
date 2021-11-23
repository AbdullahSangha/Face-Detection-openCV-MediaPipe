import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture('video/video1.mp4')

#Adding Face Detection to video
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Reading the video file until finished
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Convert image to RGB for processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    # if video finished or no Video Input
    if not ret:
        break
    
    if results.detections:
     for id , detection in enumerate(results.detections):
        #mpDraw.draw_detection(frame, detection)
        #print(detection)
        #print(detection.location_data.relative_bounding_box)
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = frame.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
        int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(frame, bbox, (255, 0, 255), 5)
        #Display Confidance of detection in percent
        cv2.putText(frame, f'{int(detection.score[0]*100)}%', 
        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 4)

    # Our operations on the frame come here
    gray = frame
    #Resize your video here
    gray = cv2.resize(gray, (640, 480))

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()


    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
 
    # displaying the frame with fps
    cv2.imshow('frame', gray)

    #If we put 10 video will be bit slow and thus near the actual frame
    cv2.waitKey(1)
    
    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()