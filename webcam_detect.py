import cv2
import sys

try:
    cascPath = sys.argv[1] # allow specified file
except:
    cascPath = 'haarcascades/haarcascade_frontalface_default.xml' # use local file
    cascPath = 'haarcascades/haarcascade_righteye_2splits.xml' # use local file
    cascPath = 'haarcascades/haarcascade_lefteye_2splits.xml' # use local file
    cascPath = 'haarcascades/haarcascade_eye.xml' # use local file

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        #flags=cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    eye_count = 0.1 # if there is 1 eye, this value is 1.1. round(1/2) = 0, but round (1.1/2) = 1.
    for (x, y, w, h) in faces:
        eye_count+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    face_count = round(eye_count/2)
    print("Detected %d faces from %d eyes." % (face_count,eye_count))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()