import cv2
from random import randrange
print('code completed');

#Load some pre-trained data : (face-frontals)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Select a image to detect face
# img = cv2.imread('mh.jpg')


#To capture video from cam

webcam = cv2.VideoCapture(0)



#------------For live Video or Webcam----------------------

#Iterate infinitely forever over frames
while True:

    #Read current frames
    successful_frame_read, frame = webcam.read()

    #convert ot grayscale
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #lets detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

    #Draw rectangles around faces
    # (x, y, w, h) = face_coordinates[0]

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    # For showing image with detected faces
    cv2.imshow('Face Detector', frame)

    # For keeping image open till any key is pressed
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


webcam.release();
'''
#-------------------------For a image(Multi face)--------------------------------
#resized image as it was not fitting in my window
img = cv2.resize(img, (960, 640))


#convert ot grayscale
grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#lets detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

#printing face co-ordinates of our test image
print(face_coordinates)

#Draw rectangles around faces
# (x, y, w, h) = face_coordinates[0]

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)


# For showing image with detected faces
cv2.imshow('Face Detector', img)

# For keeping image open till any key is pressed
cv2.waitKey()
'''