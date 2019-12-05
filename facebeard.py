from SimpleCV import Camera, Display, Image,DrawingLayer
from picamera import PiCamera
from time import sleep

#camera = PiCamera()
#camera.start_preview()
#sleep(2)


#['right_eye.xml', 'lefteye.xml', 'face3.xml', 'glasses.xml',
# 'right_ear.xml', 'fullbody.xml', 'profile.xml', 'upper_body2.xml',
# 'face.xml', 'face4.xml', 'two_eyes_big.xml', 'right_eye2.xml',
# 'left_ear.xml', 'nose.xml', 'upper_body.xml', 'left_eye2.xml',
# 'two_eyes_small.xml', 'face2.xml', 'eye.xml', 'face_cv2.xml',
# 'mouth.xml', 'lower_body.xml']


#camera.capture('img1.jpg')
#img = Image('img1pg')
img = Image('img1.jpg')

disp = Display(img.size())
#while disp.isNotDone():
    #img = cam.getImage()
   # camera.capture('img2.jpg')
    #img = Image('img2.jpg')
whatt = img.listHaarFeatures()
#print whatt
faces = img.findHaarFeatures('face.xml')
if faces is not None:
    faces = faces.sortArea()
    bigFace = faces[-1]
    bigFace.draw()
    img.save(disp)
    img.save('img11.jpg')
