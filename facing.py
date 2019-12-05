from SimpleCV import Camera, Display, Image, Color, DrawingLayer
from picamera import PiCamera
from time import sleep

def doface(aa,f1,cc,f2,ee):
    
    camera = PiCamera()
    #imgg = Image('img1.jpg')
    #disp = Display(imgg.size())
    dsize = (640,480)
    disp = Display(dsize)
    #drawing = Image('mustache.png')
    #maskk = drawing.createAlphaMask()

    #camera.start_preview()
    #sleep(2)

    #['right_eye.xml', 'lefteye.xml', 'face3.xml', 'glasses.xml',
    # 'right_ear.xml', 'fullbody.xml', 'profile.xml', 'upper_body2.xml',
    # 'face.xml', 'face4.xml', 'two_eyes_big.xml', 'right_eye2.xml',
    # 'left_ear.xml', 'nose.xml', 'upper_body.xml', 'left_eye2.xml',
    # 'two_eyes_small.xml', 'face2.xml', 'eye.xml', 'face_cv2.xml',
    # 'mouth.xml', 'lower_body.xml']

    while disp.isNotDone():
        camera.capture('img2.png')
        img = Image('img2.png')
        img = img.resize(640,480)
        #whatt = img.listHaarFeatures()
        faces = img.findHaarFeatures('face.xml')
        print 'faces:',faces
        if faces:#is not None:
            face = faces.sortArea()[-1]
            #print 'size:',face.size
            if aa == 'none':
                break
            elif aa == 'block':
                face.draw()
            else:
                f0draw = aa + '.png'
                draw0 = Image('use/'+f0draw)
                face = face.blit(draw0, pos=(100,200))
            #bigFace = face[-1]


            myface = face.crop()
            if f1 and cc is not None:
                feature01 = f1 + '.xml'
                f1draw = cc + '.png'
                draw1 = Image('/home/pi/cv/use/'+f1draw)
      
                feature1s = myface.findHaarFeatures(feature01)
                if feature1s is not None:
                    feature1 = feature1s.sortArea()[-1]
                    xpos1 = face.points[0][0] + feature1.x - (draw1.width/2)
                    ypos1 = face.points[0][1] + feature1.y #+ (2*draw1.height/3)
                    #pos = (xmust,ymust)
                    img = img.blit(draw1, pos=(xpos1,ypos1)) #mask=maskk)

            if f2 and ee is not None:
                feature02 = f2 + '.xml'
                f2draw = ee + '.png'
                draw2 = Image('/home/pi/cv/use/'+f2draw)
    
                feature2s = myface.findHaarFeatures(feature02)
                if feature2s is not None:
                    feature2 = feature2s.sortArea()[-1]
                    xpos2 = face.points[0][0] + feature2.x - (draw2.width/2)
                    ypos2 = face.points[0][1] + feature2.y #+ (2*draw2.height/3)
                    #pos = (xmust,ymust)
                    img = img.blit(draw2, pos=(xpos2,ypos2)) #mask=maskk)

            img.save(disp)
        else:
            print 'no face~~'
            #return
            


            #newlayer = DrawingLayer(img.size())
            #newlayer.text("No People", (width/2, height/2), color=Color.WHITE)
        
        
        #img.save(disp)
        #img.save('img11.jpg')


