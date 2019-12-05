
from Tkinter import *
from PIL import Image, ImageTk
import ttk
from facing import doface

root = Tk()
root.title('faceu')

class Application(Frame):

    def panit(self):
          if self.sba.get()=='':
              print('please choose a facedraw')
          else:
              print ('predicting...')
              doface(self.sba.get(),self.sbb.get(),self.sbc.get(),self.sbd.get(),self.sbe.get())
              #execfile('/home/pi/traffic/traffic_test.py')
             
    def track(self):
          print('tracking...')
          execfile('/home/pi/cv/trackcle.py')
          #os.system('traffic_train.py')
          
    def warning(self):
          print('warning...')
          execfile('/home/pi/cv/warning.py')
          exit()

    def quitt(self):
          print('quit')
          exit()


    def createWidgets(self):

        #imgg = Image.open('qr.png')
        #img = ImageTk.PhotoImage(imgg)
        # img = PhotoImage(file = 'ftlogo.png')
        # self.cam = Label(root,text='Camera',fg='red',image=img)#,width=640,height=480)
        # self.cam.img = img
        # self.cam.pack(side=LEFT,expand=NO,fill=Y)

        #self.window = tk.Tk()
        #self.window.wm_title('Video Text')
        #self.window.config(background='#FFFFFF')
        #self.canvas = Canvas(self.window, )

        self.kong1 = Label(root,text='    ',width=8)
        self.kong1.pack(side=TOP)


        self.draww = Label(root,text='Face:',width=8)
        self.draww.pack()
        self.sba = ttk.Combobox(root,width=8)
        self.sba['values'] =('none','block','huaji','kelian','ku','xiao','xiaoku','pig','ftlogo','ftlogo1','ym','xyy','smile','mgu','huajib','ccc')
        #self.ent = Entry(root,text='Type Your Directory or Press Browse Button',fg='blue')
        #self.ent.pack()     
        #self.sbb = Button(root,text='Browse...',command=self.onOpenDir)
        self.sba.pack()

        self.kongg = Label(root,text='    ',width=8)
        self.kongg.pack()

        self.feature1 = Label(root,text='Feature1:',width=8)
        self.feature1.pack()
        self.sbb = ttk.Combobox(root,width=8)
        self.sbb['values'] = ('','lefteye','left_ear','hat','glasses',)
        self.sbb.pack()
        
        self.draw1 = Label(root,text='Draw1:',width=8)
        self.draw1.pack()
        self.sbc = ttk.Combobox(root,width=8)
        self.sbc['values'] = ('','eye','ear2','love','lovee','hat','king','xueshi','glass','china','ftlogo','ftlogo1')
        self.sbc.pack()

        self.feature2 = Label(root,text='Feature2:',width=8)
        self.feature2.pack()
        self.sbd = ttk.Combobox(root,width=8)
        self.sbd['values'] = ('','right_eye','right_ear','mouth','nose',)
        self.sbd.pack()
        
        self.draw2 = Label(root,text='Draw2:',width=8)
        self.draw2.pack()
        self.sbe = ttk.Combobox(root,width=8)
        self.sbe['values'] = ('','eye','ear1','love','lovee','kiss','nose','shetou','kiss')
        self.sbe.pack() 

        
        self.b1 = Button(root,text='Pan it',command=self.panit)
        self.b1.pack()
        #self.b1.pack(side=LEFT,expand=YES)

        self.kong2 = Label(root,text='    ',width=8)
        self.kong2.pack()

        self.kong3 = Label(root,text='demo1',width=8)
        self.kong3.pack()
        self.b2 = Button(root,text='Track',command=self.track)
        #self.b2.pack(side=LEFT,expand=YES)
        self.b2.pack()

        self.kong4 = Label(root,text='demo2',width=8)
        self.kong4.pack()
        self.b3 = Button(root,text='Warning',command=self.warning)
        #self.b3.pack(side=LEFT,expand=YES)
        self.b3.pack()

        self.kong5 = Label(root,text='     ',width=8)
        self.kong5.pack()
        self.b4 = Button(root,text='Quit',command=self.quitt)
        #self.b4.pack(side=LEFT,expand=YES)
        self.b4.pack()

    def __init__(self,master = None):
        Frame.__init__(self,master)
        self.createWidgets()

def tkinterDemo():
    app = Application(master = root)
    app.mainloop()
    root.destroy()
    
if __name__=='__main__':
    tkinterDemo()
