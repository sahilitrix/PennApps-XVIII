import faceWarp
import cv2
import argparse
import sys
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.switch import Switch
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.metrics import dp, sp
from kivy.core.window import Window
from functools import partial
import time
from sys import exit
import numpy as np
import math
import argparse
import imutils
import pygame
from pygame import mixer
import keyboard
import time
import tkinter as tk
from tkinter import messagebox
import threading
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import imutils

c = 0
mixer.init()
model= load_model("dabs_weighting.model")



Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Button:
        text: 'Play'
        size_hint_x: .15
        size_hint_y: .05
        pos_hint: {'left':1, 'top':1}
        on_press: camera.play = not camera.play
    Button:
        text: 'Start'
        size_hint_x: .15
        size_hint_y: .05
        pos_hint: {'left':1, 'top':0.95}
        on_press: root.start()

    Label: 
        text: "Recording"
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'left': 1, 'top':0.9}
    Switch:
        id: recording_switch
        size_hint_x: .07
        size_hint_y: .05
        pos_hint: {'right':0.15, 'top':0.9}
        on_touch_down: root.toggle_Recording()

    Label: 
        text: "Filter Select"
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'left': 1, 'top':0.85}
    Image:
        size_hint_x: .15
        size_hint_y: .15
        pos_hint: {'right':0.125, 'top':0.8}
        source: 'demo/hulk.jpg'
    Image:
        size_hint_x: .15
        size_hint_y: .15
        pos_hint: {'right':0.235, 'top':0.8}
        source: 'demo/superman.jpg'
    CheckBox:
        text: 'Face1'
        active: True
        group : 'faceSelect'
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'right':0.09, 'top':0.65}
        value: True
        color: [255,150,100,1]
        id: 'hulk.jpg'
        on_press: root.change_filter('hulk.jpg' )
    CheckBox:
        text: 'Face2'
        active: False
        group : 'faceSelect'
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'right':0.2 , 'top':0.65}
        value: False
        color: [255,150,100,1]
        id: 'superman.jpg'
        on_press: root.change_filter('superman.jpg')
    Image:
        size_hint_x: .15
        size_hint_y: .15
        pos_hint: {'right':0.125, 'top':0.6}
        source: 'demo/incredibles.jpg'
    Image:
        size_hint_x: .15
        size_hint_y: .15
        pos_hint: {'right':0.235, 'top':0.6}
        source: 'demo/batman.jpg'
    CheckBox:
        text: 'Face3'
        active: False
        group : 'faceSelect'
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'right':0.09, 'top':0.45}
        value: False
        color: [255,150,100,1]
        id: 'incredibles.jpg'
        on_press: root.change_filter('incredibles.jpg')
    CheckBox:
        text: 'Face4'
        active: False
        group : 'faceSelect'
        size_hint_x: .08
        size_hint_y: .05
        pos_hint: {'right':0.2 , 'top':0.45}
        value: False
        color: [255,255,255,1]
        id: 'batman.jpg'
        on_press: root.change_filter('batman.jpg')
    
    ToggleButton: 
        text: 'Background Music'
        size_hint_x: .15
        size_hint_y: .05
        pos_hint:{'left':1, 'top':0.35}
        on_press: root.toggle_Background_Music()
        
    Button:
        text: 'Stop'
        size_hint_x: .15
        size_hint_y: .05
        pos_hint: {'left':1, 'top':0.2}
        on_press: root.stop()
    Button:
        text: 'Quit'
        size_hint_x: .15
        size_hint_y: .05
        pos_hint: {'left':1, 'top':0.15}
        on_press: root.close()
''')
    
Window.fullscreen = 'auto'

class MusicPlayer(object):
    """Plays Different sounds based on gesture given.

    Attributes:
        none so far
    """

    def __init__(self):
        """Return a MusicPlayer object."""
        self.sprinkle = mixer.Sound("sprinkle2.wav")
        self.scratch = mixer.Sound("scratch2.wav")
        self.drop = mixer.Sound("DROP_2.wav")
        self.clap = mixer.Sound("CLAP_1.wav")
        self.clap2 = mixer.Sound("CLAP_2.wav")
        self.kick = mixer.Sound("KICK_1.wav")
        self.glass = mixer.Sound("GLASS_1.wav")
        self.glass2 = mixer.Sound("GLASS_2.wav")
        #background music
        self.hulk = mixer.Sound("hulk2.wav")
        
    def PlaySound(self, sound_num):
        if sound_num == 0:
            self.clap.play()    
        if sound_num == 1:
            self.clap2.play()    
        if sound_num == 2:
            self.kick.play()     
        if sound_num == 3:
            self.kick.play()      
        if sound_num == 4:
            self.glass.play()     
        if sound_num == 5:
            self.glass2.play()     
        if sound_num == 6:
            self.drop.play()     
        if sound_num == 7:
            self.scratch.play()
        if sound_num == 8:
            self.sprinkle.play()
        #background music
        if sound_num == 10:
            self.hulk.play()
            
mMusicPlayer = MusicPlayer()

def PlayLoop(sound):
    threading.Timer(207, PlayLoop, [sound]).start()
    mMusicPlayer.PlaySound(sound)
    print ("Playing...")
    

PlayLoop(10)

class CameraClick(FloatLayout):
    
# Video file part
    
    recording = False
    background_music = False
    current_filter = "hulk.jpg"

    def toggle_Recording(self):
        self.recording = not self.recording
        print("toggle recording")
        
    def toggle_Background_Music(self):
        self.background_music = not self.background_music
        print("toggle background music")
            
    def change_filter(self, id):
        self.current_filter = id
    
    def stop(self):
        camera = cv2.VideoCapture(0)
        frame_width = int(camera.get(3))
        frame_height = int(camera.get(4))
        out = cv2.VideoWriter('x.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 35, (frame_width,frame_height))
        camera.release()
        out.release()
        cv2.destroyAllWindows()
        
    def close(self):
        cv2.destroyAllWindows()
        exit(0)
        
    def play_background_music(self):
        print("ADD BACKGROUND MUSIC")
        
    # Video cam part
    def warp_face_from_webcam(self):
        
        previousNum = 0
        newNumCount = 0
        
        video_out_fn = './demo/demo_arni.mov'
        facial_mask_fn = './demo/' + self.current_filter
        """
        Function to read video frames from the web cam, replace first found face by the face from the still image
        and show processed frames in a window. Also all processed frames will be save as a video.
    
        :param facial_mask_fn: path to the still image with a face
        :param video_out_fn: path to the video file which will have 'replaced' face
        """
    
        #facial_mask = cv2.cvtColor(cv2.imread(facial_mask_fn), cv2.COLOR_HSV2RGB)
        facial_mask = cv2.imread(facial_mask_fn)
        facial_mask_lm = faceWarp.find_landmarks(facial_mask, faceWarp.predictor)
    
        cam = cv2.VideoCapture(0)
        frame_size = (640, 480) # downsample size, without downsampling too many frames dropped
        
        frame_width = int(cam.get(3))
        frame_height = int(cam.get(4))
     
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #out = cv2.VideoWriter('a.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
     
        video_out = cv2.VideoWriter(
            filename=video_out_fn,
            fourcc=cv2.VideoWriter_fourcc('m', '2', 'v', '1'), # works good on OSX, for other OS maybe try other codecs
            frameSize=frame_size,
            fps=50.0,
            isColor=True)
        
        def nothing():
            pass
        cv2.namedWindow('Sliders')
        cv2.createTrackbar("erode", "Sliders", 2,10,nothing)
        cv2.createTrackbar("thresh", "Sliders", 70,249,nothing)

        cv2.createTrackbar('R','Sliders',0,255,nothing)
        cv2.createTrackbar('G','Sliders',90,255,nothing)
        cv2.createTrackbar('B','Sliders',0,255,nothing)
        while True:
            erode = cv2.getTrackbarPos("erode", "Sliders")
            thresh = cv2.getTrackbarPos("thresh", "Sliders")

            r = cv2.getTrackbarPos("R", "Sliders")
            g = cv2.getTrackbarPos("G", "Sliders")
            b = cv2.getTrackbarPos("B", "Sliders")

            ret, frame_in = cam.read()
            frame_in=cv2.flip(frame_in,1)
            kernel = np.ones((3,3),np.uint8)
            roi=frame_in[0:900, 0:900]
            
            #cv2.rectangle(frame_in,(0,0),(900,900),(0,255,0),0)    
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
            lower_skin = np.array([r,g,b], dtype=np.uint8)
            upper_skin = np.array([20,255,255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            mask = cv2.erode(mask,kernel,iterations = erode)

            mask = cv2.GaussianBlur(mask,(5,5),200) 

            ret,thresh = cv2.threshold(mask, thresh, 255, 0)

            mask = cv2.resize(mask, (80, 60))
        #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mask', 320,240)
            cv2.moveWindow('mask', 980,0)
            cv2.imshow('mask',mask)
            mask = mask.astype("float") / 255.0
            mask = img_to_array(mask)
            mask = np.expand_dims(mask, axis=0)
            
            (none, dabL, dabR, hitL, hitR, whipL, whipR, tPose, shoot, keke) = model.predict(mask)[0]
            values = [none, dabL, dabR, hitL, hitR, whipL, whipR, tPose, shoot, keke]
            currentNum = values.index(max(values))

            if(currentNum == previousNum):
                if(newNumCount < 7):
                    newNumCount = newNumCount + 1
                print (str(newNumCount) + " occurences of " + str(currentNum))
            else:
                newNumCount = 0
                previousNum = currentNum
                print ("New number: " + str(currentNum) + ", " + str(previousNum))
        
            if(newNumCount == 3):
                print ("Play")
                if currentNum == 1:
                    mMusicPlayer.PlaySound(0)
                if currentNum == 2:
                    mMusicPlayer.PlaySound(1)
                if currentNum == 3:
                    mMusicPlayer.PlaySound(2)
                if currentNum == 4:
                    mMusicPlayer.PlaySound(3)
                if currentNum == 5:
                    mMusicPlayer.PlaySound(4)
                if currentNum == 6:
                    mMusicPlayer.PlaySound(5)
                if currentNum == 7:
                    mMusicPlayer.PlaySound(6)
                if currentNum == 8:
                    mMusicPlayer.PlaySound(7)
                if currentNum == 9:
                    mMusicPlayer.PlaySound(8)
        
            # Downsample frame - otherwise processing is too slow
            frame_in = cv2.resize(frame_in, dsize=frame_size)
            #frame_in = cv2.cvtColor(frame_in, cv2.COLOR_HSV2RGB)
            frame_out = faceWarp.face_warp(facial_mask, facial_mask_lm, frame_in)
            if self.recording:
                video_out.write(frame_out)
            cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Recording', 700,525)
            cv2.moveWindow("Recording", 280,0)
            cv2.imshow('Recording', frame_out)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if ch == ord(' '):
                break

        cam.release()
        video_out.release()
        cv2.destroyAllWindows()
        
    def start(self):
        if self.background_music:
            self.play_background_music()
        self.warp_face_from_webcam()
    
    
class TestCamera(App):
    def build(self):
        return CameraClick()   

if __name__ == "__main__":
    # Let's parse running arguments and decide what we will do
    parser = argparse.ArgumentParser(description='Warp a still-image face around the other face in a video.')
    parser.add_argument('stillface', metavar='STILLFACE',
                        help='the full path to jpg file with face', default='./flash.jpg')
    parser.add_argument('inputvideo', metavar='VIDEOIN',
                        help='the full path to input video file where face will be changed. If "0" is provided \
                        then the web cam will be used', default='0')
    parser.add_argument('outputvideo', metavar='VIDEOOUT',
                        help='the full path to output video file with the new face. If "0" is provided then \
                        process video will be shown on the screen, but not saved. (.MOV format)',default='0')
    args = parser.parse_args()
    args.inputvideo = '0'
    try:
        print('*** Start webcam to save to file: {} ***'.format(args.outputvideo))
        TestCamera().run()
        print('\n*** Done! ***')
    except:
        print('*** Something went wrong. Error: {} ***'.format(sys.exc_info()))





 



