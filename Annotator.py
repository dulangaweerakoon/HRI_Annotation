#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2
from IPython.display import HTML
import os
#from IPython import display
#from bokeh.plotting import figure
#from bokeh.io import output_notebook, show, push_notebook
from matplotlib import pyplot as plt
import time
#get_ipython().run_line_magic('matplotlib', 'inline')

import io
import base64
#%matplotlib nbagg

#from IPython.core.display import Video
#from IPython.display import HTML
import pyglet
import vlc
import subprocess

from tkinter import messagebox
from tkinter import simpledialog
from shutil import copyfile
from shutil import rmtree

import tkinter as tk
root = tk.Tk()
root.withdraw()
import tarfile




# In[15]:


root = "Dulanga/"
parts = [os.path.join(root,dI) for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI)) and "Instructor" in dI]
print(parts)





import pandas as pd
import numpy as np
#df = pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
# Show in Jupyter
#df = pd.read_csv('csv_dir/train.csv')
df = pd.DataFrame(columns=['img_id','bbox','query'])
done_df = pd.DataFrame(columns=['path'])
img_id = 0
#done_df  =  pd.read_csv('done.csv')
#df.to_csv()



class ExtractImageWidget(object):
    def __init__(self,img,index,ins,depth_path):
        self.original_image = img
        self.index = index
        self.depth_path = depth_path

        #self.img_id = img_id 
        self.ins = ins

        # Resize image, remove if you want raw image size
        #self.original_image = cv2.resize(self.original_image, (640, 556))
        self.clone = self.original_image.copy()

        cv2.namedWindow('Frame '+str(self.index))
        cv2.setMouseCallback('Frame '+str(self.index), self.extract_coordinates)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        global df
        global img_id
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)
            img_path = '{0:06d}'.format(img_id)+".jpg"
            cv2.imwrite('images/'+img_path,self.original_image)
            bbox = [self.image_coordinates[0][0],self.image_coordinates[0][1],self.image_coordinates[1][0],self.image_coordinates[1][1]]
            df = df.append({'img_id':img_path,'bbox':str(bbox), 'query':self.ins},ignore_index=True)
            depth_files = [dI for dI in os.listdir(self.depth_path)]
            os.mkdir("depth/"+str(img_id))
            for file_ in depth_files:
            	copyfile(os.path.join(self.depth_path,file_),"depth/"+str(img_id)+"/"+file_)
            img_id+=1
            cv2.imshow('Frame '+str(self.index), self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone


for part in parts:
    cfg_paths = [dI for dI in os.listdir(part) if os.path.isdir(os.path.join(part,dI))]
    for cfg_path in cfg_paths:
       
        vid_path=os.path.join(part,cfg_path)+"/"+cfg_path+".mp4"
        tar_source_path = os.path.join(part,cfg_path)+"/short_throw_depth.tar"
        tar_dest_path = os.path.join(part,cfg_path)+"/short_throw_depth/" 
        tar = tarfile.open(tar_source_path)
        tar.extractall(path=tar_dest_path)
        tar.close()
        depth_paths = [os.path.join(tar_dest_path,dI) for dI in os.listdir(tar_dest_path)]
        depth_paths.sort()
        print(depth_paths)
        print(vid_path)
        if (len(done_df[done_df['path']==vid_path].index.tolist())!=0):
        	print("Already Annotated!")
        	continue
        #messagebox.showinfo("Annotation","Annotating Video: "+vid_path)
        cap = cv2.VideoCapture(vid_path) 
 
 
        # Instance = vlc.Instance('--fullscreen')
        # player = Instance.media_player_new()
        # Media = Instance.media_new(vid_path)
        # Media.get_mrl()
        # player.set_media(Media)
        # player.play()

        # time.sleep(5) # Or however long you expect it to take to open vlc
        # while player.is_playing():
        #     time.sleep(1)
        subprocess.call(['vlc',vid_path,'--play-and-exit'])
        answer = simpledialog.askstring("Transcribe", "What did the instructor say?")
        print(answer)
        ret = True
        i = 1
        while(ret):
            # Capture frame-by-frame
            ret, frame = cap.read()


            # Our operations on the frame come here
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret==False:
            	continue
            # Display the resulting frame
            #cv2.imshow("Frame",frame)
            #img_id +=1 
            roi_widget = ExtractImageWidget(frame,i,answer,os.path.join(part,cfg_path)+"/short_throw_depth")
            cv2.imshow('Frame '+str(i), roi_widget.show_image())
            key = cv2.waitKey(0)
            if key == ord('q'):
            	done_df.to_csv('done.csv',index=False)
            	df.to_csv('csv_dir/train.csv',index=False)
            	rmtree(os.path.join(part,cfg_path)+"/short_throw_depth")
            	exit()
            if key == ord('n'):
            	cv2.destroyAllWindows()
            	break
            cv2.destroyAllWindows()
            i+=1

            #display.display(plt.gcf())
            #display.clear_output(wait=True)
            #time.sleep(0.001)
            #plt.pause(0.1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        #print("Here")
        done_df = done_df.append({'path':vid_path},ignore_index=True)
        rmtree(os.path.join(part,cfg_path)+"/short_throw_depth")

# When everything done, release the capture
done_df.to_csv('done.csv',index=False)
df.to_csv('csv_dir/train.csv',index=False)
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




