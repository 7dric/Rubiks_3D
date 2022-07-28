# 7dric's Rubik's Cube hand controlled

## Features of the program:
This program display a Rubik's Cube wich can be controlled by using your **LEFT Hand** in front of your webcam. 
- **Rotation of the cube** : moove your hand up and down for an X axis rotation and Left or Right for an Y axis Rotation (There is no depht control on the Z axis). 
- **Rotation of a face** : use either your keyboard with the **LEFT/RIGHT arrows** or either place your **thumb/pinky in the palm of your LEFT hand**
- **mixing the rubiks** : press the **"m" key**
- **solving the rubiks** : press the **"s" key**
- **reset the rubiks** : press the **"r" key**

## Libraries :
To be able to run this program you need to have thoose python libraies on your computer: 

- pygame
- OpenGL
- OpenCV
- mediapipe
- numpy 
- rubik_solver
- threading


## How I ended up with a rubik's:
At the beginning I just wanted to do some image processing with OpenCV, then I discover the hand gesture detector on the Mediapipe library and wanted to find a small and fun project where I can use this technology. So I've started by animating a simple cube with OpenGL by controlling it with my hand. Then Guillaume Mevisse told me that he wanted to do the same but with a rubik's cube so I steal his idea. 
