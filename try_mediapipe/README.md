# attempt to create virtual keyboard 
(I've excluded ipynb that was used to experiment)

lastest version is 14.py, now it can 
- determine which finger is most prominently pointed
- track the position of your hand to create a plane that fits your hand's landmarks when the capture button is clicked

Clockwise from top left:
- Input: the current image being processed
- The hand's 3D skeleton and plotted landmarks with r2 as title
- The hand's 2D skeleton and plotted landmarks
- Input image and distance from camera plotted as point color, with brighter colors indicating closer proximity
- Capture button, used to create a plane by clicking it
- Captured image, which was used to create the plane

### ring finger up
![image](https://user-images.githubusercontent.com/65396522/236886311-e9bb65b9-2025-4096-bace-96106f4b04c0.png)
### index finger up
![image](https://user-images.githubusercontent.com/65396522/236886316-3e492bc2-a494-44c5-b00c-dadd5845c379.png)
### thumb up
![image](https://user-images.githubusercontent.com/65396522/236885833-6c87281a-3675-4ea1-be0f-c868fe93dac6.png)
### index finger up
![image](https://user-images.githubusercontent.com/65396522/236885547-8bfd1ab0-1640-4647-9931-a71ed7dffe00.png)

<!-- 
a lot of problem to be solved, including 
- hand depth (z axis) inconsistant, trying to solved with depth estimation and hand constraint


## preview
left image is origin (input) image\
right image is 3d hand and plane plotted\
title of right image is r square of (plane) linear regression 

![image](https://user-images.githubusercontent.com/65396522/236679625-690459b2-6156-474e-8ebe-3c4e205073c9.png)\
![image](https://user-images.githubusercontent.com/65396522/236679631-fc6433aa-f96c-436f-951a-abea91d91042.png) -->
<!--
![image](https://user-images.githubusercontent.com/65396522/236679632-2d90f866-85a2-43d9-a944-4a57aec2042c.png)
![image](https://user-images.githubusercontent.com/65396522/236679635-c7a9288c-68e1-456f-a9df-1df6307721d3.png)-->
