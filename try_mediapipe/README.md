# attempt to create virtual keyboard 
(I've exclude ipynb that was used to experiment)

lastest version is 12.py, now they can track hand and create plane that fit most of your hand landmark to use those plane as virtual keyboard. (when you open your palm of your hand)

a lot of problem to be solved, including 
- hand depth (z axis) inconsistant, trying to solved with depth estimation and hand constraint


## preview
left image is origin (input) image\
right image is 3d hand and plane plotted\
title of right image is r square of (plane) linear regression 

![image](https://user-images.githubusercontent.com/65396522/236679625-690459b2-6156-474e-8ebe-3c4e205073c9.png)\
![image](https://user-images.githubusercontent.com/65396522/236679631-fc6433aa-f96c-436f-951a-abea91d91042.png)
<!--
![image](https://user-images.githubusercontent.com/65396522/236679632-2d90f866-85a2-43d9-a944-4a57aec2042c.png)
![image](https://user-images.githubusercontent.com/65396522/236679635-c7a9288c-68e1-456f-a9df-1df6307721d3.png)-->
