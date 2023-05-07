import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing Libraries
import cv2
import mediapipe as mp
from math import hypot
# import screen_brightness_control as sbc
import numpy as np
# from IPython import display
import matplotlib.pyplot as plt 
import custom_drawing_utils
import ipywidgets as widgets

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,
	model_complexity=1,
	min_detection_confidence=0.75,
	min_tracking_confidence=0.75,
	max_num_hands=2)

Draw = custom_drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
from matplotlib.widgets import Button

fig = plt.figure(figsize=(12,7))
ax_cam = fig.add_subplot(151)
ax = fig.add_subplot(152,projection='3d')
ax_button = fig.add_subplot(153)
capture_button = Button(ax_button, 'capture')
ax_cap = fig.add_subplot(154)
plt.ion()

fig.show()
fig.canvas.draw()
def convertLandmarkToList(landmarks):
	XYZ = []
	for _id, landmark in enumerate(landmarks):
		XYZ .append ([landmark.x,landmark.y,landmark.z])
	return np.array([*zip(*XYZ)])
xmin,xmax,ymin,ymax,zmin,zmax = [1,-1]*3
def update_xyzminmax(x,y,z):
	global xmin,xmax,ymin,ymax,zmin,zmax
	xmin = min(xmin,x.min())
	xmax = max(xmax,x.max())
	ymin = min(ymin,y.min())
	ymax = max(ymax,y.max())
	zmin = min(zmin,z.min())
	zmax = max(zmax,z.max())

def get_plot_surface(X, Y, Z):
	def plot_surface(ax):
		ax.plot_surface(-X, Y, -Z, alpha=.1)
	return plot_surface
plot_surface_func = None
from traitlets import CBool
class To_capture(widgets.DOMWidget):
	value = CBool(False, sync=True)
	# value = False
	
saved_frame = None
to_capture = To_capture()

def set_to_capture(*a,**k):
	# global to_capture
	to_capture.value = True

# button1.on_click(set_to_capture)
capture_button.on_clicked(set_to_capture)

while True:    
	# Read video frame by frame
	ret, frame = cap.read()

	
	if not ret:
		print("failed to grab frame")
		break

	# # Flip image
	frame = cv2.flip(frame, 1)

	# Convert BGR image to RGB image
	frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	ax_cam.clear()
	ax_cam.imshow(frameRGB)

	# Process the RGB image
	Process = hands.process(frameRGB)

	# display.clear_output(wait=True)
	ax.clear()
	
	ax.set_xlim3d(xmin, xmax)
	ax.set_ylim3d(ymin, ymax)
	ax.set_zlim3d(zmin, zmax)
	
	if 1 and Process.multi_hand_landmarks:
		# landmark = Process.multi_hand_landmarks[-1]
		landmark = Process.multi_hand_world_landmarks[-1]
		Draw.plot_landmarks(
			landmark,
			mpHands.HAND_CONNECTIONS,
			ax=ax
			)
		XYZ = convertLandmarkToList(landmark.landmark)
		# Define the data
		x,y,z = XYZ
		update_xyzminmax(x,y,z)
		X = np.stack((x, y), axis=-1)
		y = z

		# Fit the linear regression plane
		line_reg_model = LinearRegression()
		reg = line_reg_model.fit(X, y)

		# Get the coefficients and intercept
		a, b = reg.coef_
		c = reg.intercept_ 
		r2 = r2_score(z,line_reg_model.predict(X),)

		ax.title.set_text(f'{r2}')

		ax.text(xmin,ymin,zmin,f'{r2}', fontdict=None)

		
		if to_capture.value and r2>.8 :
			saved_frame = frameRGB.copy()
			to_capture.value = False
			print('captured')
			
		if 1:
			x = np.linspace(xmin, xmax,10)
			y = np.linspace(ymin, ymax,10)

			X,Y = np.meshgrid(x,y)
			
			Z=a*X + b*Y + c
			if r2>.8 or plot_surface_func==None:
				plot_surface_func = get_plot_surface(X,Y,Z)
			surf = plot_surface_func(ax)

	
	if saved_frame is not None:ax_cap.imshow(saved_frame)

	plt.pause(0.1)


cap.release()