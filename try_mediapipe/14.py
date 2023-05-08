
# Importing Libraries

import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import cv2
from math import hypot
import numpy as np
import matplotlib.pyplot as plt 
import custom_drawing_utils
Draw = custom_drawing_utils
import mediapipe as mp
from matplotlib.widgets import Button
from traitlets import CBool

# define class
class Plane():
	def __init__(self):
		self.model = None
	def fit(self,x,y,z):
		self.x,self.y,self.z = x,y,z
		X = np.stack((self.x, self.y), axis=-1)
		Y = self.z
		self.model = LinearRegression()
		self.reg = self.model.fit(X, Y)
	@property
	def a(self):
		a, b = self.reg.coef_
		return a
	@property
	def b(self):
		a, b = self.reg.coef_
		return b
	@property
	def c(self):
		c = self.reg.intercept_ 
		return c
	@property
	def r2(self):
		X = np.stack((self.x, self.y), axis=-1)
		r2 = r2_score(self.z,self.model.predict(X),)
		return r2
	def raw_dist(self,x,y,z):
		# a*X + b*Y - Z + c = 0
		#  |Axo + Byo+ Czo + D|/√(A2 + B2 + C2)
		A,B,C,D = self.a,self.b,-1,self.c
		dist = A*x+B*y+C*z+D / (A**2+B**2+C**2)**.5
		return dist
	def dist(self,x,y,z):
		return np.abs(self.raw_dist(x,y,z))
	def side(self,x,y,z):
		return (self.raw_dist(x,y,z) > 0 ) 
	def get_Z(self,x,y):
		X = np.stack((x, y), axis=-1)
		return self.model.predict(X)
		
class To_capture():
	def __init__(self):
		self._value = False
		self._r2 = .9
		self._t = time.time()
	@property
	def value(self):return self._value
	@value.setter
	def value(self,value):
		if value == True:
			self._r2 = .95
		self._value = value
		self._t = time.time()
	@property
	def r2(self):return max(self._r2-.005*self.t,.8)
	@property
	def t(self):return time.time()-self._t

# define custom function
def custom_norm(x):
	return x
	return x/x.max()*100
def convertLandmarkToList(landmarks):
	XYZ = []
	for _id, landmark in enumerate(landmarks):
		XYZ .append ([landmark.x,landmark.y,landmark.z])
	return np.array([*zip(*XYZ)])

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,
	model_complexity=1,
	min_detection_confidence=0.75,
	min_tracking_confidence=0.75,
	max_num_hands=2)


# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# create plot
fig = plt.figure(figsize=(12,7))
ax_cam = fig.add_subplot(231)
ax = fig.add_subplot(232,projection='3d')
ax_skeleton_dist_cam = fig.add_subplot(233)
ax_point_dist_cam = fig.add_subplot(236)
ax_button = fig.add_subplot(235)
capture_button = Button(ax_button, 'capture')
ax_cap = fig.add_subplot(234)
plt.ion()
fig.show()
fig.canvas.draw()

xmin,xmax,ymin,ymax,zmin,zmax = [10,-10]*3
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
		ax.plot_surface(X, Y, Z, alpha=.1)
	return plot_surface
plot_surface_func = None
	
saved_frame = None
to_capture = To_capture()

def set_to_capture(*a,**k):
	# global to_capture
	to_capture.value = True

# button1.on_click(set_to_capture)
capture_button.on_clicked(set_to_capture)

plane = None
start_flag = True

# while not close 
while plt.fignum_exists(fig.number):    
	# Read video frame by frame
	ret, frame = cap.read()
	if not ret:
		print("failed to grab frame")
		break

	# Flip image
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
		landmarks = Process.multi_hand_world_landmarks
		landmarks = Process.multi_hand_landmarks

		# convert landmark to list of x,y,z
		XYZ = np.concatenate([convertLandmarkToList(landmark.landmark) for landmark in landmarks], axis = 1)
		x,y,z = XYZ
		
		# update 3d plot and boundary
		update_xyzminmax(-z,x,-y)
		for landmark in landmarks:
			Draw.plot_landmarks(
				landmark,
				mpHands.HAND_CONNECTIONS,
				ax=ax
				)
		
		# create current plane from input data
		current_plane  = Plane()
		current_plane.fit(x, y, z)

		# show r2
		r2 = current_plane.r2
		ax.title.set_text(f'{r2}')
		if start_flag or ( to_capture.value and r2>to_capture.r2 ) :
			if start_flag : to_capture.value = True
			else: to_capture.value = False

			# assign plane with current plane 
			plane = current_plane

			# create function to plot plane
			Xs = np.linspace(xmin, xmax,10)
			Ys = np.linspace(ymin, ymax,10)
			X,Y = np.meshgrid(Xs,Ys)
			Z=plane.a*X + plane.b*Y + plane.c
			X,Y,Z = -Z,X,-Y,
			plot_surface_func = get_plot_surface(X,Y,Z)
			
			# save and show captured image
			saved_frame = frameRGB.copy()
			ax_cap.title.set_text(f'{r2}')
			ax_cap.clear()
			ax_cap.imshow(saved_frame)
			
		# plot all hand landmarks and skeleton
		ax_skeleton_dist_cam.clear()
		for handlm in landmarks:
			Draw.draw_landmarks_plt( handlm,
								mpHands.HAND_CONNECTIONS, ax = ax_skeleton_dist_cam)
		ax_skeleton_dist_cam.scatter(x,-y,c = plane.side(x, y, z))

		# plot hand landmarks that go through plane
		ax_point_dist_cam.clear()
		ax_point_dist_cam.imshow(frameRGB)
		dist = custom_norm(plane.dist(x, y, z))*plane.side(x, y, z)
		ax_point_dist_cam.scatter(x*frameRGB.shape[1],y*frameRGB.shape[0],c=dist)

		# plot plane
		surf = plot_surface_func(ax)
		
		start_flag = False

	plt.pause(0.5)

cap.release()