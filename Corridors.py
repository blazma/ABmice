# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: bbujfalussy - ubalazs317@gmail.com
A framework for storing virtual corridor properties for behavioral experiments
We define a class - Corridor - and the collector class.

"""

import numpy as np
from string import *
from sys import version_info
import os
import pickle

class Corridor:
	'defining properties of a single corridor'
	def __init__(self, name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width=470, reward='Right', length=7168, height=768, width=1024):
		self.name = name

		self.left_image = left_image
		self.right_image = right_image
		self.end_image = end_image
		self.floor_image = floor_image
		self.ceiling_image = ceiling_image

		self.length = length
		self.height = height
		self.width = width
		section_length = float(self.length - self.width)

		# zone_shift = 0.0233 # Rita wants the zones to start at the near-edge of the monitor
		self.reward = reward # currently all zones in a given corridor are identical. In the future, we could have a vector for encoding different zone properties
		self.N_zones = len(reward_zone_starts)
		zone_shift = 0 ## 0.05 for Rita
		self.reward_zone_starts = np.array(reward_zone_starts) / section_length + zone_shift # relative position of reward zone starts [0, 1]
		self.reward_zone_ends = self.reward_zone_starts + zone_width / section_length

		if (self.N_zones > 0):
			for i in np.arange(self.N_zones):
				if self.reward_zone_starts[i] < 0:
					self.reward_zone_starts[i] = 0
				if self.reward_zone_starts[i] > 1:
					self.reward_zone_starts[i] = 1
				if self.reward_zone_ends[i] < self.reward_zone_starts[i]:
					self.reward_zone_ends[i] = self.reward_zone_starts[i]
				if self.reward_zone_ends[i] > 1:
					self.reward_zone_ends[i] = 1

	def print_images(self):
		s = self.name + ': left: ' + self.left_image + ', right: ' + self.right_image
		# ss =  'end: ' + self.end_image + ', floor: ' + self.floor_image + ', ceiling: ' + self.ceiling_image
		print(s)
		# print(ss)

	def print_zones(self):
		s = self.name + ' number of reward zones: ' + str(self.N_zones) + ' ' + ', reward zone starts: ' + str(self.reward_zone_starts) + ', zone ends:' + str(self.reward_zone_ends)
		print(s)

class Corridor_list:
	'class for storing corridor properties'
	def __init__(self, image_path, experiment_name):
		self.image_path = image_path
		self.num_VRs = 0
		self.name = experiment_name
		self.corridors = []

	def add_corridor(self, name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width=470, reward='Right', length=7168, height=768, width=1024):
		self.corridors.append(Corridor(name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width, reward, length, height, width))
		self.num_VRs = self.num_VRs + 1

	def print_images(self):
		for i in range(self.num_VRs):
			self.corridors[i].print_images()

	def print_zones(self):
		for i in range(self.num_VRs):
			self.corridors[i].print_zones()

	def write(self):
		fname = self.image_path + '/' + self.name + '_corridors.pkl'
		f = open(fname, 'wb')
		pickle.dump(self, f)
		f.close()

# # # cc = Corridor('RN_1_cheese_left.png', 'RN_1_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5380])
# # # Cors.add_corridor('RN_1_cheese_left.png', 'RN_1_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5380])

###################################################################
## default corridor sizes 1:6 ratio
###################################################################
##			pixel
## length: 7168 = 6144 + 1024
## width: 1024
## height: 768

###################################################################
## long corridor sizes 1:9 ratio
###################################################################
##			pixel
## length: 10240 = 9216 + 1024 
## width: 1024
## height: 768




###################################################################
## corridors for Rita
######################################################################################################

## corridor number 		left image 			right image 		number of zones 		reward side
## 0					grey_wall			grey_wall			0						None
## 1					9 cheese			9 cheese			9						Right	
## 2					7 cheese			7 cheese			7						Right	
## 3					7 cheese			7 cheese			7						Right	
## 4					7 cheese			7 cheese			7						Right	
## 5					5 cheese			5 cheese			5						Right	
## 6					5 cheese			5 cheese			5						Right	
## 7					5 cheese			5 cheese			5						Right	
## 8					3 cheese			3 cheese			3						Right	
## 9					3 cheese			3 cheese			3						Right	
## 10					3 cheese			3 cheese			3						Right	
## 11					1 cheese			1 cheese			3						Right	

## 12					purple striped		purple striped		1						Left	
## 13					purple striped		purple striped		1						Right	
## 14					purple squared		purple squared		1						Left	
## 15					purple squared		purple squared		1						Right	
## 16					green striped		green striped		1						Left	
## 17					green striped		green striped		1						Right	
## 18					green squared		green squared		1						Left	
## 19					green squared		green squared		1						Right	


# Cors = Corridor_list('./', 'contingency_learning')

# Cors.add_corridor('grey', 'grey_wall.png', 'grey_wall.png', 'grey_end_wall.png', 'grey_floor_ceiling.png', 'grey_floor_ceiling.png', [], reward='None')

# Cors.add_corridor('9_cheese', 'RN_9_cheese_left.png', 'RN_9_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 820, 1500, 2180, 2860, 3540, 4220, 4900, 5580])

# Cors.add_corridor('7_cheese_a', 'RN_7_cheese_left_a.png', 'RN_7_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 1140, 1940, 2740, 3940, 4740, 5500])
# Cors.add_corridor('7_cheese_b', 'RN_7_cheese_left_b.png', 'RN_7_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 940, 1740, 2540, 3340, 4140, 5500])
# Cors.add_corridor('7_cheese_c', 'RN_7_cheese_left_c.png', 'RN_7_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [340, 1540, 2340, 3390, 4090, 4790, 5500])

# Cors.add_corridor('5_cheese_a', 'RN_5_cheese_left_a.png', 'RN_5_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 1740, 2540, 4140, 5500])
# Cors.add_corridor('5_cheese_b', 'RN_5_cheese_left_b.png', 'RN_5_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 1340, 3140, 4640, 5500])
# Cors.add_corridor('5_cheese_c', 'RN_5_cheese_left_c.png', 'RN_5_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1140, 2740, 3940, 4740, 5500])

# Cors.add_corridor('3_cheese_a', 'RN_3_cheese_left_a.png', 'RN_3_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1140, 2140, 5250])
# Cors.add_corridor('3_cheese_b', 'RN_3_cheese_left_b.png', 'RN_3_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 3640, 5250])
# Cors.add_corridor('3_cheese_c', 'RN_3_cheese_left_c.png', 'RN_3_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1740, 4140, 5250])

# Cors.add_corridor('1_cheese', 'RN_1_cheese_left.png', 'RN_1_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])

# Cors.add_corridor('purple_striped_left', 'RN_wall_striped_purple_left_002.png', 'RN_wall_striped_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250], reward='Left')
# Cors.add_corridor('purple_striped_right', 'RN_wall_striped_purple_left_002.png', 'RN_wall_striped_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])
# Cors.add_corridor('purple_squared_left', 'RN_wall_squared_purple_left_002.png', 'RN_wall_squared_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250], reward='Left')
# Cors.add_corridor('purple_squared_right', 'RN_wall_squared_purple_left_002.png', 'RN_wall_squared_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])

# Cors.add_corridor('green_striped_left', 'RN_wall_striped_green_left_002.png', 'RN_wall_striped_green_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250], reward='Left')
# Cors.add_corridor('green_striped_right', 'RN_wall_striped_green_left_002.png', 'RN_wall_striped_green_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])
# Cors.add_corridor('green_squared_left', 'RN_wall_squared_green_left_002.png', 'RN_wall_squared_green_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250], reward='Left')
# Cors.add_corridor('green_squared_right', 'RN_wall_squared_green_left_002.png', 'RN_wall_squared_green_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])


# Cors.print_zones()
# Cors.print_images()

# Cors.write()

# input_path = './contingency_learning_corridors.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	if version_info.major == 2:
# 		corridors_list = pickle.load(input_file)
# 	elif version_info.major == 3:
# 		corridors_list = pickle.load(input_file, encoding='latin1')
# 	input_file.close()


# ######################################################################################################
# ## Corridors for Snezana
# ######################################################################################################
# ## corridor number 		left image 			right image 		number of zones 		reward side
# ## 0					grey_wall			grey_wall			0						None
# ## 1					9 cheese			9 cheese			9						Right	
# ## 2					7 cheese			7 cheese			7						Right	
# ## 3					7 cheese			7 cheese			7						Right	
# ## 4					7 cheese			7 cheese			7						Right	
# ## 5					5 cheese			5 cheese			5						Right	
# ## 6					5 cheese			5 cheese			5						Right	
# ## 7					5 cheese			5 cheese			5						Right	
# ## 8					3 cheese			3 cheese			3						Right	
# ## 9					3 cheese			3 cheese			3						Right	
# ## 10					3 cheese			3 cheese			3						Right	
# ## 11					1 cheese			1 cheese			3						Right	
# ## 12					green squared		green squared		1						Right	
# ## 13					green squared near	green squared near	1						Right	
# ## 14					purple striped		purple striped		1						Right	
# ## 15					purple striped near	purple striped near	1						Right	
# ## 16					purple squared 		purple squared 		1 						Right

# Cors = Corridor_list('./', 'TwoMazes')

# Cors.add_corridor('grey', 'grey_wall.png', 'grey_wall.png', 'grey_end_wall.png', 'grey_floor_ceiling.png', 'grey_floor_ceiling.png', [], reward='None')

# Cors.add_corridor('9_cheese', 'RN_9_cheese_left.png', 'RN_9_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [0, 680, 1360, 2040, 2720, 3400, 4080, 4760, 5440])

# Cors.add_corridor('7_cheese_a', 'RN_7_cheese_left_a.png', 'RN_7_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [0, 1000, 1940, 2740, 3940, 4740, 5500])
# Cors.add_corridor('7_cheese_b', 'RN_7_cheese_left_b.png', 'RN_7_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [0, 940, 1740, 2540, 3340, 4140, 5500])
# Cors.add_corridor('7_cheese_c', 'RN_7_cheese_left_c.png', 'RN_7_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [340, 1540, 2340, 3390, 4090, 4790, 5500])

# Cors.add_corridor('5_cheese_a', 'RN_5_cheese_left_a.png', 'RN_5_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 1740, 2540, 4140, 5500])
# Cors.add_corridor('5_cheese_b', 'RN_5_cheese_left_b.png', 'RN_5_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [0, 1340, 3140, 4640, 5500])
# Cors.add_corridor('5_cheese_c', 'RN_5_cheese_left_c.png', 'RN_5_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1000, 2740, 3940, 4740, 5500])

# Cors.add_corridor('3_cheese_a', 'RN_3_cheese_left_a.png', 'RN_3_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1000, 2140, 5500])
# Cors.add_corridor('3_cheese_b', 'RN_3_cheese_left_b.png', 'RN_3_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 3640, 5500])
# Cors.add_corridor('3_cheese_c', 'RN_3_cheese_left_c.png', 'RN_3_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1740, 4140, 5500])

# Cors.add_corridor('1_cheese', 'RN_1_cheese_left.png', 'RN_1_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5500])

# Cors.add_corridor('green_squared_right', 'RN_wall_squared_green_left_002.png', 'RN_wall_squared_green_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5275])
# Cors.add_corridor('green_squared_right', 'SR_wall_squared_green_left_001.png', 'SR_wall_squared_green_right_001.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [3890])

# Cors.add_corridor('purple_striped_right', 'RN_wall_striped_purple_left_002.png', 'RN_wall_striped_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5275])
# Cors.add_corridor('purple_striped_right', 'SR_wall_striped_purple_left_001.png', 'SR_wall_striped_purple_right_001.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [2682])

# Cors.add_corridor('purple_squared_right', 'RN_wall_squared_purple_left_002.png', 'RN_wall_squared_purple_right_002.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5275])


# Cors.print_zones()
# Cors.print_images()

# Cors.write()

# # # input_path = './TwoMazes_corridors.pkl'
# # # if (os.path.exists(input_path)):
# # # 	input_file = open(input_path, 'rb')
# # # 	corridors_list = pickle.load(input_file)
# # # 	input_file.close()


######################################################################################################
## Corridors for Imola
######################################################################################################
## corridor number 		end image 	floor image 	left image 			right image 		number of zones 		reward side
## 0					grey 		grey 			grey_wall			grey_wall			0						None
## 1					sine 		sine 			grey_wall			grey_wall			12						Right	
## 2					square 		square 			grey_wall			grey_wall			12						Right	
## 3					sine 		sine 			patterns			patterns			1						Right	
## 4					square 		square 			patterns			patterns			1						Right	
## 5					sine 		sine 			patterns			patterns			1						Right	
## 6					square 		square 			patterns			patterns			1						Right	
## 7					sine 		sine 			patterns			patterns			1						Right	

# Cors = Corridor_list('./', 'NearFar')
# #						 left_image, right_image, end_image, floor_image, ceiling_image
# Cors.add_corridor('grey', 'grey_wall.png', 'grey_wall.png', 'grey_end_wall.png', 'grey_floor_ceiling.png', 'grey_floor_ceiling.png', [], reward='None')

# Cors.add_corridor('all_sine', 'mABd0LeftRightcorridor.png', 'mABd0LeftRightcorridor.png', 'mAEnd.png', 'mAFloor.png', 'floor_ceiling.png', [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500], zone_width=480)
# Cors.add_corridor('all_square', 'mABd0LeftRightcorridor.png', 'mABd0LeftRightcorridor.png', 'mBEnd.png', 'mBFloor.png', 'floor_ceiling.png', [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500], zone_width=480)

# Cors.add_corridor('sine', 'mAd1Leftcorridor.png', 'mAd1Rightcorridor.png', 'mAEnd.png', 'mAFloor.png', 'floor_ceiling.png', [2382], zone_width=1179)
# Cors.add_corridor('square', 'mBd1Leftcorridor.png', 'mBd1Rightcorridor.png', 'mBEnd.png', 'mBFloor.png', 'floor_ceiling.png', [4769], zone_width=1179)

# Cors.add_corridor('sine2', 'mAd1Leftcorridor.png', 'mAd1Rightcorridor.png', 'mAEnd.png', 'mAFloor.png', 'floor_ceiling.png', [2382], zone_width=1179)
# Cors.add_corridor('square2', 'mBd1Leftcorridor.png', 'mBd1Rightcorridor.png', 'mBEnd.png', 'mBFloor.png', 'floor_ceiling.png', [4769], zone_width=1179)
# Cors.add_corridor('star', 'mCd1Leftcorridor.png', 'mCd1Rightcorridor.png', 'mCEnd.png', 'mCFloor.png', 'floor_ceiling.png', [3560], zone_width=1179)

# Cors.print_zones()
# Cors.print_images()

# Cors.write()


######################################################################################################
## Long Corridors for Imola
######################################################################################################
## corridor number 		end image 	floor image 	left image 			right image 		number of zones 		reward side
## 0					grey 		grey 			grey_wall			grey_wall			0						None
## 1					sine 		sine 			grey_wall			grey_wall			12						Right	
## 2					square 		square 			grey_wall			grey_wall			12						Right	
## 3					grey 		grey 			leaves				leaves				1						Right	
## 4					grey 		grey 			people				people				1						Right	
## 5					grey 		grey 			leaves				leaves				1						Right	
## 6					grey 		grey 			people				people				1						Right	
## 7					grey 		grey 			green people		green people		1						Right	

#Cors = Corridor_list('./', 'NearFarLong')
# #						 left_image, right_image, end_image, floor_image, ceiling_image
#Cors.add_corridor('grey', 'grey_wall.png', 'grey_wall.png', 'grey_end_wall.png', 'grey_floor_ceiling.png', 'grey_floor_ceiling.png', [], reward='None')
#
#Cors.add_corridor('all_sine', 'mABd0LeftRightcorridor.png', 'mABd0LeftRightcorridor.png', 'mAEnd.png', 'mAFloor.png', 'floor_ceiling.png', [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500], zone_width=480)
#Cors.add_corridor('all_square', 'mABd0LeftRightcorridor.png', 'mABd0LeftRightcorridor.png', 'mBEnd.png', 'mBFloor.png', 'floor_ceiling.png', [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500], zone_width=480)
#
# # def add_corridor(self, name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width=470, reward='Right', length=7168, height=768, width=1024):
#Cors.add_corridor('leaf', 'LongLeftA.png', 'LongRightA.png', 'mAEnd.png', 'Long_floorA.png', 'Long_ceiling.png', [6500], zone_width=1000, length=10240) 
#Cors.add_corridor('people', 'LongLeftB.png', 'LongRightB.png', 'mBEnd.png', 'Long_floorB.png', 'Long_ceiling.png', [4350], zone_width=1000, length=10240)# 2500-3000
#
#Cors.add_corridor('people2', 'LongLeftB.png', 'LongRightB.png', 'mBEnd.png', 'Long_floorB.png', 'Long_ceiling.png', [4350], zone_width=1000, length=10240)
#Cors.add_corridor('green people', 'LongLeftC.png', 'LongRightC.png', 'mCEnd.png', 'Long_floorC.png', 'Long_ceiling.png', [7500], zone_width=1000, length=10240)
#Cors.add_corridor('leaf2', 'LongLeftA.png', 'LongRightA.png', 'mAEnd.png', 'Long_floorA.png', 'Long_ceiling.png', [6500], zone_width=1000, length=10240)
#
#Cors.print_zones()
#Cors.print_images()
#
#Cors.write()
#
#


#####################################################################################################
# Long Corridors for the morphing experiment of Kata
#####################################################################################################
# corridor number 	end image 	floor image 	left image 			right image 		number of zones 		reward side
## corridor number 		left image 			right image 		number of zones 		reward side
## 0					grey_wall			grey_wall			0						None
## 1					9 cheese			9 cheese			9						Right	
## 2					7 cheese			7 cheese			7						Right	
## 3					7 cheese			7 cheese			7						Right	
## 4					7 cheese			7 cheese			7						Right	
## 5					5 cheese			5 cheese			5						Right	
## 6					5 cheese			5 cheese			5						Right	
## 7					5 cheese			5 cheese			5						Right	
## 8					3 cheese			3 cheese			3						Right	
## 9					3 cheese			3 cheese			3						Right	
## 10					3 cheese			3 cheese			3						Right	
## 11					1 cheese			1 cheese			3						Right	

## 12					4-0 grid_HC			4-0 grid_HC			1						Right	
## 13					3-1 grid_HC			3-1 grid_HC			1						Left	
## 14					1-3 grid_HC			1-3 grid_HC			1						Left	
## 15					0-4 grid_HC			0-4 grid_HC			1						Right	

## 16					4-0 grid_MidC		4-0 grid_MidC		1						Right	
## 17					3-1 grid_MidC		3-1 grid_MidC		1						Left	
## 18					1-3 grid_MidC		1-3 grid_MidC		1						Left	
## 19					0-4 grid_MidC		0-4 grid_MidC		1						Right	

## 20					4-0 grid_LowC		4-0 grid_LowC		1						Right	
## 21					3-1 grid_LowC		3-1 grid_LowC		1						Left	
## 22					2-2 grid_LowC		2-2 grid_LowC		1						Left	
## 23					1-3 grid_LowC		1-3 grid_LowC		1						Left	
## 24					0-4 grid_LowC		0-4 grid_LowC		1						Right	

## 25					polka_LowC			polka_LowC			1						Left

# Cors = Corridor_list('./', 'morphing')

# Cors.add_corridor('grey', 'grey_wall.png', 'grey_wall.png', 'grey_end_wall.png', 'grey_floor_ceiling.png', 'grey_floor_ceiling.png', [], reward='None')

# Cors.add_corridor('9_cheese', 'RN_9_cheese_left.png', 'RN_9_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 820, 1500, 2180, 2860, 3540, 4220, 4900, 5580])

# Cors.add_corridor('7_cheese_a', 'RN_7_cheese_left_a.png', 'RN_7_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 1140, 1940, 2740, 3940, 4740, 5500])
# Cors.add_corridor('7_cheese_b', 'RN_7_cheese_left_b.png', 'RN_7_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 940, 1740, 2540, 3340, 4140, 5500])
# Cors.add_corridor('7_cheese_c', 'RN_7_cheese_left_c.png', 'RN_7_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [340, 1540, 2340, 3390, 4090, 4790, 5500])

# Cors.add_corridor('5_cheese_a', 'RN_5_cheese_left_a.png', 'RN_5_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 1740, 2540, 4140, 5500])
# Cors.add_corridor('5_cheese_b', 'RN_5_cheese_left_b.png', 'RN_5_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [140, 1340, 3140, 4640, 5500])
# Cors.add_corridor('5_cheese_c', 'RN_5_cheese_left_c.png', 'RN_5_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1140, 2740, 3940, 4740, 5500])

# Cors.add_corridor('3_cheese_a', 'RN_3_cheese_left_a.png', 'RN_3_cheese_right_a.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1140, 2140, 5250])
# Cors.add_corridor('3_cheese_b', 'RN_3_cheese_left_b.png', 'RN_3_cheese_right_b.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [940, 3640, 5250])
# Cors.add_corridor('3_cheese_c', 'RN_3_cheese_left_c.png', 'RN_3_cheese_right_c.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [1740, 4140, 5250])

# Cors.add_corridor('1_cheese', 'RN_1_cheese_left.png', 'RN_1_cheese_right.png', 'black_end_wall.png', 'floor_ceiling.png', 'floor_ceiling.png', [5250])

# Cors.add_corridor('4-0-grid_high_contrast', 'Grid4_0_HC_Left.png', 'Grid4_0_HC_Right.png', 'KataEnd_HC.png', 'Kata_floor_HC.png', 'Kata_ceiling_HC.png', [6000], zone_width=800, length=10240, reward='Right')
# Cors.add_corridor('3-1-grid_high_contrast', 'Grid3_1_HC_Left.png', 'Grid3_1_HC_Right.png', 'KataEnd_HC.png', 'Kata_floor_HC.png', 'Kata_ceiling_HC.png', [6000], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('1-3-grid_high_contrast', 'Grid1_3_HC_Left.png', 'Grid1_3_HC_Right.png', 'KataEnd_HC.png', 'Kata_floor_HC.png', 'Kata_ceiling_HC.png', [7500], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('0-4-grid_high_contrast', 'Grid0_4_HC_Left.png', 'Grid0_4_HC_Right.png', 'KataEnd_HC.png', 'Kata_floor_HC.png', 'Kata_ceiling_HC.png', [7500], zone_width=800, length=10240, reward='Right')

# Cors.add_corridor('4-0-grid_mid_contrast', 'Grid4_0_MC_Left.png', 'Grid4_0_MC_Right.png', 'KataEnd_MC.png', 'Kata_floor_MC.png', 'Kata_ceiling_MC.png', [6000], zone_width=800, length=10240, reward='Right')
# Cors.add_corridor('3-1-grid_mid_contrast', 'Grid3_1_MC_Left.png', 'Grid3_1_MC_Right.png', 'KataEnd_MC.png', 'Kata_floor_MC.png', 'Kata_ceiling_MC.png', [6000], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('1-3-grid_mid_contrast', 'Grid1_3_MC_Left.png', 'Grid1_3_MC_Right.png', 'KataEnd_MC.png', 'Kata_floor_MC.png', 'Kata_ceiling_MC.png', [7500], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('0-4-grid_mid_contrast', 'Grid0_4_MC_Left.png', 'Grid0_4_MC_Right.png', 'KataEnd_MC.png', 'Kata_floor_MC.png', 'Kata_ceiling_MC.png', [7500], zone_width=800, length=10240, reward='Right')

# Cors.add_corridor('4-0-grid_low_contrast', 'Grid4_0_LC_Left.png', 'Grid4_0_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [6000], zone_width=800, length=10240, reward='Right')
# Cors.add_corridor('3-1-grid_low_contrast', 'Grid3_1_LC_Left.png', 'Grid3_1_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [6000], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('2-2-grid_low_contrast', 'Grid2_2_LC_Left.png', 'Grid2_2_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [5000], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('3-1-grid_low_contrast', 'Grid3_1_LC_Left.png', 'Grid3_1_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [6000], zone_width=800, length=10240, reward='Left')
# Cors.add_corridor('0-4-grid_low_contrast', 'Grid0_4_LC_Left.png', 'Grid0_4_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [7500], zone_width=800, length=10240, reward='Right')

# Cors.add_corridor('polka_low_contrast', 'polka_LC_Left.png', 'polka_LC_Right.png', 'KataEnd_LC.png', 'Kata_floor_LC.png', 'Kata_ceiling_LC.png', [7500], zone_width=1179, length=10240, reward='Left')

# Cors.print_zones()
# Cors.print_images()

# Cors.write()


# input_path = './NearFarLong_corridors.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	corridor_list = pickle.load(input_file)
# 	input_file.close()


# input_path = './contingency_learning_corridors.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	corridor_list = pickle.load(input_file)
# 	input_file.close()
