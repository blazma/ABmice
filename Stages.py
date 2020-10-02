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

class Stage:
	'defining properties of a single experiment stage'
	def __init__(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.level = level
		self.stage = stage
		self.corridors = corridors
		self.next_stage = next_stage
		self.rule = rule
		self.condition = condition
		self.name = name
		self.random = random
		self.N_corridors = len(corridors)

		if substages == 0:
			self.substages = ['a'] * self.N_corridors
		else :
			self.substages = substages

	def print_props(self):
		s = self.level + '\t stage: ' + str(self.stage)+ '\t substage: ' + str(self.substages) + '\t corridors:' + str(self.corridors) + '\t next stage:' + str(self.next_stage) + '\t rule: ' + self.rule + '\t condition:' + self.condition + '\t name:' + self.name
		print(s)

class Stage_collection:
	'class for storing corridor properties'
	def __init__(self, image_path, experiment_name):
		self.image_path = image_path
		self.num_stages = 0
		self.name = experiment_name
		self.stages = []

	def add_stage(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.stages.append(Stage(level, stage, corridors, next_stage, rule, condition, name, substages=substages, random=random))
		self.num_stages = self.num_stages + 1

	def print_table(self):
		ss = 'level \t\t stage \t\t substage \t\t corridors \t next_stage \t rule \t condition \t name \n'
		print (ss)
		for i in range(self.num_stages):
			self.stages[i].print_props()

	def write(self):
		fname = self.image_path + '/' + self.name + '_stages.pkl'
		f = open(fname, 'wb')
		pickle.dump(self, f)
		f.close()

############################################################################
## Stages for Rita
############################################################################

# level		stage 	substage	corridors 		random		next_stage		rule 		condition	name

# lick&run	0		a			0				pseudo		1				lick&run	either		pretrain

# lick_zone	1		a			1				pseudo		1b				Pavl/Oper  	both		9_cheese		
# 			1		b			2-4				pseudo		1c				Pavl/Oper  	both		7_cheese		
# 			1		c			5-7				pseudo		1d				Pavl/Oper  	both		5_cheese		
# 			1		d			8-10			pseudo		1e				Pavl/Oper  	both		3_cheese
# 			1		e			11				pseudo		2-5				Pavl/Oper  	both		1_cheese

# color 	2		a			12,17			pseudo		6-9 			correct 	correct		green_striped
# 	 		3		a			13,16			pseudo		6-9 			correct 	correct		purple_striped
# 	 		4		a			14,19			pseudo		6-9 			correct 	correct		green_square
# 	 		5		a			15,18			pseudo		6-9 			correct 	correct		purple_square

# pattern	6		a			12,15			pseudo		10, 12			correct 	correct		square_purple
# 	 		7		a			13,14			pseudo		10, 12			correct 	correct		stripe_purple
# 	 		8		a			16,19			pseudo		10, 12			correct 	correct		square_green
# 	 		9		a			17,18			pseudo		10, 12			correct 	correct		stripe_green

# compound	10		a			12,14,17,19		pseudo		14				correct 	correct		green
# 	 		11		a			13,15,16,18		pseudo		15				correct 	correct		purple
# 	 		12		a			12,15,16,19		pseudo		16				correct 	correct		square
# 	 		13		a			13,14,17,18		pseudo		17				correct 	correct		striped

# switching	14		a			12,14,17,19		pseudo		14				correct 	correct		green -> purple
# 			14		b			13,15,16,18		pseudo		11				correct 	correct		green -> purple
# 	 		15		a			13,15,16,18		pseudo		15				correct 	correct		purple -> square
# 	 		15		b			12,15,16,19		pseudo		12				correct 	correct		purple -> square
# 	 		16		a			12,15,16,19		pseudo		16				correct 	correct		square -> striped
# 	 		16		b			13,14,17,18		pseudo		13				correct 	correct		square -> striped
# 	 		17		a			13,14,17,18		pseudo		17				correct 	correct		striped -> green
# 	 		17		b			12,14,17,19		pseudo		10				correct 	correct		striped -> green


# CONDITIONS:
# both condition: water is geven from both lick ports
# either condition: water is geven from one of the lick ports selected randomly - only use this with two lick ports!
# correct: water is given according to the rule specifioed by the corridor (left or right lick port)

# RULE:
# Pavlovian rule: water is always given, even without licking. Total reward consumed is counted.
# operant rule: water is given for correct licking. Total reward consumed is counted.
# correct: water is given for correct licking. Only correct choices are counted
# float: probability of reward after correct choice

# stage_list = Stage_collection('.', 'contingency_learning')
# stage_list.add_stage(level='lick&run', stage=0, corridors=[0], next_stage=[1], rule='pretrain', condition='either', name='pretrain')

# stage_list.add_stage(level='lick_zone', stage=1, corridors=[1,2,3,4,5,6,7,8,9,10,11], next_stage=[2,3,4,5], rule='Pavlovian', condition='either', name='9 -> 1 lick zone', substages=[0,1,1,1,2,2,2,3,3,3,4])

# stage_list.add_stage(level='color      ', stage=2, corridors=[12, 17], next_stage=[6,7,8,9], rule='correct', condition='correct', name='green_(striped)')
# stage_list.add_stage(level='color      ', stage=3, corridors=[13, 16], next_stage=[6,7,8,9], rule='correct', condition='correct', name='purple_(striped)')
# stage_list.add_stage(level='color      ', stage=4, corridors=[14, 19], next_stage=[6,7,8,9], rule='correct', condition='correct', name='green_(square)')
# stage_list.add_stage(level='color      ', stage=5, corridors=[15, 18], next_stage=[6,7,8,9], rule='correct', condition='correct', name='purple_(square)')

# stage_list.add_stage(level='pattern ', stage=6, corridors=[12, 15], next_stage=[13, 15], rule='correct', condition='correct', name='square_(purple)')
# stage_list.add_stage(level='pattern ', stage=7, corridors=[13, 14], next_stage=[13, 15], rule='correct', condition='correct', name='stripe_(purple)')
# stage_list.add_stage(level='pattern ', stage=8, corridors=[16, 19], next_stage=[13, 15], rule='correct', condition='correct', name='square_(green)')
# stage_list.add_stage(level='pattern ', stage=9, corridors=[17, 18], next_stage=[13, 15], rule='correct', condition='correct', name='stripe_(green)')

# stage_list.add_stage(level='compound', stage=10, corridors=[12, 14, 17, 19], next_stage=[14], rule='correct', condition='both', name='green')
# stage_list.add_stage(level='compound', stage=11, corridors=[13, 15, 16, 18], next_stage=[15], rule='correct', condition='both', name='purple')
# stage_list.add_stage(level='compound', stage=12, corridors=[12, 15, 16, 19], next_stage=[16], rule='correct', condition='both', name='square')
# stage_list.add_stage(level='compound', stage=13, corridors=[13, 14, 17, 18], next_stage=[17], rule='correct', condition='both', name='stripe')

# stage_list.add_stage(level='switching', stage=14, corridors=[12, 14, 17, 19, 13, 15, 16, 18], next_stage=[11], rule='correct', condition='both', name='green -> purple', random='10min switch', substages=[0,0,0,0,1,1,1,1])
# stage_list.add_stage(level='switching', stage=15, corridors=[13, 15, 16, 18, 12, 15, 16, 19], next_stage=[12], rule='correct', condition='both', name='purple -> square', random='10min switch', substages=[0,0,0,0,1,1,1,1])
# stage_list.add_stage(level='switching', stage=16, corridors=[12, 15, 16, 19, 13, 14, 17, 18], next_stage=[13], rule='correct', condition='both', name='square -> stripe', random='10min switch', substages=[0,0,0,0,1,1,1,1])
# stage_list.add_stage(level='switching', stage=17, corridors=[13, 14, 17, 18, 12, 14, 17, 19], next_stage=[10], rule='correct', condition='both', name='stripe -> green', random='10min switch', substages=[0,0,0,0,1,1,1,1])

# stage_list.print_table()

# stage_list.write()

# input_path = './contingency_learning_stages.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	if version_info.major == 2:
# 		stage_list = pickle.load(input_file)
# 	elif version_info.major == 3:
# 		stage_list = pickle.load(input_file, encoding='latin1')
# 	input_file.close()


############################################################################
## Stages for Snezana
############################################################################

# level		stage 	substage	corridors 		random		next_stage		rule 		condition	name

# lick&run	0		a			0				pseudo		1a				lick&run	either		pretrain

# lick_zone	1		a			1				pseudo		1b				Pavl/Oper  	both		9_cheese		
# 			1		b			2-4				pseudo		1c				Pavl/Oper  	both		7_cheese		
# 			1		c			5-7				pseudo		1d				Pavl/Oper  	both		5_cheese		
# 			1		d			8-10			pseudo		1e				Pavl/Oper  	both		3_cheese		
# 			1		e			11				pseudo		2				Pavl/Oper  	both		1_cheese		

# 1 maze 	2		a			12				pseudo		3				correct 	correct		green square
# 	 		3		a			13				pseudo		7				correct 	correct		green square NEAR
# 	 		4		a			14				pseudo		5				correct 	correct		purple striped
# 	 		5		a			15				pseudo		8				correct 	correct		purple striped NEAR
# 	 		6		a			16				pseudo		9				correct 	correct		purple_square

# 2 maze	7		a			12, 13			pseudo		4				correct 	correct		green_Q
# 	 		8		a			14, 15			pseudo		6				correct 	correct		purple_S
# 	 		9		a			12, 16			pseudo		9				correct 	correct		square



# stage_list = Stage_collection('./', 'TwoMazes')
# stage_list.add_stage(level='pretrain', stage=0, corridors=[0], next_stage=[1], rule='lick&run', condition='either', name='pretrain')

# stage_list.add_stage(level='lick_zone', stage=1, corridors=[1,2,3,4,5,6,7,8,9,10,11], next_stage=[2,3,4,5], rule='Pavlovian', condition='either', name='9 -> 1 lick zone', substages=[0,1,1,1,2,2,2,3,3,3,4])

# # stage_list.add_stage(level='lick_zone', stage=1, corridors=[1], next_stage=[2], rule='Pavlovian', condition='both', name='9_cheese')
# # stage_list.add_stage(level='lick_zone', stage=2, corridors=[2,3,4], next_stage=[3], rule='Pavlovian', condition='either', name='7 cheese')
# # stage_list.add_stage(level='lick_zone', stage=3, corridors=[5,6,7], next_stage=[3], rule='Pavlovian', condition='either', name='5 cheese')
# # stage_list.add_stage(level='lick_zone', stage=4, corridors=[8,9,10], next_stage=[3], rule='Pavlovian', condition='either', name='3 cheese')
# # stage_list.add_stage(level='lick_zone', stage=5, corridors=[11], next_stage=[3], rule='Pavlovian', condition='either', name='1 cheese')

# stage_list.add_stage(level='1 maze', stage=6, corridors=[12], next_stage=[7], rule='correct', condition='correct', name='green square')
# stage_list.add_stage(level='1 maze', stage=7, corridors=[13], next_stage=[11], rule='correct', condition='correct', name='green square NEAR')

# stage_list.add_stage(level='1 maze', stage=8, corridors=[14], next_stage=[9], rule='correct', condition='correct', name='purple striped')
# stage_list.add_stage(level='1 maze', stage=9, corridors=[15], next_stage=[12], rule='correct', condition='correct', name='purple striped NEAR')

# stage_list.add_stage(level='1 maze', stage=10, corridors=[16], next_stage=[13], rule='correct', condition='correct', name='purple square')

# stage_list.add_stage(level='2 maze', stage=11, corridors=[12,13], next_stage=[8], rule='correct', condition='correct', name='green_Q')
# stage_list.add_stage(level='2 maze', stage=12, corridors=[14,15], next_stage=[10], rule='correct', condition='correct', name='purple_S')
# stage_list.add_stage(level='2 maze', stage=13, corridors=[12,16], next_stage=[11], rule='correct', condition='correct', name='square')

# stage_list.print_table()

# stage_list.write()

# input_path = './TwoMazes_stages.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	stage_list = pickle.load(input_file)
# 	input_file.close()



############################################################################
## Stages for the NearFar task
############################################################################

# level		stage 	substage	corridors 		random		next_stage		rule 		condition	name
# lick&run	0		a			0				pseudo		1				lick&run	either		pretrain
# diff_1	1		a			1-2				pseudo		2				Pavl/Oper  	both		all-reward		
# diff_2	2		a			3-4				pseudo		2				Pavl/Oper  	both		1 zone		
# 			2		b			5-7				pseudo		2				Pavl/Oper  	both		1 zone		



# stage_list = Stage_collection('./', 'NearFar')

# stage_list.add_stage(level='pretrain', stage=0, corridors=[0], next_stage=[1], rule='lick&run', condition='either', name='pretrain')
# stage_list.add_stage(level='diff_1', stage=1, corridors=[1,2], next_stage=[2], rule='Pavlovian', condition='either', name='all_reward')
# stage_list.add_stage(level='diff_2', stage=2, corridors=[3,4,5,6,7], next_stage=[2], rule='Pavlovian', condition='both', name='1_zone', substages=[0,0,1,1,1])

# stage_list.print_table()

# stage_list.write()



############################################################################
## Stages for the NearFarLong task
############################################################################

# level		stage 	substage	corridors 		random		next_stage		rule 		condition	name
# lick&run	0		a			0				pseudo		1				lick&run	either		pretrain
# lick_zone	1		a			1				pseudo		1b				Pavl/Oper  	both		9_cheese		
# 			1		b			2-4				pseudo		1c				Pavl/Oper  	both		7_cheese		
# 			1		c			5-7				pseudo		1d				Pavl/Oper  	both		5_cheese		
# 			1		d			8-10			pseudo		1e				Pavl/Oper  	both		3_cheese
# 			1		e			11				pseudo		2				Pavl/Oper  	both		1_cheese

# diff_1	2		a			12-13			pseudo		3				Pavl/Oper  	both		1 zone		
# 			2		b			14-15			pseudo		3				Pavl/Oper  	both		1 zone		
# diff_2	3		a			14-15			pseudo		3				Pavl/Oper  	both		1 zone		
# 			3		b			16-18			pseudo		3				Pavl/Oper  	both		1 zone		

# stage_list = Stage_collection('.', 'NearFarLong')
# stage_list.add_stage(level='lick&run', stage=0, corridors=[0], next_stage=[1], rule='pretrain', condition='either', name='pretrain')

# stage_list.add_stage(level='lick_zone', stage=1, corridors=[1,2,3,4,5,6,7,8,9,10,11], next_stage=[2], rule='Pavlovian', condition='either', name='9 -> 1 lick zone', substages=[0,1,1,1,2,2,2,3,3,3,4])
# stage_list.add_stage(level='diff_1', stage=2, corridors=[12, 13, 14, 15], next_stage=[3], rule='Pavlovian', condition='both', name='1_zone', substages=[0,0,1,1])
# stage_list.add_stage(level='diff_2', stage=3, corridors=[14,15,16,17,18], next_stage=[3], rule='Pavlovian', condition='both', name='1_zone', substages=[0,0,1,1,1])

# stage_list.print_table()

# stage_list.write()

############################################################################
## Stages for the morphing experiment of Kata
############################################################################

# level		stage 	substage	corridors 		random		next_stage		rule 		condition	name
# lick&run	0		a			0				pseudo		1				lick&run	either		pretrain

# lick_zone	1		a			1				pseudo		1b				operant 	correct		9_cheese		
# 			1		b			2-4				pseudo		1c				operant 	correct		7_cheese		
# 			1		c			5-7				pseudo		1d				operant 	correct		5_cheese		
# 			1		d			8-10			pseudo		1e				operant 	correct		3_cheese
# 			1		e			11				pseudo		2-5				operant 	correct		1_cheese

# contrast 	2		a			12-13-14-15		6-0-0-6		3				operant 	correct		extreme		
# 			3		a			16-17-18-19		6-1-1-6		4				operant 	0.9			mid-contrast		
# 			4		a			20-21-23-24		6-1-1-6		5				operant 	0.9			low-contrast		
# morph		5		a			20-22-24		6-2-6		6				operant 	0.9			mid-morph		
# 			6		a			20-21-22-23-24	6-1-1-1-6	7				operant 	0.9			psychometric		
# 			7		a			20-25-24		6-2-6		5				operant 	0.9			new_corridor		


# stage_list = Stage_collection('.', 'morphing')
# stage_list.add_stage(level='lick&run', stage=0, corridors=[0], next_stage=[1], rule='pretrain', condition='either', name='pretrain')

# stage_list.add_stage(level='lick_zone', stage=1, corridors=[1,2,3,4,5,6,7,8,9,10,11], next_stage=[2,3,4,5], rule='Pavlovian', condition='either', name='9 -> 1 lick zone', substages=[0,1,1,1,2,2,2,3,3,3,4])

# stage_list.add_stage(level='contrast', stage=2, corridors=[12,13,14,15], next_stage=[3], rule='correct', condition='correct', name='extreme', random=[6,0,0,6])
# stage_list.add_stage(level='contrast', stage=3, corridors=[16,17,18,19], next_stage=[4], rule=0.9, condition='correct', name='mid-contrast', random=[6,1,1,6])
# stage_list.add_stage(level='contrast', stage=4, corridors=[20,21,23,24], next_stage=[5], rule=0.9, condition='correct', name='low-contrast', random=[6,1,1,6])
# stage_list.add_stage(level='morph', stage=5, corridors=[20,22,24], next_stage=[6], rule=0.9, condition='correct', name='morph', random=[6,2,6])
# stage_list.add_stage(level='morph', stage=6, corridors=[20,21,22,23,24], next_stage=[7], rule=0.9, condition='correct', name='psychometric', random=[6,1,1,1,6])
# stage_list.add_stage(level='morph', stage=7, corridors=[20,25,24], next_stage=[5], rule=0.9, condition='correct', name='new_corridor', random=[6,2,6])



# stage_list.print_table()

# stage_list.write()


# input_path = './TwoMazes_stages.pkl'
# if (os.path.exists(input_path)):
# 	input_file = open(input_path, 'rb')
# 	stage_list = pickle.load(input_file)
# 	input_file.close()


