# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:58:20 2022

@author: luko.balazs
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_point_clicker import clicker
from typing import Tuple
from BAZSI_load_resolution_xml import get_resolution
plt.close('all')

# #175
# base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb175_imaging'

# session = 'srb175_220207'
# session = 'srb175_220208'
# session = 'srb175_220209'
# session = 'srb175_220210'
# session = 'srb175_220211'
# session = 'srb175_220212'
# session = 'srb175_220213'
# session = 'srb175_220214'
# session = 'srb175_220215'
# session = 'srb175_220216'
# session = 'srb175_220217'
# session = 'srb175_220218'
# session = 'srb175_220221'
# session = 'srb175_220222'
# session = 'srb175_220223'
# session = 'srb175_220223a'
# session = 'srb175_220224'

# #176
# base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb176_imaging'

# session = 'srb176_220207'
# session = 'srb176_220208'
# session = 'srb176_220209'
# session = 'srb176_220210'

# #182
# base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb182_imaging'

# session = 'srb182_220228'
# session = 'srb182_220302'
# session = 'srb182_220303'
# session = 'srb182_220303T2'
# session = 'srb182_220304'

# #184
# base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb184_imaging'

# session = 'srb184_220318'
# session = 'srb184_220321'
# session = 'srb184_220322'
# session = 'srb184_220323'
# session = 'srb184_220324'
# session = 'srb184_220325'
# session = 'srb184_220326'


# #185
base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb185_imaging'

# session = 'srb185_220317'
session = 'srb185_220318'
# session = 'srb185_220321'
# session = 'srb185_220322'
# session = 'srb185_220323'
# session = 'srb185_220324'
# session = 'srb185_220325'
# session = 'srb185_220326'


#186
# base = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb186_imaging'

# session = 'srb186_220322'
# session = 'srb186_220323'
# session = 'srb186_220324'
# session = 'srb186_220325'
# session = 'srb186_220326'

imaging_logfile_name = r'C:\Users\luko.balazs\Documents\KOKI\UBI_analyser\data\srb210_imaging\srb210_220623-004\srb210_220623-004.xml'
x,y,z = get_resolution(imaging_logfile_name)
print(x,y,z)

suite2p_folder = base + '/' + session

F_string = suite2p_folder + '/F.npy'
iscell_string = suite2p_folder + '/iscell.npy'
ops_string = suite2p_folder + '/ops.npy'
stat_string = suite2p_folder + '/stat.npy'


ops = np.load(ops_string, allow_pickle = True)
ops = ops.item()
stat = np.load(stat_string, allow_pickle = True)
iscell = np.load(iscell_string, allow_pickle = True )
cell_index = np.nonzero(iscell[:,0]==1)[0]


x_center = []
y_center = []
for i in cell_index:
    x_center.append(stat[i]['med'][1])
    y_center.append(stat[i]['med'][0])
x_center = np.array(x_center)
y_center = np.array(y_center)


points = []

fig, ax = plt.subplots()
# ax.imshow(ops['meanImgE'], cmap='gray', origin='lower')
# ax.imshow(ops['meanImg'], cmap='gray', origin='lower')
ax.imshow(np.log(ops['meanImg']), cmap='gray', origin='lower')

plt.scatter(x_center, y_center, c='r', marker = 'D', s=10)

#GUI
klicker = clicker(ax, ['line', 'deep orient.'], markers=['o', 'x'])

def class_changed_cb(new_class: str):
    print(f'The newly selected class is {new_class}')


def point_added_cb(position: Tuple[float, float], klass: str):
    x, y = position
    print("New point of class {klass} added at {x=}, {y=}")
    points.append([x,y])
    print(points)

def point_removed_cb(position: Tuple[float, float], klass: str, idx):
    x, y = position

    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(idx, 'th')
    # print("The {idx}{suffix} point of class {klass} with position {x=:.2f}, {y=:.2f}  was removed")


klicker.on_class_changed(class_changed_cb)
klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)
# klicker.get_positions() - method to get coordinates - this should be used in class call

#GUI end

#class to calculate signed distances
class deep_superficial:
    def __init__(self, positions, x_center, y_center, image, suite2p_folder):
        self.pos=np.array(positions['line'])
        self.flag_point=np.array(positions['deep orient.'])
        # print('input points shape', self.pos.shape, self.pos)
        # print('flag coordinates', self.flag_point)
        self.x_center = x_center
        self.y_center = y_center
        self.image = image
        self.suite2p_folder = suite2p_folder
        
        self.draw_initial()
        # self.save_distances()
    
    def draw_initial(self):
        fig, ax = plt.subplots()
        # ax.imshow(ops['meanImgE'], cmap='gray',origin='lower')
        ax.imshow(np.log(ops['meanImgE']), cmap='gray',origin='lower')
        ax.scatter(self.x_center, self.y_center, c='r', marker = 'D')
        ax.scatter(self.pos[:,0], self.pos[:,1], c='b')
        ax.plot(self.pos[:,0], self.pos[:,1], c='g')
        points=[]
        for i in range(self.pos.shape[0]-1):
            x1, y1=self.pos[i,:]
            x2, y2=self.pos[i+1,:]
            n1, n2 = y1-y2, x2-x1
            x = np.arange(x1, x2, 1)
            if i == 0:
                x = np.arange(0, x2, 1)
            if i == self.pos.shape[0]-2:
                x = np.arange(x1, self.image.shape[0], 1)
                    
            for j in x:
                y=(-n1*j+n1*x1+n2*y1)/n2
                points.append([j,y])
        self.points=np.array(points)
        plt.scatter(self.points[:,0],self.points[:,1], marker='x',c='g')
        plt.show()
        
        data = np.transpose(self.points)
        # print('data:',data)
        cov = np.cov(data)
        # print(cov)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # print('eigenvalues: ',eigenvalues)
        # print(' eigenvectors: ', eigenvectors)
        
        # plt.figure()
        for i in range(len(eigenvalues)):
            # print(eigenvectors[i][0], eigenvectors[i][1])
            plt.plot([0, eigenvectors[i][0]*512], [0, eigenvectors[i][1]*-512], c='b')
        plt.show()
        
        #rotation
        plt.figure()
        cells = np.zeros((self.x_center.size, 2))
        cells[:,0] = self.x_center
        cells[:,1] = self.y_center
        self.rot_line = rotate_points(self.points, eigenvectors)
        self.rot_cells = rotate_points(cells, eigenvectors)
        self.rot_flag = rotate_points(self.flag_point[0,:], eigenvectors)

        plt.scatter(self.rot_line[:,0],self.rot_line[:,1], c='g')
        plt.scatter(self.rot_cells[:,0],self.rot_cells[:,1], c='r')
        plt.scatter(self.rot_flag[:,0],self.rot_flag[:,1], c='b')
        

        
        # distances
        self.distances = np.ones((self.x_center.size))*-1
        self.target_x = np.ones((self.x_center.size))*-1
        self.target_y = np.ones((self.y_center.size))*-1
        for i in range(self.x_center.size):
            x=self.rot_cells[i,0]
            y=self.rot_cells[i,1]
            d = np.power((np.power((self.rot_line[:,0]-x),2)+np.power((self.rot_line[:,1]-y),2)),1/2)
            index=np.argmin(d)
            self.distances[i] = d[index]
            target_x, target_y = self.rot_line[index,:]
            self.target_x[i] = target_x
            self.target_y[i] = target_y
            if self.rot_cells[i,1]<self.rot_line[index,1]:
                self.distances[i] = self.distances[i]*-1
            plt.plot([x, target_x], [y, target_y], c='y')
        #check reference point

        d = np.power((np.power((self.rot_line[:,0]-self.rot_flag[0,0]),2)+np.power((self.rot_line[:,1]-self.rot_flag[0,1]),2)),1/2)
        index=np.argmin(d)
        target_x, target_y = self.rot_line[index,:]
        if target_y > self.rot_flag[0,1]:
            self.distances = self.distances*-1

        distance_str = [str(np.round_(x,1)) for x in self.distances]
        for i in range(self.x_center.size):
            plt.text(self.rot_cells[i,0],self.rot_cells[i,1], distance_str[i])
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        # +1) why the -1 * a sajátvektornál?
    def save_distances(self):

        np.save(self.suite2p_folder + '/deep_sup_distances.npy', self.distances*x)
        print('depth saved')

def rotate_points(points_in, eigenvectors):
    
    if np.ndim(points_in)==1:
        points_in = np.reshape(points_in, [int(points_in.size/2),2])
    points_out = np.copy(points_in)

    i_ = np.array([eigenvectors[0,0],eigenvectors[0,1]])
    j_ = np.array([eigenvectors[1,0],eigenvectors[1,1]])
    for i in range(points_in.shape[0]):
        x = points_in[i,0]
        y = points_in[i,1]
        temp1 = x * i_
        temp2 = y * j_

        points_out[i,:] = temp1[0]+temp2[0],temp1[1]+temp2[1]
        
    return points_out

# A=deep_superficial(klicker.get_positions(),x_center, y_center, ops['meanImgE'],suite2p_folder)


