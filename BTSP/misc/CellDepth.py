import matplotlib.pyplot as plt
import numpy as np
from mpl_point_clicker import clicker
from typing import Tuple
import xml.etree.ElementTree as ET
import pandas as pd


class CellDepth:
    def __init__(self, base_dir, output_dir, animal, session, imaging_logfile_name, show_rois=True, meanImg=True):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.animal = animal
        self.session = session
        self.imaging_logfile_name = imaging_logfile_name
        self.show_rois = show_rois
        self.gui = None
        self.point_categories = ["deep", "superficial"]
        self.meanImg = meanImg
        if self.meanImg:
            self.imgType = "meanImg"
        else:
            self.imgType = "max_proj"

        # session data
        self.ops = None
        self.stat = None
        self.iscell = None
        self.cell_index = None

        # ROI data
        self.x_center = []  # will be converted to ndarray
        self.y_center = []

        # line vertices selected by user
        self.points = []

        # figures to be saved
        self.fig_select = None
        self.fig_distances = None

        imaging_logfile_path = f"{self.base_dir}/{self.animal}_imaging/{self.session}/{imaging_logfile_name}"
        x, y, z = self.get_resolution(imaging_logfile_path)
        self.load_data()
        self.select_lines()

    def get_resolution(self, imaging_logfile_path):
        tree = ET.parse(imaging_logfile_path)
        root = tree.getroot()

        PV = root[1]
        for child in PV:
            if child.attrib['key'] == 'micronsPerPixel':
                # print(child.attrib)
                microns_x = child[0].attrib['value']
                microns_y = child[1].attrib['value']
                microns_z = child[2].attrib['value']

        if microns_x != microns_y:
            print('Warning! X and Y resolution of recording is not the same')

        return microns_x, microns_y, microns_z

    def load_data(self):
        suite2p_folder = f"{self.base_dir}/{self.animal}_imaging/{self.session}/"

        self.ops = np.load(f"{suite2p_folder}/ops.npy", allow_pickle=True)
        self.ops = self.ops.item()
        self.stat = np.load(f"{suite2p_folder}/stat.npy", allow_pickle=True)
        self.iscell = np.load(f"{suite2p_folder}/iscell.npy", allow_pickle=True)
        self.cell_index = np.nonzero(self.iscell[:, 0] == 1)[0]

        for i in self.cell_index:
            self.x_center.append(self.stat[i]['med'][1])
            self.y_center.append(self.stat[i]['med'][0])
        self.x_center, self.y_center = np.array(self.x_center), np.array(self.y_center)

    def __class_changed_cb(self, new_class: str):
        pass

    def __point_added_cb(self, position: Tuple[float, float], klass: str):
        x, y = position
        self.points.append([x, y])

    def __point_removed_cb(self, position: Tuple[float, float], klass: str, idx):
        x, y = position
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(idx, 'th')

    def select_lines(self):
        self.fig_select, ax = plt.subplots()
        ax.set_title(self.session)
        ax.imshow(np.log(self.ops[self.imgType]), cmap='gray', origin='lower')
        if self.show_rois:
            plt.scatter(self.x_center, self.y_center, c='r', marker='D', s=10)
        self.gui = clicker(ax, self.point_categories, markers=["o", "*", "x"])
        self.gui.on_class_changed(self.__class_changed_cb)
        self.gui.on_point_added(self.__point_added_cb)
        self.gui.on_point_removed(self.__point_removed_cb)
        plt.show()

    def rotate_points(self, points_in, eigenvectors):
        if np.ndim(points_in) == 1:
            points_in = np.reshape(points_in, [int(points_in.size / 2), 2])
        points_out = np.copy(points_in)

        i_ = np.array([eigenvectors[0, 0], eigenvectors[0, 1]])
        j_ = np.array([eigenvectors[1, 0], eigenvectors[1, 1]])
        for i in range(points_in.shape[0]):
            x = points_in[i, 0]
            y = points_in[i, 1]
            temp1 = x * i_
            temp2 = y * j_

            points_out[i, :] = temp1[0] + temp2[0], temp1[1] + temp2[1]

        return points_out

    def calc_points(self, pos):
        points = []
        for i in range(pos.shape[0] - 1):
            x1, y1 = pos[i, :]
            x2, y2 = pos[i + 1, :]
            n1, n2 = y1 - y2, x2 - x1
            x = np.arange(x1, x2, 1)
            if i == 0:
                x = np.arange(0, x2, 1)
            if i == pos.shape[0] - 2:
                x = np.arange(x1, self.ops['meanImgE'].shape[0], 1)

            for j in x:
                y = (-n1 * j + n1 * x1 + n2 * y1) / n2
                points.append([j, y])
        return np.array(points)

    def calc_rotations(self, points):
        data = np.transpose(self.points_deep)
        cov = np.cov(data)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        cells = np.zeros((self.x_center.size, 2))
        cells[:, 0] = self.x_center
        cells[:, 1] = self.y_center
        rot_line = self.rotate_points(points, eigenvectors)
        rot_cells = self.rotate_points(cells, eigenvectors)
        rot_flag = self.rotate_points(self.flag_point[0, :], eigenvectors)

        return rot_line, rot_cells, rot_flag

    def calculate_distances(self, show_percentages=True, show_masks=True):
        self.positions = self.gui.get_positions()
        self.pos_deep = np.array(self.positions['deep'])
        self.pos_sup = np.array(self.positions['superficial'])

        if  self.pos_deep.size == 0:
            print("please select points before running me!")
            return

        self.points_deep = self.calc_points(self.pos_deep)
        self.points_sup = self.calc_points(self.pos_sup)

        data = np.transpose(self.points_deep)  # note: this means that everything will be rotated with respect to deep line
        cov = np.cov(data)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        cells = np.zeros((self.x_center.size, 2))
        cells[:, 0] = self.x_center
        cells[:, 1] = self.y_center
        rot_line_deep = self.points_deep
        rot_line_sup = self.points_sup
        rot_cells = cells

        scale = 1.2
        self.fig_distances, ax = plt.subplots()
        ax.imshow(np.log(self.ops[self.imgType]), cmap='gray', origin='lower')
        ax.scatter(rot_line_deep[:, 0], rot_line_deep[:, 1], c='cyan')
        ax.scatter(rot_line_sup[:, 0], rot_line_sup[:, 1], c='orange')

        # distances
        self.distances_deep = np.ones((self.x_center.size)) * -1
        self.distances_sup = np.ones((self.x_center.size)) * -1
        self.target_x = np.ones((self.x_center.size)) * -1
        self.target_y = np.ones((self.y_center.size)) * -1
        for i in range(self.x_center.size):
            x = rot_cells[i, 0]
            y = rot_cells[i, 1]
            dist_deep = np.power((np.power((rot_line_deep[:, 0] - x), 2) + np.power((rot_line_deep[:, 1] - y), 2)), 1 / 2)
            index = np.argmin(dist_deep)
            self.distances_deep[i] = dist_deep[index]
            target_x, target_y = rot_line_deep[index, :]
            self.target_x[i] = target_x
            self.target_y[i] = target_y
            if rot_cells[i, 1] < rot_line_deep[index, 1]:
                self.distances_deep[i] = self.distances_deep[i] * -1

            dist_sup = np.power((np.power((rot_line_sup[:, 0] - x), 2) + np.power((rot_line_sup[:, 1] - y), 2)), 1 / 2)
            index = np.argmin(dist_sup)
            self.distances_sup[i] = dist_sup[index]
            target_x, target_y = rot_line_sup[index, :]
            self.target_x[i] = target_x
            self.target_y[i] = target_y
            if rot_cells[i, 1] > rot_line_sup[index, 1]:
                self.distances_sup[i] = self.distances_sup[i] * -1

        distance_str = [f"{np.round(self.distances_deep[d], 1)}, {np.round(self.distances_sup[d], 1)}" for d in range(len(self.distances_deep))]
        self.norm_distances = [self.distances_sup[i_cell] / (self.distances_deep[i_cell] + self.distances_sup[i_cell]) for i_cell in range(len(self.distances_deep))]
        ax.scatter(rot_cells[:, 0], rot_cells[:, 1], c=self.norm_distances, cmap="jet")

        # colored masks
        if show_masks:
            im = np.ones((self.ops['Ly'], self.ops['Lx']))
            im[:] = np.nan
            for i in range(self.x_center.size):
                roi = self.cell_index[i]
                ypix = self.stat[roi]["ypix"]
                xpix = self.stat[roi]["xpix"]
                color = self.norm_distances[i]
                im[ypix, xpix] = color
            ax.imshow(im, cmap="jet", alpha=0.4, origin='lower')

        offset = 1
        if show_percentages:
            for i in range(self.x_center.size):
                ax.text(rot_cells[i, 0]+offset, rot_cells[i, 1]+offset, f"{100*self.norm_distances[i]:.2f}%", color="white")
                pass
        plt.show()

    def save_results(self):
        pos_deep_labeled = [["deep", *coord] for coord in self.pos_deep]
        pos_sup_labeled = [["superficial", *coord] for coord in self.pos_deep]
        #pos_flag_labeled = [["flag", *coord] for coord in self.flag_point]
        join_labeled = pos_deep_labeled + pos_sup_labeled #+ pos_flag_labeled
        df_pos = pd.DataFrame(join_labeled)
        df_pos.to_excel(f"{self.output_dir}/cellDepth_boundaries_{self.session}.xlsx", index=False)

        roi_dict = {
            "ROI": self.cell_index,
            "distance (deep)": self.distances_deep,
            "distance (sup)": self.distances_sup,
            "depth": self.norm_distances
        }
        df_roi = pd.DataFrame.from_dict(roi_dict, orient="index").T
        df_roi.to_excel(f"{self.output_dir}/cellDepth_depths_{self.session}_{self.imgType}.xlsx", index=False)

        self.fig_select.savefig(f"{self.output_dir}/cellDepth_selection_{self.session}_{self.imgType}.png")
        self.fig_distances.savefig(f"{self.output_dir}/cellDepth_distances_{self.session}_{self.imgType}.png")
