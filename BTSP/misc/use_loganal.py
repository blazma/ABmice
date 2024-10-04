from Stages import *
from Corridors import *
from LogAnal import Session

datapath = "D:\\CA1\\"
date_time = "2021-10-31_11-01-07"
name = "KS028"
task = "NearFarLong"

sess = Session(datapath, date_time, name, task)
sess.calc_behavior_score()
sess.plot_session()
