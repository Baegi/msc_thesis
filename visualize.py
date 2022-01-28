from collections import defaultdict
from os import times
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import math
from datetime import datetime


def transform_coords(geo_y, geo_x):
    map_bounds = [ -25, 28, 54, 82] # west, south, east, north
    map_width = map_bounds[2] - map_bounds[0]
    map_height = map_bounds[3] - map_bounds[1]

    screen_x = int(600 * (geo_x - map_bounds[0]) / map_height / 1.2) # /1.2 for 120% N-S stretching of the map
    screen_y = int(600 * (1 - (geo_y - map_bounds[1]) / map_height))

    #print(f"{geo_x}/{geo_y} -> {screen_x}/{screen_y}")
    return screen_x, screen_y



class MainWindow(QMainWindow):
    def __init__(self):

        self.app = QApplication([])
        super().__init__()


        self.setWindowTitle("Flight Path Visualizer")

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.master_layout = QHBoxLayout()


        self.map_layout = QVBoxLayout()

        self.map_view = QGraphicsView()
        self.map_scene = QGraphicsScene()
        self.map_view.setScene(self.map_scene)

        self.eur_map = self.map_scene.addPixmap(QPixmap("europe.svg").scaledToHeight(600))

        self.arrow_polygon = QPolygonF([QPointF(*e) for e in [(-5, 5), (5, 0), (-5, -5), (0, 0)]])

        self.plane_arrows = dict()

        self.sensor_lines = defaultdict(list)

        self.map_layout.addWidget(self.map_view)

        self.master_layout.addLayout(self.map_layout)
        #self.master_layout.addLayout(self.map_layout)


        self.right_layout = QVBoxLayout()

        self.time_label = QLabel()
        self.right_layout.addWidget(self.time_label)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimumWidth(240)
        self.right_layout.addWidget(self.time_slider)

        self.plane_checkboxGroup = QGroupBox("Planes to track")
        self.plane_checkboxGroupLayout = QVBoxLayout()
        self.plane_checkboxGroup.setLayout(self.plane_checkboxGroupLayout)
        self.right_layout.addWidget(self.plane_checkboxGroup)

        self.plane_checkboxes = dict()

        #self.master_layout.addLayout(self.right_layout)
        self.master_layout.addLayout(self.right_layout)

        self.centralWidget.setLayout(self.master_layout)
        self.show()
        

    def update_map(self, timestamp):
        time_str = datetime.isoformat(datetime.fromtimestamp(timestamp))
        self.time_label.setText(time_str)

        for icao in self.paths:
            # clear sensor lines
            for s_line in self.sensor_lines[icao]:
                self.map_scene.removeItem(s_line)
            self.sensor_lines[icao].clear()

            if not self.plane_checkboxes[icao].isChecked():
                self.plane_arrows[icao].hide()
                continue

            if self.paths[icao][0]["timestamp"] > timestamp or self.paths[icao][-1]["timestamp"] < timestamp:
                print("hiding", timestamp, self.paths[icao][0]["timestamp"], self.paths[icao][-1]["timestamp"])
                self.plane_arrows[icao].hide()
                continue

            self.plane_arrows[icao].show()
            for i in range(len(self.paths[icao])):
                if self.paths[icao][i]["timestamp"] >= timestamp and self.paths[icao][i]["position"]:
                    print(self.paths[icao][i]["position"])
                    self.plane_arrows[icao].setPos(*transform_coords(*self.paths[icao][i]["position"]))

                    # find next message for which position is known
                    for j in range(i+1, len(self.paths[icao])):
                        if self.paths[icao][j]["position"]:
                            # set angle
                            dx = self.paths[icao][j]["position"][1] - self.paths[icao][i]["position"][1]
                            dy = self.paths[icao][j]["position"][0] - self.paths[icao][i]["position"][0]
                            if dx == 0:
                                self.plane_arrows[icao].setRotation(0 if dy >= 0 else 180)
                            else:
                                rot = math.atan(dy / dx) * 360 / (2*math.pi)
                                if dx < 0:
                                    rot = 360 - rot
                                self.plane_arrows[icao].setRotation(rot)
                            print("angle:", self.plane_arrows[icao].rotation())
                            break

                    # update sensor lines
                    for sensor in self.paths[icao][i]["sensors"]:
                        self.sensor_lines[icao].append(QGraphicsLineItem(QLineF(QPointF(
                            *transform_coords(self.sensors[sensor[1]]["lat"], self.sensors[sensor[1]]["lon"])), self.plane_arrows[icao].pos())))
                        self.map_scene.addItem(self.sensor_lines[icao][-1])

                    break

        


    def visualize_flight_paths(self, flight_paths, sensors):
        self.paths = flight_paths
        self.sensors = sensors

        t_min = 9999999999
        t_max = 0
        for path in self.paths.values():
            t_min = min(t_min, math.ceil(path[0]["timestamp"]))
            t_max = max(t_max, math.floor(path[-1]["timestamp"]))

        self.time_slider.setRange(t_min, t_max)

        self.time_slider.valueChanged.connect(self.update_map)

        for icao in self.paths:
            self.plane_arrows[icao] = QGraphicsPolygonItem(self.arrow_polygon)
            self.plane_arrows[icao].setBrush(QBrush(Qt.BrushStyle.SolidPattern))
            self.map_scene.addItem(self.plane_arrows[icao])
            self.plane_arrows[icao].hide()

            self.plane_checkboxes[icao] = QCheckBox(icao)
            self.plane_checkboxGroupLayout.addWidget(self.plane_checkboxes[icao])
            self.plane_checkboxes[icao].setChecked(True)
            self.plane_checkboxes[icao].stateChanged.connect(lambda: self.update_map(self.time_slider.value()))

        self.time_slider.setValue(t_min)

        self.update_map(t_min)

        
        self.app.exec()

