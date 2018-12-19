#!/usr/bin/env python
# encoding=utf-8
# The MIT License (MIT)
#
# Copyright (c) 2018 Bluewhale Robot
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Randoms
#

import math

import rospy
from galileo_serial_server.msg import GalileoStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy import optimize
from scipy.spatial.distance import cdist
from std_msgs.msg import Bool
from tf.transformations import (euler_from_quaternion, quaternion_conjugate,
                                quaternion_from_euler)


class MapUpdater():
    def __init__(self):
        self.config = None
        self.tracking_status_sub = rospy.Subscriber(
            "/ORB_SLAM/trackingFlag", Bool, self.update_tracking_status)
        self.galileo_status_sub = rospy.Subscriber(
            "/galileo/status", GalileoStatus, self.update_galileo_status)
        self.min_state_distance = 0.1
        self.max_track_distance = 2
        self.max_track_angle = 3.14
        self.scores = []
        self.path_points = []
        self.tracking_pose_records = []

    def shutdown(self):
        self.tracking_status_sub.unregister()
        self.galileo_status_sub.unregister()

    def update_params(self, config, level):
        self.min_state_distance = config["min_state_distance"]
        self.max_track_distance = config["max_track_distance"]
        self.max_track_angle = config["max_track_angle"]
        return config

    def update_tracking_status(self, status):
        pass

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0])
                         + (point1[1] - point2[1]) * (point1[1] - point2[1]))

    def load_path(self, path="/home/xiaoqiang/slamdb/path.csv"):
        with open(path, "r") as nav_data_file:
            nav_data_str = nav_data_file.readline()
            while len(nav_data_str) != 0:
                pos_x = float(nav_data_str.split(" ")[0])
                pos_y = float(nav_data_str.split(" ")[1])
                pos_z = float(nav_data_str.split(" ")[2])
                self.path_points.append([pos_x, pos_y, pos_z])
                nav_data_str = nav_data_file.readline()
        for point in self.path_points:
            if len(self.scores) >= 1:
                last_point = self.scores[-1]['font']['pose'].pose.position
                distance = self.distance(
                    [pos_x, pos_y], [last_point.x, last_point.y])
                if distance < self.min_state_distance:
                    continue

            point_font = PoseStamped()
            point_font.header.frame_id = "map"
            point_font.pose.position.x = point[0]
            point_font.pose.position.y = point[1]
            point_font.pose.position.z = point[2]
            axis = self.get_target_direction(
                [point[0], point[1]], self.path_points)
            q_angle = quaternion_from_euler(
                0, 0, math.atan2(axis[1], axis[0]), axes='sxyz')
            point_font.pose.orientation = Quaternion(*q_angle)
            point_back = PoseStamped()
            point_back.header.frame_id = "map"
            point_back.pose.position.x = point[0]
            point_back.pose.position.y = point[1]
            point_back.pose.position.z = point[2]
            q_angle = quaternion_from_euler(
                0, 0, math.atan2(axis[1], axis[0]) + math.pi, axes='sxyz')
            point_back.pose.orientation = Quaternion(*q_angle)

            self.scores.append({
                'font': {
                    'pose': point_font,
                    'is_track': False,
                    'yaw': math.atan2(axis[1], axis[0]),
                },
                'back': {
                    'pose': point_back,
                    'is_track': False,
                    'yaw': math.atan2(axis[1], axis[0]) + math.pi
                }
            })

    def pose_distance(self, pose1, pose2):
        return self.distance([pose1.pose.position.x, pose1.pose.position.y],
                             [pose2.pose.position.x, pose2.pose.position.y])

    def cal_score(self):
        # 计算路径点的评分
        for score in self.scores:
            score_posi = [score['font']['pose'].pose.position.x,
                          score['font']['pose'].pose.position.y]
            near_records = filter(lambda point: self.distance(score_posi, [point['pose'].pose.position.x, point['pose'].pose.position.y]) < self.max_track_distance, self.tracking_pose_records)
            font_near_records = filter(lambda point: abs(score['font']['yaw'] - point['yaw']) < self.max_track_angle, near_records)
            back_near_records = filter(lambda point: abs(score['back']['yaw'] - point['yaw']) < self.max_track_angle, near_records)
            if len(font_near_records) >= 1:
                score['font']['is_track'] = True
            else:
                score['font']['is_track'] = False
            if len(back_near_records) >= 1:
                score['back']['is_track'] = True
            else:
                score['back']['is_track'] = False
        # 发布计算结果

        

    def update_galileo_status(self, galileo_status):
        if galileo_status.navStatus == 1 and galileo_status.mapStatus == 0 and len(self.scores) == 0:
            # 在导航状态但未载入路径点
            self.load_path()
        if len(self.tracking_pose_records) >= 1:
            previous_posi = [self.tracking_pose_records[-1]['pose'].pose.position.x,
                             self.tracking_pose_records[-1]['pose'].pose.position.y]
            
            # 距离太近，且角度差别不大
            if self.pose_distance(previous_posi, [galileo_status.currentPosX, galileo_status.currentPosY]) < self.min_state_distance and \
                abs(galileo_status.currentAngle - self.tracking_pose_records[-1]['yaw']) < self.max_track_angle:
                return
                
        currentPose = PoseStamped()
        currentPose.header.frame_id = 'map'
        currentPose.pose.position.x = galileo_status.currentPosX
        currentPose.pose.position.y = galileo_status.currentPosY
        q_angle = quaternion_from_euler(
                0, 0, galileo_status.currentAngle, axes='sxyz')
        currentPose.pose.orientation = Quaternion(*q_angle)
        if galileo_status.visualStatus != 1:
            tracking_flag = False
        else:
            tracking_flag = True
        self.tracking_pose_records.append({
            'pose': currentPose,
            'is_tracking': tracking_flag,
            'yaw': galileo_status.currentAngle,
        })

    def get_target_direction(self, target_point, nav_path_points):
        target_point_2d = target_point
        nav_path_points_2d = nav_path_points
        # 获取距此目标点最近的路径点
        nearest_point = self.closest_node(target_point_2d, nav_path_points_2d)
        # 找距离此路径点最近的其他路径点
        nav_path_points_2d_filterd = filter(
            lambda point: point[0] != target_point_2d[0] and point[1] != target_point_2d[1], nav_path_points_2d)
        nearest_point_2 = self.closest_node(
            nearest_point, nav_path_points_2d_filterd)
        # 距此路径点第二近点
        nav_path_points_2d_filterd = filter(
            lambda point: point[0] != nearest_point_2[0] and point[1] != nearest_point_2[1], nav_path_points_2d_filterd)
        nearest_point_3 = self.closest_node(
            nearest_point, nav_path_points_2d_filterd)

        def f_1(x, A, B):
            return A*x + B
        A1, = optimize.curve_fit(f_1, [nearest_point[0], nearest_point_2[0], nearest_point_3[0]],
                                 [nearest_point[1], nearest_point_2[1], nearest_point_3[1]])[0]
        if nearest_point_3[0] >= nearest_point[0]:
            return (1 / A1, 1)
        else:
            return (-1 / A1, -1)

    def closest_node(self, node, nodes):
        filtered_nodes = filter(
            lambda point: point[0] != node[0] or point[1] != node[1], nodes)
        return filtered_nodes[cdist([node], filtered_nodes).argmin()]
