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
from move_base_msgs.msg import MoveBaseActionGoal
from galileo_serial_server.msg import GalileoStatus
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from nav_msgs.msg import Path
from scipy import optimize
from scipy.spatial.distance import cdist
from std_msgs.msg import Bool
import tf
from tf.transformations import (euler_from_quaternion, quaternion_conjugate,
                                quaternion_from_euler)
from visualization_msgs.msg import Marker
from nav_msgs.srv import GetPlan, GetPlanRequest, GetMapResponse


class MapUpdater():
    def __init__(self):
        self.config = None
        self.listener = tf.TransformListener(True, rospy.Duration(10.0))
        self.min_state_distance = 0.1
        self.max_track_distance = 2
        self.max_track_angle = 3.14
        self.galileo_status = None
        self.scores = []
        self.path_points = []
        self.tracking_pose_records = []
        self.tracking_pose_records_2d = []
        self.check_update_flag = False
        self.current_goal = None
        self.current_plan = None

        # 通过全局规划器，计算目标点的朝向
        rospy.loginfo("waiting for move_base/make_plan service")
        rospy.wait_for_service("/move_base/make_plan")
        rospy.loginfo("waiting for move_base/make_plan service succeed")
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        self.init_markers()

        # init subs
        self.galileo_status_sub = rospy.Subscriber(
            "/galileo/status", GalileoStatus, self.update_galileo_status
        )
        self.goal_sub = rospy.Subscriber(
            "/move_base/goal", MoveBaseActionGoal, self.update_goal
        )
        self.plan_sub = rospy.Subscriber(
            "/move_base/NLlinepatrolPlanner/plan", Path, self.update_plan
        )

    def shutdown(self):
        self.galileo_status_sub.unregister()
        self.goal_sub.unregister()
        self.plan_sub.unregister()

    def update_goal(self, goal):
        self.current_goal = goal

    def update_plan(self, plan):
        self.current_plan = plan

    def update_params(self, config, level):
        self.min_state_distance = config["min_state_distance"]
        self.max_track_distance = config["max_track_distance"]
        self.max_track_angle = config["max_track_angle"]
        return config

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0])
                         + (point1[1] - point2[1]) * (point1[1] - point2[1]))

    def check_update(self, goal=None):
        if self.galileo_status is None:
            return False
        # 只有处在导航状态下才进行更新
        if self.galileo_status.navStatus != 1 or self.galileo_status.targetNumID == -1:
            return False
        # 获取当前目标点
        if self.current_plan is None:
            return False

        # 如果当前在比较剧烈的转动中，则开启更新
        if self.galileo_status.currentSpeedTheta != 0:
            rospy.loginfo("angle : " + str(abs(self.galileo_status.currentSpeedX / self.galileo_status.currentSpeedTheta)))

        if self.galileo_status.currentSpeedTheta != 0 and \
                abs(self.galileo_status.currentSpeedX / self.galileo_status.currentSpeedTheta) < 0.2:
            return False

        if goal is None:
            if self.current_goal == None:
                return False
            goal = self.current_goal.goal.target_pose
        # 获取当前导航路径
        # 获取当前位置
        current_pose = PoseStamped()
        current_pose.header.frame_id = "map"
        current_pose.pose.position.x = self.galileo_status.currentPosX
        current_pose.pose.position.y = self.galileo_status.currentPosY
        current_pose.pose.position.z = 0
        q_angle = quaternion_from_euler(
            0, 0, self.galileo_status.currentAngle, axes='sxyz')
        current_pose.pose.orientation = Quaternion(*q_angle)

        # 截断路径，从当前位置至之后10米范围
        current_index = self.closest_node_index([self.galileo_status.currentPosX, self.galileo_status.currentPosY],
            [[pose.pose.position.x, pose.pose.position.y] for pose in self.current_plan.poses]
        )
        self.current_plan.poses = self.current_plan.poses[current_index:]
        # 当前位置之后10米范围内
        current_path = self.current_plan.poses[:100]

        # 缩减当前路径至关键路径点
        path_key_points = []
        for point in current_path:
            # 检查距离，跳过距离近的点
            if len(path_key_points) >= 1:
                last_point = path_key_points[-1]['pose'].pose
                distance = self.distance([last_point.position.x,  last_point.position.y],
                                         [point.pose.position.x,
                                             point.pose.position.y]
                                         )
                if distance < self.min_state_distance:
                    continue

            current_pose_q = [point.pose.orientation.x, point.pose.orientation.y,
                              point.pose.orientation.z, point.pose.orientation.w]
            yaw = euler_from_quaternion(current_pose_q)[2]
            path_key_points.append({
                'pose': point,
                'yaw': yaw
            })
        self.good_markers.points = [record['pose'].pose.position for record in filter(
            lambda record: record["is_tracking"], self.tracking_pose_records)]
        self.marker_pub.publish(self.good_markers)

        # 计算当前路径关键点的坏点
        bad_points = []
        for key_point in path_key_points:
            near_records = filter(lambda point: self.distance([key_point['pose'].pose.position.x, key_point['pose'].pose.position.y], [
                                  point['pose'].pose.position.x, point['pose'].pose.position.y]) < self.max_track_distance, self.tracking_pose_records)
            same_direction_points = filter(lambda point: self.get_angle(
                key_point['yaw'], point['yaw']) < self.max_track_angle, near_records)
            if len(same_direction_points) == 0:
                # 还没有追踪情况的记录
                continue
            track_points = filter(
                lambda point: point['is_tracking'], same_direction_points)
            if len(track_points) == 0:
                bad_points.append(key_point)
        # 根据需要判断是否要开启更新地图
        if len(bad_points) > 0:
            self.bad_markers.points = [
                point['pose'].pose.position for point in bad_points]
            self.marker_pub.publish(self.bad_markers)
            rospy.logwarn("################")
            rospy.logwarn(len(bad_points))
            return True
        return False

    def is_need_save(self):
        # 检查当前是否修要保存更新的地图
        pass
        

    def load_path(self, path="/home/xiaoqiang/slamdb/path.csv"):
        with open(path, "r") as nav_data_file:
            nav_data_str = nav_data_file.readline()
            while len(nav_data_str) != 0:
                pos_x = float(nav_data_str.split(" ")[0])
                pos_y = float(nav_data_str.split(" ")[1])
                pos_z = float(nav_data_str.split(" ")[2])
                self.path_points.append([pos_x, pos_y, pos_z])
                nav_data_str = nav_data_file.readline()
        nav_path_points_2d = [[point[0], point[2]]
                              for point in self.path_points]

        for point in self.path_points:
            # 检查距离，跳过距离近的点
            if len(self.scores) >= 1:
                last_point = self.path_points[self.scores[-1]['index']]
                distance = self.distance(
                    [point[0], point[2]], [last_point[0], last_point[2]])
                if distance < self.min_state_distance:
                    continue

            pose_in_world = PoseStamped()
            pose_in_world.header.frame_id = "ORB_SLAM/World"
            pose_in_world.header.stamp = rospy.Time(0)
            pose_in_world.pose.position = Point(point[0], point[1], point[2])

            axis = self.get_target_direction(
                [point[0], point[2]], nav_path_points_2d)
            q_angle = quaternion_from_euler(
                math.pi / 2, -math.atan2(axis[1], axis[0]), 0, axes='sxyz')
            pose_in_world.pose.orientation = Quaternion(*q_angle)

            # 转至map坐标系
            tf_flag = False
            while not tf_flag and not rospy.is_shutdown():
                try:
                    t = rospy.Time(0)
                    self.listener.waitForTransform("ORB_SLAM/World", "map", t,
                                                   rospy.Duration(1.0))
                    tf_flag = True
                except (tf.LookupException, tf.ConnectivityException,
                        tf.ExtrapolationException, tf.Exception) as e:
                    tf_flag = False
                    rospy.logwarn("获取TF失败 ORB_SLAM/World->map")
                    rospy.logwarn(e)
            pose_in_map = self.listener.transformPose(
                "map", pose_in_world)
            point_font = PoseStamped()
            point_font.header.frame_id = "map"
            point_font.pose.position = pose_in_map.pose.position
            point_font.pose.orientation = pose_in_map.pose.orientation
            yaw = euler_from_quaternion([pose_in_map.pose.orientation.x, pose_in_map.pose.orientation.y,
                                         pose_in_map.pose.orientation.z, pose_in_map.pose.orientation.w])[2]

            point_back = PoseStamped()
            point_back.header.frame_id = "map"
            point_back.pose.position.x = point[0]
            point_back.pose.position.y = point[1]
            point_back.pose.position.z = point[2]
            q_angle = quaternion_from_euler(
                0, 0, yaw + math.pi, axes='sxyz')
            point_back.pose.orientation = Quaternion(*q_angle)

            self.scores.append({
                'index': self.path_points.index(point),
                'font': {
                    'pose': point_font,
                    'is_track': False,
                    'yaw': yaw
                },
                'back': {
                    'pose': point_back,
                    'is_track': False,
                    'yaw': yaw + math.pi
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
            near_records = filter(lambda point: self.distance(score_posi, [
                                  point['pose'].pose.position.x, point['pose'].pose.position.y]) < self.max_track_distance, self.tracking_pose_records)

            font_near_records = filter(lambda point: self.get_angle(
                score['font']['yaw'], point['yaw']) < self.max_track_angle and point['is_tracking'], near_records)
            back_near_records = filter(lambda point: self.get_angle(
                score['back']['yaw'], point['yaw']) < self.max_track_angle and point['is_tracking'], near_records)

            if len(font_near_records) >= 1:
                score['font']['is_track'] = True
            else:
                score['font']['is_track'] = False
            if len(back_near_records) >= 1:
                score['back']['is_track'] = True
            else:
                score['back']['is_track'] = False
        # 发布计算结果
        bad_points = filter(
            lambda score: not score['back']['is_track'], self.scores)
        self.bad_markers.points = [
            score['font']['pose'].pose.position for score in bad_points]
        self.marker_pub.publish(self.bad_markers)
        good_points = filter(
            lambda score: score['back']['is_track'], self.scores)
        self.good_markers.points = [
            score['font']['pose'].pose.position for score in good_points]

        self.marker_pub.publish(self.good_markers)
        self.marker_pub.publish(self.bad_markers)

    def init_markers(self):
        # Define a marker publisher.
        self.marker_pub = rospy.Publisher(
            '~score_markers', Marker, queue_size=0)

        self.good_markers = Marker()
        self.good_markers.ns = "waypoints"
        self.good_markers.id = 0
        self.good_markers.type = Marker.CUBE_LIST
        self.good_markers.action = Marker.ADD
        self.good_markers.lifetime = rospy.Duration(0)
        self.good_markers.scale.x = 0.05
        self.good_markers.scale.y = 0.05
        self.good_markers.color.r = 0
        self.good_markers.color.g = 1
        self.good_markers.color.b = 0
        self.good_markers.color.a = 1
        self.good_markers.header.frame_id = 'map'
        self.good_markers.header.stamp = rospy.Time.now()
        self.good_markers.points = []

        self.bad_markers = Marker()
        self.bad_markers.ns = "waypoints"
        self.bad_markers.id = 1
        self.bad_markers.type = Marker.CUBE_LIST
        self.bad_markers.action = Marker.ADD
        self.bad_markers.lifetime = rospy.Duration(0)
        self.bad_markers.scale.x = 0.05
        self.bad_markers.scale.y = 0.05
        self.bad_markers.color.r = 1
        self.bad_markers.color.g = 0
        self.bad_markers.color.b = 0
        self.bad_markers.color.a = 1
        self.bad_markers.header.frame_id = 'map'
        self.bad_markers.header.stamp = rospy.Time.now()
        self.bad_markers.points = []

        self.back_good_markers = Marker()
        self.back_good_markers.ns = "waypoints"
        self.back_good_markers.id = 0
        self.back_good_markers.type = Marker.CUBE_LIST
        self.back_good_markers.action = Marker.ADD
        self.back_good_markers.lifetime = rospy.Duration(0)
        self.back_good_markers.scale.x = 0.05
        self.back_good_markers.scale.y = 0.05
        self.back_good_markers.color.r = 0
        self.back_good_markers.color.g = 1
        self.back_good_markers.color.b = 0
        self.back_good_markers.color.a = 1
        self.back_good_markers.header.frame_id = 'map'
        self.back_good_markers.header.stamp = rospy.Time.now()
        self.back_good_markers.points = []

        self.back_bad_markers = Marker()
        self.back_bad_markers.ns = "waypoints"
        self.back_bad_markers.id = 1
        self.back_bad_markers.type = Marker.CUBE_LIST
        self.back_bad_markers.action = Marker.ADD
        self.back_bad_markers.lifetime = rospy.Duration(0.8)
        self.back_bad_markers.scale.x = 0.05
        self.back_bad_markers.scale.y = 0.05
        self.back_bad_markers.color.r = 1
        self.back_bad_markers.color.g = 0
        self.back_bad_markers.color.b = 0
        self.back_bad_markers.color.a = 1
        self.back_bad_markers.header.frame_id = 'map'
        self.back_bad_markers.header.stamp = rospy.Time.now()
        self.back_bad_markers.points = []

    def update_galileo_status(self, galileo_status):
        self.galileo_status = galileo_status
        if len(self.tracking_pose_records) >= 1:
            near_points = filter(lambda record: self.distance([record['pose'].pose.position.x, record['pose'].pose.position.y],
                                                              [galileo_status.currentPosX, galileo_status.currentPosY]) < self.min_state_distance, self.tracking_pose_records)
            same_direction_points = filter(lambda record: self.get_angle(
                record['yaw'], galileo_status.currentAngle) < self.max_track_angle, near_points)
            if len(same_direction_points) != 0:
                for record in same_direction_points:
                    if galileo_status.visualStatus == 1:
                        record["is_tracking"] = True
                    else:
                        record["is_tracking"] = False
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
        self.tracking_pose_records_2d.append([
            galileo_status.currentPosX, galileo_status.currentPosY
        ])

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
        A1, _ = optimize.curve_fit(f_1, [nearest_point[0], nearest_point_2[0], nearest_point_3[0]],
                                   [nearest_point[1], nearest_point_2[1], nearest_point_3[1]])[0]
        if (nearest_point_3[0] - nearest_point[0]) * A1 >= 0:
            return (1 / A1, 1)
        else:
            return (-1 / A1, -1)

    def closest_node(self, node, nodes):
        filtered_nodes = filter(
            lambda point: point[0] != node[0] or point[1] != node[1], nodes)
        return filtered_nodes[cdist([node], filtered_nodes).argmin()]

    def closest_node_index(self, node, nodes):
        return cdist([node], nodes).argmin()

    def get_angle(self, yaw1, yaw2):
        if abs(yaw1 - yaw2) > math.pi:
            return abs(yaw1 - yaw2) - math.pi
        return abs(yaw1 - yaw2)
