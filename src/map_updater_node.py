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

import rospy
import dynamic_reconfigure.server
import dynamic_reconfigure.client
from galileo_map_updater.cfg import MapUpdaterConfig
from map_updater import MapUpdater
import time


if __name__ == "__main__":
    rospy.init_node("galileo_map_updater")
    updater = MapUpdater()
    server = dynamic_reconfigure.server.Server(MapUpdaterConfig, updater.update_params)
    client = dynamic_reconfigure.client.Client("bwMono")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()
        if updater.check_update():
            client.update_configuration({
                'update_map': True,
                'enable_gba': False,
            })
        else:
            client.update_configuration({
                'update_map': False,
                'enable_gba': False,
            })
        if updater.is_need_save():
            pass

    

    