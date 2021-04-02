#!/usr/bin/env python
import sys
import rospy
import geometry_msgs.msg as gmsg
import sensor_msgs.point_cloud2 as pc2


def callback(pc2_msg):
    pc2_msg.header.frame_id = base_cam + '_link'
    pub.publish(pc2_msg)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Write a base(center) camera!')
        exit(1)
    elif len(sys.argv) == 2:
        base_cam = sys.argv[1]
        print('Start!')
        rospy.init_node(base_cam, disable_signals=True)
        rospy.Subscriber('/' + base_cam + '/depth/color/points', pc2.PointCloud2, callback)
        pub = rospy.Publisher('/' + base_cam + '_zf', pc2.PointCloud2, queue_size=10)
        rospy.spin()
    else:
        print('Again!')
