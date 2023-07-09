#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/PoseStamped.h>
#include <string>


//source https://www.youtube.com/watch?v=oxDRuBgPOAo
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;


auto main(int argc, char ** argv) -> int {

    ros::init(argc, argv, "goal_publisher");

    MoveBaseClient ac("move_base", true);

    while(!ac.waitForServer(ros::Duration(5.0))){
        ROS_INFO("waiting for move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal goal;

    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    float goals [4][3] = {{1.,2.,1.57},{2.,0.5,1.57},{4.5,1.5,3.14},{1.,0.5,1.57}};

    int i = 0;
    while(true)
    {
        goal.target_pose.pose.position.x = goals[i][0];
        goal.target_pose.pose.position.y = goals[i][1];
        goal.target_pose.pose.orientation.w = goals[i][2];

        ac.sendGoal(goal);

        ac.waitForResult();

        ros::Duration(2.0).sleep();

        if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            {
                ROS_INFO("goal reached");
                ++i;
                if (i>3)
                {i=0;}
            }
    }
    return 0;
}