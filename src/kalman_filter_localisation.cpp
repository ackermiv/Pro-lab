#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include "eigen3/Eigen/Dense"
#include <opencv2/opencv.hpp>


// KF Localization class
class filter
{
public:
  filter()
  {
    // Initialize time
    lastTimestamp_ = ros::Time::now(); // now
    dt_ = 0;

    // Initialize Kalman filter variables
    stateMean_ = Eigen::VectorXd::Zero(STATE_DIM);
    stateMean_[0] = 0.5;
    stateMean_[1] = 0.5;


    INITIAL_COVARIANCE = 0.1;
    stateCovariance_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * INITIAL_COVARIANCE;


    stateMeanNoCorrection_ = Eigen::VectorXd::Zero(STATE_DIM);
    stateMeanNoCorrection_[0] = 0.5;
    stateMeanNoCorrection_[1] = 0.5;

    stateCovNoCorrection_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * INITIAL_COVARIANCE;

    // Initialize motion model variables
    motionModelMatrix_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    motionModelMatrix_ << 1, 0, 0, dt_, 0, 0,
                          0, 1, 0, 0, dt_, 0,
                          0, 0, 1, 0, 0, dt_,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0;

    MOTION_NOISE_VARIANCE = 0.001;
    motionNoiseCov_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * MOTION_NOISE_VARIANCE;

    motionCommandMatrix_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    motionCommandMatrix_ << 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 1;

    motionCommand_ = Eigen::VectorXd::Zero(STATE_DIM);

    // Initialize observation matrix and noise covariance
    observationMatrix_ = Eigen::MatrixXd::Identity(3, STATE_DIM);
    OBSERVATION_NOISE_VARIANCE = 1;
    observationNoiseCov_ = Eigen::MatrixXd::Identity(OBSERVATION_DIM, OBSERVATION_DIM) * OBSERVATION_NOISE_VARIANCE;
    identity_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

    // Initialize other variables
    gridSize = 200; // occupancy grid stuff
    
    // comments and value suggestions by chatGPT
    resolution = 0.05; // Specify the resolution (e.g., 0.1 units per cell)

    // Harris Corner Extraction:
    qualityLevel = 0.3; // Specify the quality level (e.g., 0.01)
    minDistance = 18.0; // Specify the minimum distance between corners (e.g., 10.0 units)
    blockSize = 3;      // Specify the block size (e.g., 3x3)
    kSize = 3;          // Specify the Sobel kernel size (e.g., 3x3)
    MAX_CORNERS = 40;
    qualityLevelOccu = 0.7;
    MAX_CORNERS_OCCU = 3;

    // matching
    maxDistance = 0.4; // Specify the maximum distance threshold (e.g., 5.0 units)

    // publisher
    pub = nhfilter.advertise<geometry_msgs::PoseWithCovarianceStamped> ("/kalmanpose", 1);


    // load map
    std::string mapPgmFile = "/home/fhtw_user/catkin_ws/src/fhtw/test_package/maps/map.pgm";
    cv::Mat map;
    map = cv::imread(mapPgmFile);
    // std::cout << map.size() << "size" << map.type() << "type" <<std::endl;

    // Extracts the map Corners from the map
    mapCorners = extractHarrisCorners(map);
    //visualizeHarrisCorners(map, mapCorners);
    updateMapCornersToWorld(mapCorners);
    // std::cout << mapCorners << std::endl;
    map.release();
  }

  // debug function to display a CV::Mat
  void displayAndWait(const cv::Mat &image)
  {
    cv::namedWindow("test image", cv::WINDOW_NORMAL);
    cv::imshow("test image", image);
    cv::waitKey(0);
    cv::destroyWindow("test image");
  }


  // Callback for Odometry message 
  // Reads odometry message and uses odom as the motion command 
  // then updates the kalman filter to current time
  void predict(const nav_msgs::Odometry::ConstPtr &msg)
  {
    // linear velocity
    double xSpeedRobot = msg->twist.twist.linear.x;
    double yspeedRobot = msg->twist.twist.linear.y; // 0 in theory
    motionCommand_[3] = xSpeedRobot * cos(stateMean_[2]);
    motionCommand_[4] = xSpeedRobot * sin(stateMean_[2]);

    // angular velocity
    motionCommand_[5] = msg->twist.twist.angular.z;

    kalmanPrediction();
  }


  // Callback for laserscan messages
  // Preprocesses laserscan data into a Corner positions in world coordinates
  // calls a prediction and a correction step of the Kalman filter (EKF for correction) 
  void correct(const sensor_msgs::LaserScan::ConstPtr &msg)
  {
    //update to current time to minimize error
    kalmanPrediction();

    // Convert LaserScan data to x/y of impact
    Eigen::VectorXd scanData = convertLaserScanToPositionInfo(msg);
    // std::cout << scanData << std::endl;

    // Create occupancy grid
    cv::Mat occupancyGrid_ = convertLaserScanToOccupancyGrid(scanData);

    // preprocessing
    cv::Mat smoothedGrid;
    cv::GaussianBlur(occupancyGrid_, smoothedGrid, cv::Size(0, 0), sigmaBlur);

    // Extracts corners from occupancy grid, convert to world coordinates
    std::vector<cv::Point2f> scanCornersRobot = extractHarrisCornersFromOccupancyGrid(smoothedGrid);
    //visualizeHarrisCorners(smoothedGrid, scanCornersRobot);
    std::vector<cv::Point2f> scanCornersWorld;
    transformOccuCornersToCoordinatesToWorld(scanCornersRobot, scanCornersWorld);

    // save only corners that are close enough to a corner on the map
    std::vector<cv::Point2f> matchedLandmarks;
    std::vector<cv::Point2f> matchedObservations;
    matchCorners(mapCorners, scanCornersWorld, matchedLandmarks, matchedObservations);
    //std::cout << matchedCorners << std::endl;


    // correct EKF based on all observed landmarks (that exist on the map too)
    kalmanCorrection2(matchedLandmarks, matchedObservations);

    // prepare and publish the pose with Covariance
    fillPoseMessage(pose); 
    pub.publish(pose);


    // update kalman filter (not EFK)
    // kalmanCorrection(z);  //not in use anymore
  }

  void visualizeHarrisCorners(const cv::Mat &map, const std::vector<cv::Point2f> &locmapCorners)
  {
    cv::Mat mapWithCorners = map.clone();
    // cv::cvtColor(map, mapWithCorners, cv::COLOR_GRAY2BGR);

    for (const auto &mapCorner : locmapCorners)
    {
      cv::circle(mapWithCorners, mapCorner, 5, cv::Scalar(0), cv::FILLED);
    }

    //if map is a world map that has a lot of unneccesary blank space around it crop it
    if(map.rows>2000) 
    {
      cv::Rect myROI(1800, 1800, 400, 400);
      cv::Mat croppedRef(mapWithCorners, myROI);
      displayAndWait(croppedRef);
    }

    else displayAndWait(mapWithCorners);
  }


private:
  // time
  ros::Time lastTimestamp_; // last timestamp
  double dt_;               // Time step in s

  // comments and default values by chatGPT
  // occupancy grid stuff
  int gridSize;      // Specify the grid size (e.g., 800x600)
  double resolution; // Specify the resolution (e.g., 0.1 units per cell)

  // blurring
  double sigmaBlur = 0.5;

  // Harris Corner Extraction:
  int MAX_CORNERS;
  int MAX_CORNERS_OCCU;
  double qualityLevel; // Specify the quality level (e.g., 0.01)
  double minDistance;  // Specify the minimum distance between corners (e.g., 10.0 units)
  int blockSize;       // Specify the block size (e.g., 3x3)
  int kSize;           // Specify the Sobel kernel size (e.g., 3x3)
  double qualityLevelOccu;

  // Corner match
  std::vector<cv::Point2f> mapCorners;
  double maxDistance; // Specify the maximum distance threshold (e.g., 5.0 units)

  int STATE_DIM = 6; // Dimension of the state vector
  int OBSERVATION_DIM = 3; // Dimension of the observation vector 

  // kalman filter constants
  double INITIAL_COVARIANCE;
  double MOTION_NOISE_VARIANCE;
  double OBSERVATION_NOISE_VARIANCE;

  // Kalman filter variables and matrices
  Eigen::MatrixXd stateCovariance_; // State covariance matrix  Large sigma
  Eigen::VectorXd stateMean_;       // State mean vector  mu
  Eigen::VectorXd stateMeanNoCorrection_; // State mean vector  just predictions no corrections
  Eigen::MatrixXd stateCovNoCorrection_; // State covariance matrix  Large sigma just predictions no corrections
  
  // Motion model variables
  Eigen::MatrixXd motionModelMatrix_;   // Motion model matrix A Matrix
  Eigen::MatrixXd motionNoiseCov_;      // Motion noise covariance matrix R
  Eigen::MatrixXd motionCommandMatrix_; // B Matrix
  Eigen::VectorXd motionCommand_;       // U vector

  // Observation matrix and noise covariance
  Eigen::MatrixXd observationMatrix_;   // Observation matrix C
  Eigen::MatrixXd observationNoiseCov_; // Observation noise covariance matrix Q
  Eigen::MatrixXd identity_;

  // predictions
  Eigen::VectorXd predictedStateMean;
  Eigen::MatrixXd predictedStateCov;


  //from https://answers.ros.org/question/313452/ros-publishing-initialpose-topic-in-code/
  ros::Publisher pub; 
  geometry_msgs::PoseWithCovarianceStamped pose;           //We want to send this data structure (a ROS string), see 
  ros::NodeHandle nhfilter;


  // updates dt_ and dependencies of dt_
  void updateTime()
  {
    ros::Time now = ros::Time::now();
    ros::Duration duration = now - lastTimestamp_;
    dt_ = duration.toSec();
    lastTimestamp_ = now;
    // std::cout << dt_ << "dt" << std::endl;
    motionModelMatrix_ << 1, 0, 0, dt_, 0, 0,
        0, 1, 0, 0, dt_, 0,
        0, 0, 1, 0, 0, dt_,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0;
  }

  // Kalman filter update step
  void kalmanPrediction()
  {
    updateTime();

    // Predict step
    predictedStateMean = (motionModelMatrix_ * stateMean_) + (motionCommandMatrix_ * motionCommand_);
    predictedStateCov = (motionModelMatrix_ * stateCovariance_ * motionModelMatrix_.transpose()) + motionNoiseCov_;

    stateMean_ = predictedStateMean;
    stateCovariance_ = predictedStateCov;


    // comparison for debugging/testing 
    stateMeanNoCorrection_= (motionModelMatrix_ * stateMeanNoCorrection_) + (motionCommandMatrix_ * motionCommand_);
    stateCovNoCorrection_ = (motionModelMatrix_ * stateCovNoCorrection_ * motionModelMatrix_.transpose()) + motionNoiseCov_;

     //std::cout << "statemean_" << stateMean_.transpose() <<std::endl;
  }

  // Kalman filter (without extended) no longer in use, correction happens in kalmanCorrection2
  void kalmanCorrection(const Eigen::VectorXd &z)
  {

    // Correction step
    Eigen::MatrixXd innovationCov = observationMatrix_ * predictedStateCov * observationMatrix_.transpose() + observationNoiseCov_;
    Eigen::MatrixXd kalmanGain = predictedStateCov * observationMatrix_.transpose() * innovationCov.inverse();

    stateMean_ = predictedStateMean + kalmanGain * (z - observationMatrix_ * predictedStateMean);
    stateCovariance_ = (identity_- kalmanGain * observationMatrix_) * predictedStateCov;

    // Publish updated position and covariance
    // ...
  }

  // Convert LaserScan data to cartesian coordinates (robot coordinate system) positionInfo refers to the contact point of the laser
  Eigen::VectorXd convertLaserScanToPositionInfo(const sensor_msgs::LaserScan::ConstPtr &msg)
  {
    const std::vector<float> &ranges = msg->ranges;
    double angle_min = msg->angle_min;
    double angle_increment = msg->angle_increment;

    double angle_start = angle_min + stateMean_[2];

    Eigen::VectorXd positionInfo(2 * ranges.size()); // Assuming each laser beam provides x and y coordinates

    for (size_t i = 0; i < ranges.size(); ++i)
    {
      double range = ranges[i];
      double angle = angle_start + i * angle_increment;

      // Filter out invalid measurements
      if (range < msg->range_min || range > msg->range_max)
      {
        // Set invalid measurement to NaN
        positionInfo(2 * i) = std::numeric_limits<double>::quiet_NaN();     // quelle: chatGPT
        positionInfo(2 * i + 1) = std::numeric_limits<double>::quiet_NaN(); // quelle: chatGPT
      }
      else
      {
        // Convert polar coordinates to Cartesian coordinates
        double x = range * std::cos(angle);
        double y = range * -std::sin(angle);

        // Store Cartesian coordinates in position information vector
        positionInfo(2 * i) = x;
        positionInfo(2 * i + 1) = y;
      }
    }
    // std::cout << "positionInfo size:" << positionInfo.size()<< std::endl;
    return positionInfo;
  }

  // Convert laserscan position information (XY) to occupancy grid
  cv::Mat convertLaserScanToOccupancyGrid(const Eigen::VectorXd &scanData)
  {

    cv::Mat occupancyGrid(gridSize, gridSize, CV_8UC1, cv::Scalar(255)); // Initialize grid with free cells
    // std::cout << "size" << occupancyGrid.size() << std::endl;
    for (int i = 0; i < scanData.size(); i += 2)
    {
      double x = scanData(i);
      double y = scanData(i + 1);

      int col = static_cast<int>(x / resolution + gridSize / 2);
      int row = static_cast<int>(y / resolution + gridSize / 2);

      if (col >= 0 && col < gridSize && row >= 0 && row < gridSize)
      {
        occupancyGrid.at<uchar>(row, col) = 0; // Set occupied cell
      }
    }

    return occupancyGrid;
  }

  std::vector<cv::Point2f> extractHarrisCornersFromOccupancyGrid(const cv::Mat &occupancyGrid)
  {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(occupancyGrid, corners, MAX_CORNERS_OCCU, qualityLevelOccu, minDistance, cv::Mat(), blockSize, false, kSize);

    return corners;
  }

  // extracts corners from map.pgm
  std::vector<cv::Point2f> extractHarrisCorners(const cv::Mat &map) 
  {
    cv::Mat grayMap;
    cv::cvtColor(map, grayMap, cv::COLOR_BGR2GRAY);

    // std::cout << "size" << grayMap.size() << "maxcorners" << MAX_CORNERS << "ql" << qualityLevel << "mindistance" << minDistance << std::endl;
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(grayMap, corners, MAX_CORNERS, qualityLevel, minDistance, cv::Mat(), blockSize, false, kSize);

    return corners;
  }

  // coordinate transform from image to world coodrdinates
  void updateMapCornersToWorld(std::vector<cv::Point2f> &mapCorners)
  {

    for (auto &corner : mapCorners)
    {
      // Transform map corner to world coordinates
      corner.x = corner.x * resolution - 100; // value from map.yaml, trouble reading files
      corner.y = corner.y * resolution - 100;
    }
  }

  // coordinate transform from occupancy grid corners to world coodrdinates
  void transformOccuCornersToCoordinatesToWorld(const std::vector<cv::Point2f> &robotCorners, std::vector<cv::Point2f> &worldCorners)
  {
    double robotX = stateMean_[0];
    double robotY = stateMean_[1];
    // double robotTheta = stateMean_[2]; //already rotated
    cv::Point2f worldXY;

    for (auto &corner : robotCorners)
    {
      // transform occupancy grid to robot coordinates
      double scaledX = (corner.x - (gridSize / 2)) * resolution;
      double scaledY = (corner.y - (gridSize / 2)) * resolution;

      // Transform robot corner to world coordinates
      // double rotatedX = corner.x * cos(robotTheta) - corner.y * sin(robotTheta);
      // double rotatedY = corner.x * sin(robotTheta) + corner.y * cos(robotTheta);
      worldXY.x = scaledX + robotX;
      worldXY.y = scaledY + robotY;
      worldCorners.push_back(worldXY);
    }
  }


  // Matches Observed Corners with Landmarks
  void matchCorners(const std::vector<cv::Point2f> &cornersMap, const std::vector<cv::Point2f> &cornersObservation, std::vector<cv::Point2f> &matchesLandmark, std::vector<cv::Point2f> &matchesObservation)
  {

    for (const auto &corner1 : cornersMap)
    {
      double minDistance = std::numeric_limits<double>::max();
      cv::Point2f bestMatch;

      for (const auto &corner2 : cornersObservation)
      {
        double distance = cv::norm(corner1 - corner2);

        if (distance < minDistance)
        {
          minDistance = distance;
          bestMatch = corner2;
        }
      }

      if (minDistance <= maxDistance)
      {
        matchesLandmark.push_back(corner1);
        matchesObservation.push_back(bestMatch);
      }
    }
  }

  void kalmanCorrection2(const std::vector<cv::Point2f> &Landmarks, const std::vector<cv::Point2f> &Observations) // nach Thrun
  {
    Eigen::VectorXd summu = Eigen::VectorXd::Zero(STATE_DIM);
    Eigen::MatrixXd sumsigma = Eigen::MatrixXd::Zero (STATE_DIM,STATE_DIM);
    Eigen::Vector2d delta;

    // goes through observed corners and computes them into a measurement or pseudo measurement with r, phi and a signature=0;
    if (Landmarks.size()>0) std::cout << "landmarks " << Landmarks.size() << std::endl;

    for (size_t i = 0; i < Landmarks.size(); ++i)
    {
      double deltax = (Landmarks[i].x - stateMean_[0]);
      double deltay = (Landmarks[i].y - stateMean_[1]);
      delta << deltax, deltay;
      double q = delta.transpose() * delta;
      double sq = std::sqrt(q);
      Eigen::VectorXd zhat(3);
      zhat(0) = sq;
      zhat(1) = atan2((Landmarks[i].y - stateMean_[1]), (Landmarks[i].x - stateMean_[0])) - stateMean_[2];
      zhat(2) = 0;
      
      Eigen::MatrixXd H = Eigen::MatrixXd::Identity(OBSERVATION_DIM, STATE_DIM);
      H << sq * deltax, -sq * deltay, 0, 0, 0, 0,
          deltay, deltax, -1, 0, 0, 0,
          0, 0, 0, 0, 0, 0;

      /* std::cout << "H " << H << std::endl;
      std::cout << "predictedStateCov "<< predictedStateCov << std::endl;
      std::cout << "ovservationNoiseCov" << observationNoiseCov_ <<  std::endl; */

      Eigen::MatrixXd innovationCov = (H * predictedStateCov * H.transpose() + observationNoiseCov_);
      Eigen::MatrixXd K = predictedStateCov * H.transpose() * innovationCov.inverse();

      // updated measurements into the correct form to follow Thrun
      deltax = (Observations[i].x - stateMean_[0]);
      deltay = (Observations[i].y - stateMean_[1]);
      delta << deltax, deltay;
      q = delta.transpose() * delta;
      sq = std::sqrt(q);
      Eigen::VectorXd z(3);
      z(0) = sq;
      z(1) = atan2((Observations[i].y - stateMean_[1]), (Observations[i].x - stateMean_[0])) - stateMean_[2];
      z(2) = 0;


      summu = summu + K*(z-zhat);
      sumsigma = sumsigma + K * H ;

    }
      
    //std::cout << "summu " << summu.transpose() << std::endl;
    stateMean_ = predictedStateMean + summu;
    stateCovariance_ = (identity_ - sumsigma) * predictedStateCov;
    
    //std::cout << "stateMean_ " << stateMean_ << std::endl;
  }

  // adapted from https://answers.ros.org/question/313452/ros-publishing-initialpose-topic-in-code/
  void fillPoseMessage(geometry_msgs::PoseWithCovarianceStamped &pose )
  {   
      std::string fixed_frame = "map";
      pose.header.frame_id = fixed_frame;
      pose.header.stamp = ros::Time::now();

      // set x,y coord
      pose.pose.pose.position.x = stateMean_[0];
      pose.pose.pose.position.y = stateMean_[1];
      pose.pose.pose.position.z = 0.0;

      // set theta
      tf::Quaternion quat;
      quat.setRPY(0.0, 0.0, stateMean_[2]);
      tf::quaternionTFToMsg(quat, pose.pose.pose.orientation);
    
      for (int i=0; i<6; ++i)
      {
        for (int j=0; j<6; ++j)
        {    
          pose.pose.covariance[6*i+j] = stateCovariance_.coeff(i,j);
        }
      }
  }

};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "kf_localization_node");
  ros::NodeHandle nh;

  filter kf;


  // Subscribe to LaserScan topic
  ros::Subscriber odomSub = nh.subscribe<nav_msgs::Odometry>("odom", 1, &filter::predict, &kf);
  ros::Subscriber laserScanSub = nh.subscribe<sensor_msgs::LaserScan>("scan", 1, &filter::correct, &kf);

  ros::spin();

  return 0;
}
