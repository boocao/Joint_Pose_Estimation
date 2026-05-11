#ifndef PROJECT_FUNCS_H
#define PROJECT_FUNCS_H

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <chrono>
#include <pcl/common/io.h>
#include <string>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <unistd.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/impl/project_inliers.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/intersections.h>
#include <pcl_conversions/pcl_conversions.h> // 提供 pcl::fromROSMsg
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/angles.h> // 用于角度计算
#include <pcl/segmentation/sac_segmentation.h>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include <math.h>
#include <ceres/ceres.h>
#include <vector>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <cv_bridge/cv_bridge.h> // 提供 cv_bridge 功能

struct RuntimeStats {
    std::vector<double> times_ms;

    void add(double ms);
    double mean() const;
    double min() const;
    double max() const;
};

class ScopedTimer {
public:
    explicit ScopedTimer(RuntimeStats& stats);
    ~ScopedTimer();

private:
    RuntimeStats& stats_;
    std::chrono::steady_clock::time_point start_;
};

void print_runtime_stats(const std::string& name, const RuntimeStats& stats);

class PoseEstimation
{
private:
    ////过程全局变量
    cv::Mat inner_mat;
    cv::Mat dist_mat;
    Eigen::Matrix4f T_lidar2cam;

    ////结构体
    struct PointXYZIRL
    {
        PCL_ADD_POINT4D; // quad-word XYZ
        float intensity;
        uint16_t label;                 ///< point label
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    } EIGEN_ALIGN16;
    struct PointXYZIA
    {
        PCL_ADD_POINT4D; // quad-word XYZ
        float intensity;
        float cosangle;
        float distance;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    } EIGEN_ALIGN16;

    ////私有函数

public:
    PoseEstimation(const YAML::Node &config, int temp);
    int getData(const YAML::Node &config, std::vector<cv::Mat> &images,std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pointclouds);

    ////图像函数
    cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &R);
    void extract_image_points_test(const YAML::Node &config,cv::Mat& img,std::vector<cv::Point2f>& img_corners,
                              std::vector<cv::Point3f> &img_corners_3d);
    void extract_image_points_four(const YAML::Node &config,cv::Mat& img,std::vector<cv::Point2f>& img_corners,
                                              std::vector<cv::Point3f> &img_corners_3d);

    ////点云函数
    static bool point_cmp(pcl::PointXYZI a, pcl::PointXYZI b);
    void point_cb(pcl::PointCloud<pcl::PointXYZI>::Ptr data, pcl::PointCloud<pcl::PointXYZI>::Ptr final_no_ground);
    int preprocess_point_clouds(const YAML::Node &config,std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs);

    void extract_pc_feature_test(const YAML::Node &config,pcl::PointCloud<pcl::PointXYZI>::Ptr &pc,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature);
    void extract_pcd_points(const YAML::Node &config, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_points);
    void sortPatternCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr pc,std::vector<pcl::PointXYZ> &v);

};


#endif //PROJECT_FUNCS_H
