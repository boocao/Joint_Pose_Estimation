#include "funcs.h"
#include "calculate_r_t.h"
#include "calculate_r_t_3d.h"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class RealTimeNode {
public:
    RealTimeNode(ros::NodeHandle nh, ros::NodeHandle pnh)
            : nh_(nh), pnh_(pnh), config_(loadConfig(pnh_)), app_(config_, temp_) {
        param_init(config_);

        pnh_.param<std::string>("image_topic", image_topic_, "/blackflys/image_raw");
        pnh_.param<std::string>("pointcloud_topic", pointcloud_topic_, "/hdl64_points");
        pnh_.param<double>("sync_tolerance", sync_tolerance_, 0.1);

        image_sub_ = nh_.subscribe(image_topic_, 1, &RealTimeNode::imageCallback, this);
        cloud_sub_ = nh_.subscribe(pointcloud_topic_, 1, &RealTimeNode::cloudCallback, this);

        ROS_INFO_STREAM("Realtime pose node subscribing image: " << image_topic_
                        << ", pointcloud: " << pointcloud_topic_);
    }

private:
    static YAML::Node loadConfig(ros::NodeHandle& pnh) {
        std::string config_path;
        pnh.param<std::string>("config_path", config_path, "./src/joint_pose_estimation/config_runtime.yaml");
        return YAML::LoadFile(config_path);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_image_ = msg;
        tryProcessLocked();
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_cloud_ = msg;
        tryProcessLocked();
    }

    void tryProcessLocked() {
        if (!latest_image_ || !latest_cloud_) {
            return;
        }

        const ros::Time image_stamp = latest_image_->header.stamp;
        const ros::Time cloud_stamp = latest_cloud_->header.stamp;
        if (image_stamp == last_image_stamp_ && cloud_stamp == last_cloud_stamp_) {
            return;
        }

        if (sync_tolerance_ > 0.0 && !image_stamp.isZero() && !cloud_stamp.isZero()) {
            const double dt = std::fabs((image_stamp - cloud_stamp).toSec());
            if (dt > sync_tolerance_) {
                ROS_WARN_THROTTLE(1.0, "Waiting for synchronized camera/lidar messages, dt=%.3f s", dt);
                return;
            }
        }

        const sensor_msgs::ImageConstPtr image_msg = latest_image_;
        const sensor_msgs::PointCloud2ConstPtr cloud_msg = latest_cloud_;
        last_image_stamp_ = image_stamp;
        last_cloud_stamp_ = cloud_stamp;
        processPair(image_msg, cloud_msg);
    }

    void processPair(const sensor_msgs::ImageConstPtr& image_msg,
                     const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        const auto frame_start = std::chrono::steady_clock::now();

        cv::Mat image;
        try {
            image = cv_bridge::toCvShare(image_msg, "bgr8")->image.clone();
        } catch (const cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        std::vector<cv::Point2f> img_corners;
        std::vector<cv::Point3f> img_corners_3d;
        RuntimeStats image_extract_stats;
        {
            ScopedTimer timer(image_extract_stats);
            app_.extract_image_points_four(config_, image, img_corners, img_corners_3d);
        }

        if (img_corners.size() != 4 || img_corners_3d.size() != 4) {
            ROS_WARN_THROTTLE(1.0, "Image corner extraction failed: got %zu 2D corners and %zu 3D corners",
                              img_corners.size(), img_corners_3d.size());
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_points(new pcl::PointCloud<pcl::PointXYZ>);
        RuntimeStats pointcloud_extract_stats;
        {
            ScopedTimer timer(pointcloud_extract_stats);
            app_.extract_pcd_points(config_, cloud, pc_points);
        }

        if (pc_points->points.size() < 4) {
            ROS_WARN_THROTTLE(1.0, "Point cloud corner extraction failed: got %zu points", pc_points->points.size());
            return;
        }

        std::vector<cv::Point3f> pc_points_cv;
        pc_points_cv.reserve(4);
        for (size_t i = 0; i < 4; ++i) {
            const pcl::PointXYZ& pt = pc_points->points[i];
            pc_points_cv.emplace_back(pt.x, pt.y, pt.z);
        }

        if (!extrinsic_initialized_) {
            initialize_with_horn(pc_points_cv, img_corners_3d, t, q);
            extrinsic_initialized_ = true;
        }

        std::vector<cv::Point3f> img_pose_points;
        std::vector<cv::Point3f> pc_pose_points;
        buildPosePoints(img_corners_3d, pc_points_cv, img_pose_points, pc_pose_points);

        if (img_pose_points.size() != 4 || pc_pose_points.size() != 4) {
            ROS_WARN("Pose point fusion failed.");
            return;
        }

        initialize_with_horn(pc_pose_points, img_pose_points, t, q);

        cv::Point3f center;
        cv::Point3f normal_vector;
        RuntimeStats pose_stats;
        {
            ScopedTimer timer(pose_stats);
            calculate_pose(img_pose_points, pc_pose_points, center, normal_vector);
        }

        const auto frame_end = std::chrono::steady_clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        ++frame_count_;
        ROS_INFO_STREAM(std::fixed << std::setprecision(6)
                        << "frame=" << frame_count_
                        << " time_ms=" << frame_ms
                        << " extrinsic_t=[" << t[0] << "," << t[1] << "," << t[2] << "]"
                        << " extrinsic_q_wxyz=[" << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "]"
                        << " target_center=[" << center.x << "," << center.y << "," << center.z << "]"
                        << " target_normal=[" << normal_vector.x << "," << normal_vector.y << "," << normal_vector.z << "]");
    }

    void buildPosePoints(const std::vector<cv::Point3f>& img_points,
                         const std::vector<cv::Point3f>& pc_points,
                         std::vector<cv::Point3f>& img_pose_points,
                         std::vector<cv::Point3f>& pc_pose_points) {
        const int mode = config_["estimation_mode"] ? config_["estimation_mode"].as<int>() : 3;
        if (mode != 3) {
            img_pose_points = img_points;
            pc_pose_points = pc_points;
            return;
        }

        const float alpha = 0.7f;
        const float beta = 0.7f;
        img_pose_points.resize(4);
        pc_pose_points.resize(4);

        for (size_t i = 0; i < 4; ++i) {
            const cv::Point3f pc_into_cam = transform_point(pc_points[i], t, q);

            if (!joint_filters_[i].isInitialized()) {
                joint_filters_[i].init(img_points[i], pc_into_cam);
            }

            const std::pair<cv::Point3f, cv::Point3f> filtered =
                    joint_filters_[i].update(img_points[i], pc_into_cam);

            const cv::Point3f& img_filtered = filtered.first;
            const cv::Point3f& pc_filtered_cam = filtered.second;
            cv::Point3f fused_cam;
            fused_cam.x = (img_filtered.x + pc_filtered_cam.x) * 0.5f;
            fused_cam.y = (img_filtered.y + pc_filtered_cam.y) * 0.5f;
            fused_cam.z = (img_filtered.z + pc_filtered_cam.z) * 0.5f;

            img_pose_points[i].x = (1.0f - alpha) * img_points[i].x + alpha * fused_cam.x;
            img_pose_points[i].y = (1.0f - alpha) * img_points[i].y + alpha * fused_cam.y;
            img_pose_points[i].z = (1.0f - alpha) * img_points[i].z + alpha * fused_cam.z;

            const cv::Point3f pc_back = inverse_transform_point(fused_cam, t, q);
            pc_pose_points[i].x = (1.0f - beta) * pc_points[i].x + beta * pc_back.x;
            pc_pose_points[i].y = (1.0f - beta) * pc_points[i].y + beta * pc_back.y;
            pc_pose_points[i].z = (1.0f - beta) * pc_points[i].z + beta * pc_back.z;
        }
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    YAML::Node config_;
    int temp_ = 0;
    PoseEstimation app_;

    ros::Subscriber image_sub_;
    ros::Subscriber cloud_sub_;
    sensor_msgs::ImageConstPtr latest_image_;
    sensor_msgs::PointCloud2ConstPtr latest_cloud_;
    ros::Time last_image_stamp_;
    ros::Time last_cloud_stamp_;
    std::mutex mutex_;

    std::string image_topic_;
    std::string pointcloud_topic_;
    double sync_tolerance_ = 0.1;
    bool extrinsic_initialized_ = false;
    uint64_t frame_count_ = 0;
    std::vector<KalmanFilter3D3D> joint_filters_ = std::vector<KalmanFilter3D3D>(4);
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "realtime_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    RealTimeNode node(nh, pnh);
    ros::spin();
    return 0;
}
