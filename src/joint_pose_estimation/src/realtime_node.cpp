#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <numeric>
#include <vector>
#include <iostream>
//// 这个代码要修改3个参数：下面两个和47,48行的markerLength。
// 相机内参和畸变参数（请替换为你实际标定的参数）
cv::Mat inner_mat = (cv::Mat_<double>(3, 3) << 3493.72067,    0.     , 1122.70969,
                                                         0.     , 3507.95564,  772.31803,
                                                         0.     ,    0.     ,    1.        );
cv::Mat dist_mat = (cv::Mat_<double>(1, 5) << -0.033769, 0.457682, -0.023953, -0.007134, 0.000000);

cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat& R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                          R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z) * 180.0 / CV_PI;
}

void extract_image_points_four(cv::Mat& img,std::vector<cv::Point2f>& img_corners,
                                          std::vector<cv::Point3f> &img_corners_3d){
    // ArUco 检测
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<cv::Vec3d> rvecs, tvecs; // 旋转和平移向量

    markerIds.clear();
    markerCorners.clear();
    rvecs.clear();
    tvecs.clear();

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    double markerLength = 0.2;
    double whiteEdgeWidth = 0.025;
    // 检测 ArUco 标记
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters);

    if (!markerIds.empty()) {
        std::cout << "markerIds is not empty" << std::endl;
        cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);
        cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLength, inner_mat, dist_mat, rvecs, tvecs);
        if (rvecs.size() != markerIds.size() || tvecs.size() != markerIds.size()) {
            std::cerr << "Pose estimation results mismatch!" << std::endl;
            return;
        }

        int j = 0;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 10) {
                i = static_cast<int>(i);
                cv::Mat R;
                cv::Rodrigues(rvecs[i], R);

                img_corners.push_back(markerCorners[i][j]);
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                                                               (j == 0 || j == 3 ? -0.5 : 0.5) * (markerLength + 2*whiteEdgeWidth),
                        (j == 0 || j == 1 ? 0.5 : -0.5) * (markerLength + 2*whiteEdgeWidth),
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D);
                std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;

                std::vector<cv::Point2f> projected_points;
                cv::projectPoints(std::vector<cv::Point3f>{img_corners_3d.back()},
                                  cv::Mat::zeros(3, 1, CV_64FC1), // rvec=0（因为我们用的是世界坐标）
                                  cv::Mat::zeros(3, 1, CV_64FC1), // tvec=0
                                  inner_mat,
                                  dist_mat,
                                  projected_points);
                cv::circle(img, projected_points[0], 8, cv::Scalar(0, 255, 0), -1);
            }
        }
        j = 1;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 20) {
                i = static_cast<int>(i);
                cv::Mat R;
                cv::Rodrigues(rvecs[i], R);

                img_corners.push_back(markerCorners[i][j]);
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                                                               (j == 0 || j == 3 ? -0.5 : 0.5) * (markerLength + 2*whiteEdgeWidth),
                        (j == 0 || j == 1 ? 0.5 : -0.5) * (markerLength + 2*whiteEdgeWidth),
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D);
                std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;


                std::vector<cv::Point2f> projected_points;
                cv::projectPoints(std::vector<cv::Point3f>{img_corners_3d.back()},
                                  cv::Mat::zeros(3, 1, CV_64FC1), // rvec=0（因为我们用的是世界坐标）
                                  cv::Mat::zeros(3, 1, CV_64FC1), // tvec=0
                                  inner_mat,
                                  dist_mat,
                                  projected_points);
                cv::circle(img, projected_points[0], 8, cv::Scalar(0, 255, 0), -1);

            }
        }
        j = 2;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 30) {
                i = static_cast<int>(i);
                cv::Mat R;
                cv::Rodrigues(rvecs[i], R);

                img_corners.push_back(markerCorners[i][j]);
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                                                               (j == 0 || j == 3 ? -0.5 : 0.5) * (markerLength + 2*whiteEdgeWidth),
                        (j == 0 || j == 1 ? 0.5 : -0.5) * (markerLength + 2*whiteEdgeWidth),
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D);
                std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;


                std::vector<cv::Point2f> projected_points;
                cv::projectPoints(std::vector<cv::Point3f>{img_corners_3d.back()},
                                  cv::Mat::zeros(3, 1, CV_64FC1), // rvec=0（因为我们用的是世界坐标）
                                  cv::Mat::zeros(3, 1, CV_64FC1), // tvec=0
                                  inner_mat,
                                  dist_mat,
                                  projected_points);
                cv::circle(img, projected_points[0], 8, cv::Scalar(0, 255, 0), -1);
                //======//
            }
        }
        j = 3;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 40) {
                i = static_cast<int>(i);
                cv::Mat R;
                cv::Rodrigues(rvecs[i], R);

                img_corners.push_back(markerCorners[i][j]);
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                                                               (j == 0 || j == 3 ? -0.5 : 0.5) * (markerLength + 2*whiteEdgeWidth),
                        (j == 0 || j == 1 ? 0.5 : -0.5) * (markerLength + 2*whiteEdgeWidth),
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D);
                std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;


                std::vector<cv::Point2f> projected_points;
                cv::projectPoints(std::vector<cv::Point3f>{img_corners_3d.back()},
                                  cv::Mat::zeros(3, 1, CV_64FC1), // rvec=0（因为我们用的是世界坐标）
                                  cv::Mat::zeros(3, 1, CV_64FC1), // tvec=0
                                  inner_mat,
                                  dist_mat,
                                  projected_points);
                cv::circle(img, projected_points[0], 8, cv::Scalar(0, 255, 0), -1);
                //======//
            }
        }
    }
}

void calculate_pose(const std::vector<cv::Point3f>& points,cv::Point3f& center, cv::Point3f& normal_vector) {
    size_t N = points.size();
    if (N < 3) throw std::runtime_error("Need at least 3 points");

    center = std::accumulate(points.begin(), points.end(), cv::Point3f(0, 0, 0));
    center *= (1.0f / N);

    cv::Matx33f cov(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (const auto& p : points) {
        cv::Point3f d = p - center;
        cov(0, 0) += d.x * d.x; cov(0, 1) += d.x * d.y; cov(0, 2) += d.x * d.z;
        cov(1, 0) += d.y * d.x; cov(1, 1) += d.y * d.y; cov(1, 2) += d.y * d.z;
        cov(2, 0) += d.z * d.x; cov(2, 1) += d.z * d.y; cov(2, 2) += d.z * d.z;
    }

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cv::Mat(cov), eigenvalues, eigenvectors);

    normal_vector = cv::Point3f(
            eigenvectors.at<float>(2, 0),
            eigenvectors.at<float>(2, 1),
            eigenvectors.at<float>(2, 2)
    );
}


// ROS节点类


class RealTimeNode {
public:
    RealTimeNode(ros::NodeHandle nh) {
        image_sub = nh.subscribe("/camera_0/image_raw", 1, &RealTimeNode::imageCallback, this);
    }
    // 在函数外预先定义一个空白图像作为占位符（避免重复创建）
    cv::Mat empty_placeholder;
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat image;
        try {
            image = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
//        cv::Mat img_gray;
//        cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> img_corners;
        std::vector<cv::Point3f> img_corners_3d;
        extract_image_points_four(image, img_corners, img_corners_3d);

        // 如果你想判断角点是否为空，可以在外面加判断：
        if (img_corners.empty() || img_corners.size() != 4) {
            if (img_corners.empty()) {
                ROS_WARN("No corners detected.");
            } else {
                ROS_WARN("Detected corners are not exactly 4.");
            }

            double image_scale = 0.5;
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(), image_scale, image_scale, cv::INTER_AREA);

            // 创建一个空白图像用于右侧占位
            if (empty_placeholder.empty()) {
                int plot_size_cols = resizedImage.cols * 0.5;
                empty_placeholder = cv::Mat::zeros(resizedImage.rows, plot_size_cols, CV_8UC3); // 修改这里为 cols*0.5

                int offset_x = plot_size_cols / 2;
                int offset_y = resizedImage.rows / 2;

                cv::line(empty_placeholder, cv::Point(0, offset_y), cv::Point(plot_size_cols, offset_y), cv::Scalar(150, 150, 150), 1);
                cv::line(empty_placeholder, cv::Point(offset_x, 0), cv::Point(offset_x, resizedImage.rows), cv::Scalar(150, 150, 150), 1);
            }

            cv::Mat combined;
            cv::hconcat(resizedImage, empty_placeholder, combined);

            cv::imshow("Combined View", combined);
            cv::waitKey(1);
        }
        else{
            // 可视化角点
            for (const auto& pt : img_corners)
                cv::circle(image, pt, 3, cv::Scalar(0, 255, 0), -1);

            double image_scale = 0.5;
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(), image_scale, image_scale, cv::INTER_AREA);
            //cv::imshow("Detected Markers", resizedImage);
            //cv::waitKey(1);

            ////下面是计算 pose 的部分
            cv::Point3f center, normal_vector;
            calculate_pose(img_corners_3d, center, normal_vector);

            ROS_INFO_STREAM("Center: [" << center.x << ", " << center.y << ", " << center.z << "]");
            ROS_INFO_STREAM("Normal: [" << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "]");

            // 创建一个空白的二维平面图用于可视化 x, y
            int plot_size_cols = resizedImage.cols*0.5;
            cv::Mat plot = cv::Mat::zeros(resizedImage.rows, plot_size_cols, CV_8UC3);

            // 设置可视化的原点 (图像中心)，缩放比例
            float scale = 25.0; // 每米对应像素数
            int offset_x = plot_size_cols / 2;
            int offset_y = resizedImage.rows / 2;

            // 绘制坐标系十字线
            cv::line(plot, cv::Point(0, offset_y), cv::Point(plot_size_cols, offset_y), cv::Scalar(150, 150, 150), 1);
            cv::line(plot, cv::Point(offset_x, 0), cv::Point(offset_x, resizedImage.rows), cv::Scalar(150, 150, 150), 1);

            // 绘制 center 点
            int px = static_cast<int>(center.x * scale) + offset_x;
            int py = static_cast<int>(center.z * scale) + offset_y;
            cv::circle(plot, cv::Point(px, py), 4, cv::Scalar(0, 255, 0), -1); // 绿色圆点表示中心
            //新增：计算并显示距离
            //double distance_to_origin = std::sqrt(center.x * center.x + center.z * center.z);
            double distance_to_origin = std::sqrt( center.z * center.z);
            std::stringstream ss;
            ss << "Distance:" << std::fixed << std::setprecision(2) << distance_to_origin << " m";
            cv::putText(plot, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);


            //// 模拟车辆尺寸（单位：米）
            float vehicle_length = 3.0; // 车长（从前到后）
            float vehicle_width = 1.5;  // 车宽（左右）

            // 方向向量归一化
            cv::Point3f dir_vec = normal_vector;
            float length = std::sqrt(dir_vec.x * dir_vec.x + dir_vec.z * dir_vec.z);
            if (length > 1e-6) {
                dir_vec.x /= length;
                dir_vec.z /= length;
            }
            // 强制方向向量向外（远离图像中心/坐标系原点）
            cv::Point3f to_origin(-center.x, 0, -center.z);
            float len_to_origin = std::sqrt(to_origin.x * to_origin.x + to_origin.z * to_origin.z);
            if (len_to_origin > 1e-6) {
                to_origin.x /= len_to_origin;
                to_origin.z /= len_to_origin;
            }

            float dot_product = dir_vec.x * to_origin.x + dir_vec.z * to_origin.z;
            if (dot_product > 0.0f) {
                dir_vec.x = -dir_vec.x;
                dir_vec.z = -dir_vec.z;
            }

            // 垂直于方向向量的向量（用于宽度方向）
            cv::Point3f perp_vec(-dir_vec.z, 0, dir_vec.x); // 右侧方向

            // 四个角的世界坐标（相对于center）
            std::vector<cv::Point2f> corners_2d;

            // 从尾部开始构建矩形
            for (int i = 0; i < 4; ++i) {
                float along_dir = (i / 2) * vehicle_length; // 尾部为0，头部为length
                float side_sign = (i % 2 == 0) ? -1.0 : 1.0; // 左右

                cv::Point3f corner = center
                                     + dir_vec * along_dir
                                     + perp_vec * (vehicle_width / 2.0f) * side_sign;

                int px = static_cast<int>(corner.x * scale) + offset_x;
                int py = static_cast<int>(corner.z * scale) + offset_y;
                corners_2d.emplace_back(cv::Point2f(px, py));
            }

            // 绘制四边形
            cv::line(plot, corners_2d[0], corners_2d[1], cv::Scalar(0, 0, 255), 2);
            cv::line(plot, corners_2d[1], corners_2d[3], cv::Scalar(0, 0, 255), 2);
            cv::line(plot, corners_2d[3], corners_2d[2], cv::Scalar(0, 0, 255), 2);
            cv::line(plot, corners_2d[2], corners_2d[0], cv::Scalar(0, 0, 255), 2);

            // 显示窗口
            //cv::imshow("2D Pose Visualization", plot);
            //cv::waitKey(1);

            // 合并图像
            cv::Mat combined;
            cv::hconcat(resizedImage, plot, combined);

            cv::imshow("Combined View", combined);
            cv::waitKey(1);
        }
    }
private:
    ros::Subscriber image_sub;
};


int main(int argc, char** argv) {
    ros::init(argc, argv, "real_time_node");
    ros::NodeHandle nh;
    RealTimeNode node(nh);
    ros::spin();
    return 0;
}

