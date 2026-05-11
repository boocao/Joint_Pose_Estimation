#include "funcs.h"
#include <algorithm>
#include <iomanip>
#include <numeric>

void RuntimeStats::add(double ms) {
    times_ms.push_back(ms);
}

double RuntimeStats::mean() const {
    if (times_ms.empty()) return 0.0;
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
}

double RuntimeStats::min() const {
    if (times_ms.empty()) return 0.0;
    return *std::min_element(times_ms.begin(), times_ms.end());
}

double RuntimeStats::max() const {
    if (times_ms.empty()) return 0.0;
    return *std::max_element(times_ms.begin(), times_ms.end());
}

ScopedTimer::ScopedTimer(RuntimeStats& stats)
        : stats_(stats), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    stats_.add(ms);
}

void print_runtime_stats(const std::string& name, const RuntimeStats& stats) {
    const double mean = stats.mean();
    const double fps = mean > 0.0 ? 1000.0 / mean : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "[Runtime] " << name
              << " | count: " << stats.times_ms.size()
              << " | mean: " << mean << " ms"
              << " | min: " << stats.min() << " ms"
              << " | max: " << stats.max() << " ms"
              << " | FPS: " << fps
              << std::endl;
}

PoseEstimation::PoseEstimation(const YAML::Node &config, int temp){
    inner_mat = cv::Mat(3, 3, CV_64F); // 3x3 内参矩阵
    dist_mat= cv::Mat(1, 5, CV_64F); // 1x5 畸变参数矩阵
    T_lidar2cam= Eigen::Matrix4f::Identity();
    // 读取内参矩阵
    if (config["camera_intrinsic"]["inner"]) {
        YAML::Node intrin = config["camera_intrinsic"]["inner"];
        if (intrin.size() != 9) { // 确保内参矩阵有 9 个元素 (3x3)
            std::cerr << "Error: Inner matrix must have 9 elements (3x3)." << std::endl;
        }
        for (size_t i = 0; i < intrin.size(); ++i) {
            double value = intrin[i].as<double>();
            inner_mat.at<double>(i / 3, i % 3) = value; // 按行优先填充矩阵
        }
    }
    // 读取畸变参数
    if (config["camera_intrinsic"]["dist"]) {
        YAML::Node dist = config["camera_intrinsic"]["dist"];
        if (dist.size() != 5) { // 确保畸变参数有 5 个元素
            std::cerr << "Error: Distortion matrix must have 5 elements." << std::endl;
        }
        for (size_t i = 0; i < dist.size(); ++i) {
            double value = dist[i].as<double>();
            dist_mat.at<double>(0, i) = value; // 填充畸变参数矩阵
        }
    }
    // 读取 lidar2camera 转换矩阵
    if (config["lidar2camera"]["data"]) {
        YAML::Node lidar2cam_mat = config["lidar2camera"]["data"];
        if (lidar2cam_mat.size() != 16) { // 确保转换矩阵有 16 个元素 (4x4)
            std::cerr << "Error: Lidar2Camera transformation matrix must have 16 elements (4x4)." << std::endl;
            return;
        }
        // 填充 Eigen::Matrix4f 类型的 T_lidar2cam
        for (size_t i = 0; i < 16; ++i) {
            float value = lidar2cam_mat[i].as<float>();
            T_lidar2cam(i / 4, i % 4) = value; // 按行优先填充矩阵
        }
    }
}

int PoseEstimation::getData(const YAML::Node &config, std::vector<cv::Mat> &images,
                       std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pointclouds)
{
    std::string txtName = config["txtName"].as<std::string>();
    std::string folderName = config["folderName"].as<std::string>();

    std::ifstream readtxt;
    readtxt.open(txtName);
    if (!readtxt)
    {
        std::cout << "\033[31mgetData Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }

    int n = 0;
    std::string filename;
    while (readtxt >> filename)
    {
        filename = folderName + filename;
        cv::Mat image = cv::imread(filename);
        images.push_back(image);
        //std::cout << "rows= " << image.rows << std::endl;
        //std::cout << "cols= " << image.cols << std::endl;

        if (!(readtxt >> filename))
        {
            std::cout << "\033[31mgetData Error: no pointcloud!\033[0m" << std::endl;
            std::exit(0);
        }
        filename = folderName + filename;
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *raw);
        pointclouds.push_back(raw);

        n++;
    }
    return n;
}

////处理图像
// 从旋转矩阵提取欧拉角（单位：度数）
cv::Vec3d PoseEstimation::rotationMatrixToEulerAngles(const cv::Mat &R) {
    double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6; // 检查奇异情况
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
    //输出结果为[roll, pitch, yaw]
    return cv::Vec3d(x, y, z) * 180.0 / CV_PI; // 转换为度数
}
//提取图像aruco角点
void PoseEstimation::extract_image_points_test(const YAML::Node &config,cv::Mat& img,std::vector<cv::Point2f>& img_corners,
                                     std::vector<cv::Point3f> &img_corners_3d){
    if (img.empty()) {
        std::cout << "Image not found!" << std::endl;
    }

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
    double markerLength = config["markerLength"].as<float>();
    // 检测 ArUco 标记
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters);

    if (!markerIds.empty()) {
        std::cout << "markerIds is not empty" << std::endl;;

        // 绘制检测到的 ArUco 码边框
        cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);

        std::cout << "inner_mat: " << inner_mat << std::endl;
        std::cout << "dist_mat: " << dist_mat << std::endl;
        // 每个标记器坐标系以标记器的中间为中心，Z轴垂直于标记器平面。
        // 我正常看向标定码图案时，标记点在marker坐标系（世界坐标系）的坐标是：
        //           Y↑
        //           |
        //           |
        // (-0.5,0.5)+-----------+ (0.5,0.5)
        //           |           |
        //           |  marker   |  ← Z 轴垂直指向你
        //           |           |
        // (-0.5,-0.5)+-----------+ (0.5,-0.5)-----→ X
        // 标记点顺时针存放markerCorners中：
        //    0: (-markerLength/2, markerLength/ 2,0)   ->   1: (markerLength/2, markerLength/ 2,0)
        //                       ↑                                          ↓
        //    3:(-markerLength/2， -markerLength/ 2,0)   <-   2: (markerLength/2， -markerLength/ 2,0)
        // 得到的 rvec 和 tvec 是从每个marker坐标系到相机坐标系的变换。
        cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLength, inner_mat, dist_mat, rvecs, tvecs);

        if (rvecs.size() != markerIds.size() || tvecs.size() != markerIds.size()) {
            std::cerr << "Pose estimation results mismatch!" << std::endl;
            return;
        }

        // 遍历所有检测到的 ArUco 码
        for (size_t i = 0; i < markerIds.size(); ++i) {
            std::cout << "Marker ID-" << markerIds[i] << " Corners: " << std::endl;

            // 将旋转向量转换为旋转矩阵
            std::cout << "rvec[" << i << "] = " << rvecs[i] << std::endl;
            std::cout << "tvec[" << i << "] = " << tvecs[i] << std::endl;
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);
            cv::Vec3d eulerAngles = rotationMatrixToEulerAngles(R);

            for (size_t j = 0; j < markerCorners[i].size(); j++) {
                // 存储2D角点到 img_corners
                img_corners.push_back(markerCorners[i][j]);
                // 存储3D角点到 img_corners_3d
                // 使用r, t将marker标记点转换到相机坐标系后，标志点顺序改变为：（左为像素点存放，右为转换后对应的点在opencv相机坐标系的顺序）
                //  0 -> 1       3 -> 2
                //  ↑    ↓  -->  ↑    ↓
                //  3 <- 2       0 <- 1
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                        (j == 0 || j == 3 ? -0.5 : 0.5) * markerLength,  // X 值，根据角点判断是 -0.5 还是 0.5
                        (j == 0 || j == 1 ? 0.5 : -0.5) * markerLength,  // Y 值，根据角点判断是 0.5 还是 -0.5
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D); //注意这时候的点
                // 打印 2D 和 3D 角点信息
                std::cout << "(" << markerCorners[i][j].x << ", " << markerCorners[i][j].y << ") and "
                          << "(" << point3D.x << ", " << point3D.y << ", " << point3D.z << ") " << std::endl;
            }

            ////以下为可视化
            float axisLength = 0.1f; // 坐标轴长度（单位：米）
            cv::aruco::drawAxis(img, inner_mat, dist_mat,rvecs[i], tvecs[i],axisLength);
            // 在图像上显示旋转信息
            std::string rotationText = "Rot: [" + std::to_string(eulerAngles[0]) + ", " +
                                       std::to_string(eulerAngles[1]) + ", " +
                                       std::to_string(eulerAngles[2]) + "]";
            cv::putText(img, rotationText, markerCorners[i][0] + cv::Point2f(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 4);
            // 在图像上显示位置信息
            std::string positionText = "Pos: [" + std::to_string(tvecs[i][0]) + ", " +
                                       std::to_string(tvecs[i][1]) + ", " +
                                       std::to_string(tvecs[i][2]) + "]";
            cv::putText(img, positionText, markerCorners[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 4);
        }
    }
}
void PoseEstimation::extract_image_points_four(const YAML::Node &config,cv::Mat& img,std::vector<cv::Point2f>& img_corners,
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
    double markerLength = config["markerLength"].as<float>();
    double whiteEdgeWidth = config["whiteEdgeWidth"].as<float>();
    // 检测 ArUco 标记
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters);

    if (!markerIds.empty()) {
        std::cout << "markerIds is not empty" << std::endl;
        // 绘制检测到的 ArUco 码边框
        cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);

        // 得到的 rvec 和 tvec 是从marker坐标系到相机坐标系的变换
        cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLength, inner_mat, dist_mat, rvecs, tvecs);

        if (rvecs.size() != markerIds.size() || tvecs.size() != markerIds.size()) {
            std::cerr << "Pose estimation results mismatch!" << std::endl;
            return;
        }

        // 遍历所有检测到的 ArUco 码
        int j = 0;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 10) {
//                std::cout << " marker ID-" << markerIds[i];
                i = static_cast<int>(i);
                // 将旋转向量转换为旋转矩阵
                cv::Mat R;
                cv::Rodrigues(rvecs[i], R);

                img_corners.push_back(markerCorners[i][j]);
                // 想得到aruco标定板的角点的话：
                cv::Mat corner3D = R * (cv::Mat_<double>(3, 1) <<
                        (j == 0 || j == 3 ? -0.5 : 0.5) * (markerLength + 2*whiteEdgeWidth),
                        (j == 0 || j == 1 ? 0.5 : -0.5) * (markerLength + 2*whiteEdgeWidth),
                        0) + cv::Mat(tvecs[i]);
                cv::Point3f point3D(corner3D.at<double>(0, 0),
                                    corner3D.at<double>(1, 0),
                                    corner3D.at<double>(2, 0));
                img_corners_3d.push_back(point3D);
                //std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;

                // === 开始绘制 2D 和 3D 投影点(可行！) === //
//                cv::circle(img, img_corners.back(), 7, cv::Scalar(255, 0, 0), 3);
                // 绘制从 3D 投影回来的点（红色）
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
        j = 1;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 20) {
//                std::cout << " marker ID-" << markerIds[i];
                i = static_cast<int>(i);
                // 将旋转向量转换为旋转矩阵
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
                //std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;

                // === 开始绘制 2D 和 3D 投影点(可行！) === //
//                cv::circle(img, img_corners.back(), 7, cv::Scalar(0, 255, 0), 3);
                // 绘制从 3D 投影回来的点（红色）
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
        j = 2;
        for (size_t i = 0; i < markerIds.size(); ++i){
            if (markerIds[i] == 30) {
//                std::cout << " marker ID-" << markerIds[i];
                i = static_cast<int>(i);
                // 将旋转向量转换为旋转矩阵
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
                //std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;

                // === 开始绘制 2D 和 3D 投影点(可行！) === //
//                cv::circle(img, img_corners.back(), 7, cv::Scalar(0, 0, 255), 3);
                // 绘制从 3D 投影回来的点（红色）
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
//                std::cout << " marker ID-" << markerIds[i];
                i = static_cast<int>(i);
                // 将旋转向量转换为旋转矩阵
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
                //std::cout << " 3D corner[" << j << "]: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;

                // === 开始绘制 2D 和 3D 投影点(可行！) === //
//                cv::circle(img, img_corners.back(), 7, cv::Scalar(0, 255, 255), 3);
                // 绘制从 3D 投影回来的点（红色）
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
    else {
        std::cout << "markerIds is empty !" << std::endl;
    }
}

////处理点云
bool PoseEstimation::point_cmp(pcl::PointXYZI a, pcl::PointXYZI b)
{
    return a.z < b.z;
}
void PoseEstimation::point_cb(pcl::PointCloud<pcl::PointXYZI>::Ptr data, pcl::PointCloud<pcl::PointXYZI>::Ptr final_no_ground)
{
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // For mark ground points and hold all points
    pcl::PointCloud<pcl::PointXYZI>::Ptr data_org(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*data, *data_org);

    float angle;
    PointXYZIRL point;
    pcl::PointCloud<PointXYZIRL>::Ptr g_all_pc(new pcl::PointCloud<PointXYZIRL>);

    for (size_t i = 0; i < data->points.size(); i++)
    {
        point.x = data->points[i].x;
        point.y = data->points[i].y;
        point.z = data->points[i].z;
        point.intensity = data->points[i].intensity;

        point.label = 0u; // 0 means uncluster
        g_all_pc->points.push_back(point);
    }
    //std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
    // 2.Sort on Z-axis value.
    sort(data_org->points.begin(), data_org->points.end(), point_cmp);
    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    pcl::PointCloud<pcl::PointXYZI>::iterator it = data_org->points.begin();
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < -1.5 * 2.0)
        {
            it++;
        }
        else
        {
            break;
        }
    }
    data_org->erase(data_org->points.begin(), it);
    // 4. Extract init ground seeds.
    double sum = 0;
    int cnt = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_seeds_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // Calculate the mean height value.
    for (int i = 0; i < data_org->points.size() && cnt < 20; i++)
    {
        sum += data_org->points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < lpr_height + 0.4)
        {
            g_seeds_pc->points.push_back(data_org->points[i]);
        }
    }

    g_ground_pc = g_seeds_pc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_not_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // 5. Ground plane fitter mainloop
    float d_, th_dist_d_;
    Eigen::MatrixXf normal_;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Matrix3f cov;
        Eigen::Vector4f pc_mean;
        pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean);
        // Singular Value Decomposition: SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
        // use the least singular vector as normal
        normal_ = (svd.matrixU().col(2));
        // mean ground seeds value
        Eigen::Vector3f seeds_mean = pc_mean.head<3>();

        // according to normal.T*[x,y,z] = -d
        d_ = -(normal_.transpose() * seeds_mean)(0, 0);
        // set distance threhold to `th_dist - d`
        th_dist_d_ = 0.3 - d_;

        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(data->points.size(), 3);
        int j = 0;
        for (auto p : data->points)
        {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0; r < result.rows(); r++)
        {
            if (result[r] < th_dist_d_)
            {
                g_all_pc->points[r].label = 1u; // means ground
                g_ground_pc->points.push_back(data->points[r]);
            }
            else
            {
                g_all_pc->points[r].label = 0u; // means not ground and non clusterred
                g_not_ground_pc->points.push_back(data->points[r]);
            }
        }
    }

    pcl::copyPointCloud(*g_not_ground_pc, *final_no_ground);

    // ROS_INFO_STREAM("origin: "<<g_not_ground_pc->points.size()<<" post_process: "<<final_no_ground->points.size());

    // publish ground points
    //    sensor_msgs::PointCloud2 ground_msg;
    //    pcl::toROSMsg(*g_ground_pc, ground_msg);
    //    ground_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    ground_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_ground_.publish(ground_msg);
    //
    //    // publish not ground points
    //    sensor_msgs::PointCloud2 groundless_msg;
    //    pcl::toROSMsg(*final_no_ground, groundless_msg);
    //    groundless_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    groundless_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_no_ground_.publish(groundless_msg);
    //
    //    // publish all points
    //    sensor_msgs::PointCloud2 all_points_msg;
    //    pcl::toROSMsg(*g_all_pc, all_points_msg);
    //    all_points_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    all_points_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_all_points_.publish(all_points_msg);
    //std::cout << g_ground_pc->size() << std::endl;
}
int PoseEstimation::preprocess_point_clouds(const YAML::Node &config,std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs)
{
    //// 筛选点云帧数最小为2
    if (pcs.size() < 2) {
        std::cout << "Too few point cloud frames" << std::endl;
    }
    else {
        // 遍历点云帧（从第1帧到倒数第2帧）
        for (int i = 0; i < pcs.size(); ++i) {
            // 过滤NaN点
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*pcs[i], *cloud_filtered, indices);

//            // 输出原始点云和过滤后点云的大小
//            std::cout << "Frame " << i << " - Raw point cloud size: " << pcs[i]->size() << std::endl;
//            std::cout << "Frame " << i << " - Filtered point cloud size: " << cloud_filtered->size() << std::endl;

            // 在这里可以添加进一步的处理逻辑，例如保存过滤后的点云或进行其他操作
             pcs[i] = cloud_filtered; // 如果需要替换原始点云，可以取消注释此行
        }
    }
    ////是否移除离群点
    if (config["remove_outlier"].as<bool>()) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_removed(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::StatisticalOutlierRemoval <pcl::PointXYZI> sor;
        for (int i = 0; i < pcs.size(); ++i)
        {
            sor.setInputCloud(pcs[i]);
            sor.setMeanK(50); // 考虑邻近点的数量
            sor.setStddevMulThresh(1.0); // 标准差阈值
            sor.filter(*pc_removed);
            pcl::copyPointCloud(*pc_removed, *pcs[i]); //直接覆盖原点云
        }
    }
    ////下采样点云，体素滤波
    if (config["down_sample"].as<bool>()){
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
        for (int i = 0; i < pcs.size(); ++i)
        {
            pcl::VoxelGrid<pcl::PointXYZI> pc_ds;
            pc_ds.setInputCloud(pcs[i]);
            pc_ds.setLeafSize(0.2f, 0.2f, 0.2f); // 体素大小
            pc_ds.filter(*pc_downsampled);
            pcl::copyPointCloud(*pc_downsampled, *pcs[i]); //直接覆盖原点云
        }
    }
    ////合并帧点云
    if (config["merge_frame"].as<bool>()){
#define merged_frames 3  //定义一个常量，合并5帧点云
        // Merge 3 frames point cloud
        for (int i = 0; i < pcs.size() - merged_frames - 1; ++i)
        {
            Eigen::Matrix4f T_velo_delt = Eigen::Matrix4f::Identity(); //声明一个转换矩阵
            for (int j = i + 1; j <= i + merged_frames - 1; ++j)
            {
                //使用ndt正太分布转换匹配方法
                pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
                ndt.setMaximumIterations(35);
                ndt.setTransformationEpsilon(0.1);
                ndt.setStepSize(0.1);
                ndt.setResolution(0.5);
                ndt.setInputSource(pcs[j]);
                ndt.setInputTarget(pcs[i]);
                pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_result_point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                ndt.align(*ndt_result_point_cloud_ptr, T_velo_delt);
                T_velo_delt = ndt.getFinalTransformation(); //导出转换矩阵

                for (int m = 0; m < pcs[j]->points.size(); ++m)  //这里->表示访问这个对象的属性，表示m小于这一帧点云的个数
                {
                    Eigen::Vector4f point_curr_frame(pcs[j]->points[m].x, pcs[j]->points[m].y, pcs[j]->points[m].z, 1);
                    Eigen::Vector4f point_last_frame = T_velo_delt * point_curr_frame; //将点逐个转换到last_frame这一帧中
                    // std::cout<<"point_curr_frame " << std::endl << std::setprecision(14) << std::fixed << point_curr_frame << std::endl;
                    // std::cout<<"point_last_frame " << std::endl << std::setprecision(14) << std::fixed << point_last_frame << std::endl;
                    pcl::PointXYZI point_temp;
                    point_temp.x = point_last_frame(0);
                    point_temp.y = point_last_frame(1);
                    point_temp.z = point_last_frame(2);
                    pcs[i]->push_back(point_temp); //第i帧补上这些点
                }
            }
        }
    }
    ////是否去除地面点云
    if (config["delete_ground"].as<bool>())
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr no_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
        for (int i = 0; i < pcs.size(); ++i){
            point_cb(pcs[i], no_ground_pc);
            pcl::copyPointCloud(*no_ground_pc, *pcs[i]);
        }
    }
//    ////将上述预处理过的点云可视化：
//    pcl::visualization::PCLVisualizer viewer("Viewer"); //创建可视化窗口
//    viewer.setBackgroundColor(0, 0, 0); //设置背景颜色为黑色
//    viewer.addCoordinateSystem(1); //添加坐标系
//    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pcs[2], "x"); //点云颜色处理
//    viewer.addPointCloud(pcs[0], fildColor, "cloud"); //将处理后的点云添加到可视化窗口中
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud"); //设置点云的点大小
//    viewer.resetCamera(); //重置相机
//    while (!viewer.wasStopped()) //用于保持窗口打开，直到用户关闭它
//    {
//        viewer.spinOnce();
//    }
    std::cerr << "Preprocessed!" << std::endl;
    return 0;
}


////提取点云
void PoseEstimation::extract_pc_feature_test(const YAML::Node &config,pcl::PointCloud<pcl::PointXYZI>::Ptr &pc,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature){
    int rings = config["rings"].as<int>();
    float angResolution = config["angleResolution"].as<float>(); //水平角分辨率假设为0.2度，
    int columns = config["columns"].as<int>(); //那么180会有900列
    float upperBound = config["upperBound"].as<float>();
    float lowerBound = config["lowerBound"].as<float>();
    float factor_t = ((upperBound - lowerBound) / (rings - 1)); //每两个线束多少度
    float factor = ((rings - 1) / (upperBound - lowerBound)); //rings是线束
    //创建图像，先将像素值初始化为-1
    std::vector<std::vector<float>> pc_image;
    std::vector<std::vector<float>> pc_image_copy;
    pc_image.resize(columns);
    pc_image_copy.resize(columns);
    for (int i = 0; i < pc_image.size(); i++)
    {
        pc_image[i].resize(rings);
        pc_image_copy[i].resize(rings);
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image[i][j] = -1; //先初始化为-1
    }
    ////转换3D点云为2D矩阵：[col][ring_id][dist]
    int min_col = columns - 1, max_col = 0; // 初始化列范围
    int min_ring_id = rings - 1, max_ring_id = 0; // 初始化环范围
    for (size_t i = 0; i < pc->size(); i++)
    {
        ////首先计算极角theta
        float theta = 0;
        if (pc->points[i].y == 0)
            theta = 90.0;
        else if (pc->points[i].y > 0)
        {
            float tan_theta = pc->points[i].x / pc->points[i].y;
            theta = 180 * std::atan(tan_theta) / M_PI;
        }
        else
        {
            float tan_theta = -pc->points[i].y / pc->points[i].x;
            theta = 180 * std::atan(tan_theta) / M_PI;
            theta = 90 + theta;
        }
        int col = cvFloor(theta / angResolution);
        // std::cout << "col " << col << std::endl;
        if (col < 0 || col > columns-1)
            continue;
        if (col < min_col) min_col = col;
        if (col > max_col) max_col = col;

        //然后计算线束索引
        float hypotenuse = std::sqrt(std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].y, 2)); //hypotenuse直角三角形的斜边
        float angle = std::atan(pc->points[i].z / hypotenuse); //通过反tan计算角度，单位弧度
        int ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5); //先将弧度转换度数，然后映射到线束的索引
        // std::cout << "ring_id " << ring_id << std::endl;
        if (ring_id < 0 || ring_id > rings - 1)
            continue;
        if (ring_id < min_ring_id) min_ring_id = ring_id;
        if (ring_id > max_ring_id) max_ring_id = ring_id;

        //然后用点到原心距离代替像素值
        float dist = std::sqrt(std::pow(pc->points[i].y, 2) + std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].z, 2));
        if (dist < 0.2) //距离小于阈值2，则跳过
            continue;
        if (pc_image[col][ring_id] == -1) //没有数据的话
        {
            pc_image[col][ring_id] = dist; //range
        }
        else if (dist < pc_image[col][ring_id]) //比原有点的距离值更小的话
        {
            pc_image[col][ring_id] = dist; //更新为更近距离值的点的距离值
        }
    }
    // 打印最大最小列和环的信息
    std::cout << "min_col: " << min_col << ", max_col: " << max_col << std::endl;
    std::cout << "min_ring_id: " << min_ring_id << ", max_ring_id: " << max_ring_id << std::endl;

    ////展示点云的2D图像，如果裁剪图像：
    int roi_width = max_col - min_col + 1;
    int roi_height = max_ring_id - min_ring_id + 1;
    cv::Mat pc_img = cv::Mat::zeros(roi_width, roi_height, CV_8UC1);
    float max_range = 0;
    int cnt = 0;
    std::cout<<"pc2img"<<std::endl;
    for(int i = min_col; i < max_col; i++){
        for(int j = min_ring_id; j < max_ring_id; j++){
            if((int)pc_image[i][j] > max_range) max_range = (int)pc_image[i][j];

        }
    }
    for(int i = min_col; i < max_col; i++){
        for(int j = min_ring_id; j < max_ring_id; j++){
            if (pc_image[i][j] == -1) {
                pc_image[i][j] = 0;
            } else {
                cnt++;
            }
            pc_image[i][j] = pc_image[i][j] / max_range * 255;
            pc_img.at<uchar>(i-min_col, j-min_ring_id) = (int)pc_image[i][j];
        }
    }
//    //// 如果不裁减图像：
//    cv::Mat pc_img = cv::Mat::zeros(columns, rings, CV_8UC1);
//    float max_range = 0;
//    int cnt = 0;
//    std::cout<<"pc2img："<<std::endl;
//    for(int i = 0; i < columns; i++){
//        for(int j = 0; j < rings; j++){
//            if((int)pc_image[i][j] > max_range) max_range = (int)pc_image[i][j];
//
//        }
//    }
//    for(int i = 0; i < columns; i++){
//        for(int j = 0; j < rings; j++){
//            if((int)pc_image[i][j] != -1) cnt++;
//            if((int)pc_image[i][j] == -1) pc_image[i][j] = 0;
//            pc_image[i][j] = pc_image[i][j] / max_range * 255;
//            pc_img.at<uchar>(i, j) = (int)pc_image[i][j];
//        }
//    }
//    std::cout<<"cnt = "<<cnt<<std::endl;
//    cv::imwrite("./src/joint_pose_estimation/results/pc_img.png", pc_img);
//    cv::namedWindow("pc_img", cv::WINDOW_NORMAL);
//    cv::imshow("pc_img", pc_img);
//    cv::waitKey(0);

/////////////////////////




}
void PoseEstimation::extract_pcd_points(const YAML::Node &config, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_points){
    //// 1.过滤点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    float x_min, x_max, y_min, y_max, z_min, z_max;
    x_min = config["x_min"].as<float>();
    x_max = config["x_max"].as<float>();
    y_min = config["y_min"].as<float>();
    y_max = config["y_max"].as<float>();
    z_min = config["z_min"].as<float>();
    z_max = config["z_max"].as<float>();
    pcl::PassThrough<pcl::PointXYZI> pass1;
    pass1.setInputCloud(cloud);
    pass1.setFilterFieldName("x"); //红轴
    pass1.setFilterLimits(x_min, x_max);
    pass1.filter(*filtered_cloud);
    pcl::PassThrough<pcl::PointXYZI> pass2;
    pass2.setInputCloud(filtered_cloud);
    pass2.setFilterFieldName("y"); //绿轴
    pass2.setFilterLimits(y_min,y_max);
    pass2.filter(*filtered_cloud);
    pcl::PassThrough<pcl::PointXYZI> pass3;
    pass3.setInputCloud(filtered_cloud);
    pass3.setFilterFieldName("z"); //蓝轴
    pass3.setFilterLimits(z_min, z_max);
    pass3.filter(*filtered_cloud);
//    pcl::io::savePCDFileBinary("test_filtered.pcd", *filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_copy(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*filtered_cloud, *filtered_cloud_copy);

    if (config["delete_ground_again"].as<bool>())
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr no_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
        point_cb(filtered_cloud, no_ground_pc);
        pcl::copyPointCloud(*no_ground_pc, *filtered_cloud);
    }

    ////2.平面分割
    // 平面分割
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    Eigen::Vector3f axis(1, 0, 0); //参考轴为y轴
    pcl::SACSegmentation<pcl::PointXYZI> plane_segmentation;
    plane_segmentation.setModelType(pcl::SACMODEL_PLANE); //SACMODEL_PLANE或者SACMODEL_PARALLEL_PLANE
    plane_segmentation.setDistanceThreshold(0.2); //点到拟合平面的最大允许距离
    plane_segmentation.setMethodType(pcl::SAC_RANSAC);
    plane_segmentation.setAxis(axis);
    plane_segmentation.setEpsAngle(0.2); //允许的法向量角度偏差范围（以弧度为单位）
    plane_segmentation.setOptimizeCoefficients(true);
    plane_segmentation.setMaxIterations(200);
    plane_segmentation.setInputCloud(filtered_cloud);
    plane_segmentation.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "No plane found!" << std::endl;
        return;
    }

    float a_final = coefficients->values[0] / coefficients->values[3];
    float b_final = coefficients->values[1] / coefficients->values[3];
    float c_final = coefficients->values[2] / coefficients->values[3];
    // 提取平面点云
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(filtered_cloud);
    extract.setIndices(inliers);
    extract.filter(*plane_cloud);

    ////3. 提取特征点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr edges_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (config["edge_extract_mode"].as<int>() == 0){
        // Step 1: 传入config参数
        int rings = config["rings"].as<int>();
        float upperBound = config["upperBound"].as<float>();
        float lowerBound = config["lowerBound"].as<float>();
        float board_width = config["board_width"].as<float>();
        float board_height = config["board_height"].as<float>();
        float threshold_multiplier = config["threshold_multiplier"].as<float>();
        float angle_range = (upperBound - lowerBound) / (rings - 1); // 每个线束的角度范围

        // Step 2: 遍历点云，将点分配到对应的线束层级,形成线束点云
        std::vector<pcl::PointCloud<pcl::PointXYZI>> plane_layers(rings);
        for (const auto& point : *plane_cloud) {
            // 计算点的角度（相对于Z轴）
            float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            float hypotenuse = std::sqrt(point.x * point.x + point.y * point.y);
            float angle = std::atan2(point.z, hypotenuse); // 使用atan2更稳定地计算角度

            // 确定点所属的线束层级
            if (angle >= lowerBound && angle <= upperBound) {
                int ring = int(((angle * 180 / M_PI) - lowerBound)/angle_range + 0.5);
                if (ring >= 0 && ring < rings) {
                    plane_layers[ring].push_back(point);
                }
            }
        }

        // Step 3: 遍历每个线束，计算宽度并筛选符合条件的线束点云
        std::vector<pcl::PointCloud<pcl::PointXYZI>> candidate_layers; // 存储候选点云带
        bool stop_scanning = false;
        for (int ring = 0; ring < rings && !stop_scanning; ++ring) {
            const auto& current_layer = plane_layers[ring];
            // 如果当前线束有点云，计算其宽度 (delta_y)
            if (!current_layer.empty()) {
                float min_y = std::numeric_limits<float>::max();
                float max_y = std::numeric_limits<float>::lowest();
                float min_z = std::numeric_limits<float>::max();
                float max_z = std::numeric_limits<float>::lowest();
                for (const auto &point: current_layer) {
                    if (point.y > max_y) max_y = point.y;
                    if (point.y < min_y) min_y = point.y;
                    if (point.z > max_z) max_z = point.z;
                    if (point.z < min_z) min_z = point.z;
                }
                float delta_y = max_y - min_y;
                float delta_z = max_z - min_z;

                // 判断宽度是否接近目标宽度
                if (delta_y > board_width * 0.8 && delta_y < board_width * 1.2) {
                    candidate_layers.push_back(current_layer); // 保存符合条件的点云
                } else if (delta_y > board_width * threshold_multiplier) {
                    // 如果宽度突然变大几倍，停止扫描
                    stop_scanning = true;
                }
                // 判断高度是否大于目标高度
                if (delta_z > board_height) {
                    stop_scanning = true;
                }
            }
        }

        //将候选点云加入到边缘点云
        for (const auto& layer : candidate_layers) {
            *edges_cloud += layer;
        }

        //// 4. 提取角点点云
        // 提取左上角点云和右上角、左下角和右下角点云
        pcl::PointXYZI left_top_point, right_top_point, left_bottom_point, right_bottom_point;
        // 获取顶部和底部层
        if (!candidate_layers.empty()) {
            const auto &top_layer = candidate_layers.back();   // 获取最后一层（顶部）
            const auto &bottom_layer = candidate_layers.front(); // 获取第一层（底部）

            // 提取 top_layer 中 y 最大和最小的点
            if (!top_layer.empty()) {
                left_top_point = top_layer[0];  // 初始化左上角点
                right_top_point = top_layer[0]; // 初始化右上角点

                for (const auto &point: top_layer) {
                    if (point.y > left_top_point.y) {
                        left_top_point = point;  // 更新左上角点
                    }
                    if (point.y < right_top_point.y) {
                        right_top_point = point; // 更新右上角点
                    }
                }
            }
            // 提取 bottom_layer 中 y 最大和最小的点
            if (!bottom_layer.empty()) {
                left_bottom_point = bottom_layer[0];  // 初始化左下角点
                right_bottom_point = bottom_layer[0]; // 初始化右下角点

                for (const auto &point: bottom_layer) {
                    if (point.y > left_bottom_point.y) {
                        left_bottom_point = point;  // 更新左下角点
                    }
                    if (point.y < right_bottom_point.y) {
                        right_bottom_point = point; // 更新右下角点
                    }
                }
            }
        }
        // 顺时针打印四个角点的坐标
//        std::cout << " Left  Top Point: (" << left_top_point.x << ", " << left_top_point.y << ", " << left_top_point.z << ")\n";
//        std::cout << " Right Top Point: (" << right_top_point.x << ", " << right_top_point.y << ", " << right_top_point.z << ")\n";
//        std::cout << " Right Bottom Point: (" << right_bottom_point.x << ", " << right_bottom_point.y << ", " << right_bottom_point.z << ")\n";
//        std::cout << " Left  Bottom Point: (" << left_bottom_point.x << ", " << left_bottom_point.y << ", " << left_bottom_point.z << ")\n";

        // 将四个角点按左上角点、右上角点、右下角点、左下角点顺时针存入点云
        pc_points->points.push_back(pcl::PointXYZ(left_top_point.x, left_top_point.y, left_top_point.z));
        pc_points->points.push_back(pcl::PointXYZ(right_top_point.x, right_top_point.y, right_top_point.z));
        pc_points->points.push_back(pcl::PointXYZ(right_bottom_point.x, right_bottom_point.y, right_bottom_point.z));
        pc_points->points.push_back(pcl::PointXYZ(left_bottom_point.x, left_bottom_point.y, left_bottom_point.z));
    }
    // 圆孔检测
    if (config["edge_extract_mode"].as<int>() == 1){
    }
    // 尝试新方法：是不是可以利用深度不连续/或者转为深度图提取边缘点
    if (config["edge_extract_mode"].as<int>() == 2){
        extract_pc_feature_test(config,plane_cloud,edges_cloud);
    }

    ////4. 点云可视化
    // 不显示
    if (config["pcd_show_mode"].as<int>() == 0){
        return;
    }
    // 显示范围内的
    if (config["pcd_show_mode"].as<int>() == 1){
        pcl::visualization::PCLVisualizer viewer("pcd viewer"); // 创建可视化窗口
        viewer.setBackgroundColor(0, 0, 0); // 设置背景颜色为黑色
        viewer.addCoordinateSystem(1); // 添加坐标系
        // 可视化原始点云
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(filtered_cloud_copy, "x"); // 点云颜色处理
        viewer.addPointCloud(filtered_cloud_copy, fildColor, "cloud"); // 将原始点云添加到可视化窗口中
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud"); // 设置点云的点大小
        viewer.resetCamera(); // 重置相机
        while (!viewer.wasStopped()) { // 用于保持窗口打开，直到用户关闭它
            viewer.spinOnce();
        }
    }
    // 显示板点云
    if (config["pcd_show_mode"].as<int>() == 2){
        pcl::visualization::PCLVisualizer viewer("pcd viewer"); // 创建可视化窗口
        viewer.setBackgroundColor(0, 0, 0); // 设置背景颜色为黑色
        viewer.addCoordinateSystem(1); // 添加坐标系
        // 可视化处理后的平面点云
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(edges_cloud, "x"); // 使用x坐标定色
        viewer.addPointCloud(edges_cloud, fildColor, "plane_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "plane_cloud");
        // 可视化角点
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> points_color_handler(pc_points, "x");
        viewer.addPointCloud(pc_points, points_color_handler, "points_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 11, "points_cloud");
        viewer.resetCamera(); // 重置相机
        while (!viewer.wasStopped()) { // 用于保持窗口打开，直到用户关闭它
            viewer.spinOnce();
        }
    }

}

//void PoseEstimation::sortPatternCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, std::vector<pcl::PointXYZ> &v) {
//    // 0 -- 1
//    // |    |
//    // 3 -- 2
//
//    if (v.empty()) {
//        v.resize(4);
//    }
//
//    // Transform points to polar coordinates
//    pcl::PointCloud<pcl::PointXYZ>::Ptr spherical_centers(
//            new pcl::PointCloud<pcl::PointXYZ>());
//    int top_pt = 0;
//    int index = 0;  // Auxiliar index to be used inside loop
//    for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = pc->points.begin();
//         pt < pc->points.end(); pt++, index++) {
//        pcl::PointXYZ spherical_center;
//        spherical_center.x = atan2(pt->y, pt->x);  // Horizontal
//        spherical_center.y =
//                atan2(sqrt(pt->x * pt->x + pt->y * pt->y), pt->z);  // Vertical
//        spherical_center.z =
//                sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);  // Range
//        spherical_centers->push_back(spherical_center);
//
//        if (spherical_center.y < spherical_centers->points[top_pt].y) {
//            top_pt = index;
//        }
//    }
//
//    // Compute distances from top-most center to rest of points
//    vector<double> distances;
//    for (int i = 0; i < 4; i++) {
//        pcl::PointXYZ pt = pc->points[i];
//        pcl::PointXYZ upper_pt = pc->points[top_pt];
//        distances.push_back(sqrt(pow(pt.x - upper_pt.x, 2) +
//                                 pow(pt.y - upper_pt.y, 2) +
//                                 pow(pt.z - upper_pt.z, 2)));
//    }
//
//    // Get indices of closest and furthest points
//    int min_dist = (top_pt + 1) % 4, max_dist = top_pt;
//    for (int i = 0; i < 4; i++) {
//        if (i == top_pt) continue;
//        if (distances[i] > distances[max_dist]) {
//            max_dist = i;
//        }
//        if (distances[i] < distances[min_dist]) {
//            min_dist = i;
//        }
//    }
//
//    // Second highest point shoud be the one whose distance is the median value
//    int top_pt2 = 6 - (top_pt + max_dist + min_dist);  // 0 + 1 + 2 + 3 = 6
//
//    // Order upper row centers
//    int lefttop_pt = top_pt;
//    int righttop_pt = top_pt2;
//
//    if (spherical_centers->points[top_pt].x <
//        spherical_centers->points[top_pt2].x) {
//        int aux = lefttop_pt;
//        lefttop_pt = righttop_pt;
//        righttop_pt = aux;
//    }
//
//    // Swap indices if target is located in the pi,-pi discontinuity
//    double angle_diff = spherical_centers->points[lefttop_pt].x -
//                        spherical_centers->points[righttop_pt].x;
//    if (angle_diff > M_PI - spherical_centers->points[lefttop_pt].x) {
//        int aux = lefttop_pt;
//        lefttop_pt = righttop_pt;
//        righttop_pt = aux;
//    }
//
//    // Define bottom row centers using lefttop == top_pt as hypothesis
//    int leftbottom_pt = min_dist;
//    int rightbottom_pt = max_dist;
//
//    // If lefttop != top_pt, swap indices
//    if (righttop_pt == top_pt) {
//        leftbottom_pt = max_dist;
//        rightbottom_pt = min_dist;
//    }
//
//    // Fill vector with sorted centers
//    v[0] = pc->points[lefttop_pt];      // lt
//    v[1] = pc->points[righttop_pt];     // rt
//    v[2] = pc->points[rightbottom_pt];  // rb
//    v[3] = pc->points[leftbottom_pt];   // lb
//}
