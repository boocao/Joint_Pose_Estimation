#include "funcs.h"
#include "calculate_r_t.h"
#include "calculate_r_t_3d.h"

int main(int argc, char **argv)
{
    RuntimeStats mode3_total_stats;
    RuntimeStats data_prepare_stats;
    RuntimeStats data_load_stats;
    RuntimeStats image_extract_stats;
    RuntimeStats pointcloud_extract_stats;
    RuntimeStats fusion_stats;
    RuntimeStats extrinsic_stats;
    RuntimeStats pose_stats;

    ros::init(argc, argv, "joint_pose_estimation");
    ros::Time::init();

    std::string config_dir="./src/joint_pose_estimation/config_bag.yaml";
    YAML::Node config_node = YAML::LoadFile(config_dir);

    std::string r_t_file_path = "/home/bocao/mv_codes/ws_joint_pose_estimation/src/joint_pose_estimation/results/sensors_r_t.txt";
    std::ofstream clear_r_t_file(r_t_file_path, std::ios::out); // 打开文件并清空内容
    clear_r_t_file.close();
    std::ofstream r_t_file(r_t_file_path, std::ios::app);
    std::string pose_file_path = "/home/bocao/mv_codes/ws_joint_pose_estimation/src/joint_pose_estimation/results/vehicle_pose.txt";
    std::ofstream clear_pose_file(pose_file_path, std::ios::out); // 打开文件并清空内容
    clear_pose_file.close();
    std::ofstream pose_file(pose_file_path, std::ios::app);
    std::string pixel_file_path = "/home/bocao/mv_codes/ws_joint_pose_estimation/src/joint_pose_estimation/results/pixel_feature_points.txt";
    std::ofstream clear_pixel_file(pixel_file_path, std::ios::out); // 打开文件并清空内容
    clear_pixel_file.close();
    std::ofstream pixel_file(pixel_file_path, std::ios::app);
    std::string cam_file_path = "/home/bocao/mv_codes/ws_joint_pose_estimation/src/joint_pose_estimation/results/camera_feature_points.txt";
    std::ofstream clear_cam_file(cam_file_path, std::ios::out); // 打开文件并清空内容
    clear_cam_file.close();
    std::ofstream cam_file(cam_file_path, std::ios::app);
    std::string lidar_file_path = "/home/bocao/mv_codes/ws_joint_pose_estimation/src/joint_pose_estimation/results/lidar_feature_points.txt";
    std::ofstream clear_lidar_file(lidar_file_path, std::ios::out); // 打开文件并清空内容
    clear_lidar_file.close();
    std::ofstream lidar_file(lidar_file_path, std::ios::app);

    ////////第一块，提取特征匹配点：
    const auto data_prepare_start = std::chrono::steady_clock::now();
    int temp;
    PoseEstimation app(config_node, temp);

    //获取特征角点
    std::vector<cv::Mat> images; //声明含有所有帧图像数组的向量
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs;  //声明含有多个帧点云的向量
    int dataNum = 0;
    {
        ScopedTimer timer(data_load_stats);
        dataNum = app.getData(config_node,images,pcs);
    }
    std::cout << "finish read data:" << dataNum << std::endl;

    // app.preprocess_point_clouds(config_node,pcs);

    std::vector<std::vector<cv::Point2f>> imgs_corners_temp;
    std::vector<std::vector<cv::Point3f>> imgs_corners_3d_temp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcs_points_temp;
    for (int i = 0; i < images.size(); i++)
    {
        //提取图像角点
        cv::Mat image_copy = images[i].clone();
        cv::Mat img_gray;
        cv::cvtColor(image_copy, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> img_corners_temp;
        std::vector<cv::Point3f> img_corners_3d_temp;

        {
            ScopedTimer timer(image_extract_stats);
            app.extract_image_points_four(config_node,image_copy, img_corners_temp,img_corners_3d_temp);
        }
        imgs_corners_temp.push_back(img_corners_temp);
        imgs_corners_3d_temp.push_back(img_corners_3d_temp);
        cv::imwrite("./src/joint_pose_estimation/results/imgs/" + std::to_string(i) + ".jpg", image_copy);
        if (!img_corners_temp.empty()) {
            // 计算图像像素坐标下的中心点
            cv::Point2f center(0, 0);
            for (const auto &point : img_corners_temp) {
                center.x += point.x;
                center.y += point.y;
            }
            center.x /= img_corners_temp.size();
            center.y /= img_corners_temp.size();

            if (pixel_file.is_open()) {
                for (const auto &point : img_corners_temp) {
                    pixel_file << point.x << "," << point.y << ",";
                }
                pixel_file << center.x << "," << center.y << "\n";
            } else {
                std::cerr << "Error: Unable to open pixel.txt for writing.\n";
            }
        }
        // 提取点云角点
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_points_temp(new pcl::PointCloud<pcl::PointXYZ>);
        {
            ScopedTimer timer(pointcloud_extract_stats);
            app.extract_pcd_points(config_node,pcs[i], pc_points_temp);
        }
        if (pc_points_temp->points.size() < 4) {
            std::cerr << "Warning: Point cloud " << i << " has only " << pc_points_temp->points.size()
            << " points, need at least 4 points." << std::endl;
        }
        pcs_points_temp.push_back(pc_points_temp);

    }
    const auto data_prepare_end = std::chrono::steady_clock::now();
    data_prepare_stats.add(std::chrono::duration<double, std::milli>(data_prepare_end - data_prepare_start).count());

    ////////第二块：基于上述特征角点开始操作：
    param_init(config_node);
    std::vector<cv::Point2f> imgs_corners_selected; //用于计算外参
    std::vector<cv::Point3f> imgs_corners_3d_selected;
    std::vector<cv::Point3f> pcs_points_selected;
    std::vector<cv::Point3f> img_corners_pose; //用于计算位姿
    std::vector<cv::Point3f> pc_points_pose;
    cv::Point3f center, normal_vector; //导出位姿

    //// 如果固定外参，再计算位姿：
    if (config_node["estimation_mode"].as<int>() == 0){
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_filtered;
        std::vector<std::vector<cv::Point3f>> pcs_points_filtered;

        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::vector <cv::Point3f> img_filtered(4);
            std::vector <cv::Point3f> pc_filtered(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);

                img_filtered[k] = img_pt;
                pc_filtered[k] = pc_pt_cv;

                std::cout << "pc_pt_cv[" << k << "] = ("
                          << pc_pt_cv.x << ", "
                          << pc_pt_cv.y << ", "
                          << pc_pt_cv.z << ")" << std::endl;
                std::cout << "pc_filtered[" << k << "] = ("
                          << pc_filtered[k].x << ", "
                          << pc_filtered[k].y << ", "
                          << pc_filtered[k].z << ")" << std::endl;
            }

            imgs_corners_3d_filtered.push_back(img_filtered);
            pcs_points_filtered.push_back(pc_filtered);
            if (!img_filtered.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_filtered.size();
                center.y /= img_filtered.size();
                center.z /= img_filtered.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_filtered) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_filtered.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_filtered.size();
                center.y /= pc_filtered.size();
                center.z /= pc_filtered.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_filtered) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            //// 提取平移分量
            t[0] = -1.6;
            t[1] = -0.5;
            t[2] = 0.5;
            //// Rodrigues: 旋转向量 -> 旋转矩阵 (OpenCV)
            cv::Mat r_true = (cv::Mat_<double>(3, 1) << 1.3, -1.0, 1.0);
            cv::Mat cv_rotationMatrix(3, 3, CV_64F);
            Rodrigues(r_true, cv_rotationMatrix);
            // OpenCV 矩阵转 Eigen 矩阵
            Eigen::Matrix3d mat3x3;
            cv2eigen(cv_rotationMatrix, mat3x3);
            // 用旋转矩阵构造四元数
            Eigen::Quaterniond quaternion(mat3x3);
            quaternion.normalize();  // 数值稳定，推荐加上
            // 提取四元数分量：q[0]=w, q[1]=x, q[2]=y, q[3]=z
            q[0] = quaternion.w();
            q[1] = quaternion.x();
            q[2] = quaternion.y();
            q[3] = quaternion.z();

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";
            //// 2.2 计算这些点的位姿
            img_corners_pose = imgs_corners_3d_filtered[i];
            pc_points_pose = pcs_points_filtered[i];
            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);

            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }
    //// 如果不做处理，直接3D-3D点计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 1){
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_filtered;
        std::vector<std::vector<cv::Point3f>> pcs_points_filtered;

        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::vector <cv::Point3f> img_filtered(4);
            std::vector <cv::Point3f> pc_filtered(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);

                img_filtered[k] = img_pt;
                pc_filtered[k] = pc_pt_cv;

                std::cout << "pc_pt_cv[" << k << "] = ("
                          << pc_pt_cv.x << ", "
                          << pc_pt_cv.y << ", "
                          << pc_pt_cv.z << ")" << std::endl;
                std::cout << "pc_filtered[" << k << "] = ("
                          << pc_filtered[k].x << ", "
                          << pc_filtered[k].y << ", "
                          << pc_filtered[k].z << ")" << std::endl;
            }

            imgs_corners_3d_filtered.push_back(img_filtered);
            pcs_points_filtered.push_back(pc_filtered);
            if (!img_filtered.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_filtered.size();
                center.y /= img_filtered.size();
                center.z /= img_filtered.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_filtered) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_filtered.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_filtered.size();
                center.y /= pc_filtered.size();
                center.z /= pc_filtered.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_filtered) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_filtered[j][k]);
                    pcs_points_selected.push_back(pcs_points_filtered[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            //// solver全局计算外参参数t，q
//            calcul_r_t_3d(pcs_points_selected,imgs_corners_3d_selected, t, q);
//            calcul_r_t_3d_with_constraints(pcs_points_selected,imgs_corners_3d_selected, t, q);

            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = imgs_corners_3d_filtered[i];
            pc_points_pose = pcs_points_filtered[i];

            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }
    //// 如果先转进相机空间，再加权融合，再转回相机空间；再计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 2){
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_fused;
        std::vector<std::vector<cv::Point3f>> pcs_points_fused;
        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::cout << "=== frame: " << i << " ==="<< std::endl;

            if (i == 0) {
                // 提取前四个角点作为初始图像点集
                std::vector <cv::Point3f> img_init(imgs_corners_3d_temp[i].begin(), imgs_corners_3d_temp[i].begin() + 4);
                std::vector <cv::Point3f> pc_init;
                boost::shared_ptr <pcl::PointCloud<pcl::PointXYZ>> pc_shared_ptr = pcs_points_temp[i];
                for (size_t j = 0; j < 4; ++j) {
                    const pcl::PointXYZ &point = pc_shared_ptr->points[j];
                    pc_init.emplace_back(point.x, point.y, point.z);
                }
                // 调用 initialize_with_horn 初始化 t 和 q
                initialize_with_horn(pc_init, img_init, t, q);
            }

            float alpha = 0.7f; // 相机点修正权重
            float beta  = 0.7f; // 雷达点修正权重
            std::vector <cv::Point3f> img_fused(4);
            std::vector <cv::Point3f> pc_fused(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);
                cv::Point3f pc_pt_into_cam = transform_point(pc_pt_cv, t, q);

                // 融合点（相机系下）
                cv::Point3f fused_pt;
                fused_pt.x = (img_pt.x + pc_pt_into_cam.x) * 0.5f;
                fused_pt.y = (img_pt.y + pc_pt_into_cam.y) * 0.5f;
                fused_pt.z = (img_pt.z + pc_pt_into_cam.z) * 0.5f;

                // ---- 修正相机点 ----
                img_fused[k].x = (1 - alpha) * img_pt.x + alpha * fused_pt.x;
                img_fused[k].y = (1 - alpha) * img_pt.y + alpha * fused_pt.y;
                img_fused[k].z = (1 - alpha) * img_pt.z + alpha * fused_pt.z;

                // ---- 修正雷达点 ----
                cv::Point3f pc_pt_back = inverse_transform_point(fused_pt, t, q); // 融合点转回雷达系
                pc_fused[k].x = (1 - beta) * pc_pt_cv.x + beta * pc_pt_back.x;
                pc_fused[k].y = (1 - beta) * pc_pt_cv.y + beta * pc_pt_back.y;
                pc_fused[k].z = (1 - beta) * pc_pt_cv.z + beta * pc_pt_back.z;
            }

            imgs_corners_3d_fused.push_back(img_fused);
            pcs_points_fused.push_back(pc_fused);
            if (!img_fused.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_fused.size();
                center.y /= img_fused.size();
                center.z /= img_fused.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_fused) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_fused.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_fused.size();
                center.y /= pc_fused.size();
                center.z /= pc_fused.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_fused) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_fused[j][k]);
                    pcs_points_selected.push_back(pcs_points_fused[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            //// solver全局计算外参参数t，q
//            calcul_r_t_3d(pcs_points_selected,imgs_corners_3d_selected, t, q);
//            calcul_r_t_3d_with_constraints(pcs_points_selected,imgs_corners_3d_selected, t, q);

            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = img_fused;
            pc_points_pose = pc_fused;

            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }
    //// 如果先转进相机空间，再一起滤波、加权融合，再转回相机空间；再计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 3){
        const auto mode3_start = std::chrono::steady_clock::now();
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_fused;
        std::vector<std::vector<cv::Point3f>> pcs_points_fused;
        std::vector<KalmanFilter3D3D> joint_filters(4);
        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::cout << "=== frame: " << i << " ==="<< std::endl;
            if (i == 0) {
                // 提取前四个角点作为初始图像点集
                std::vector <cv::Point3f> img_init(imgs_corners_3d_temp[i].begin(), imgs_corners_3d_temp[i].begin() + 4);
                std::vector <cv::Point3f> pc_init;
                boost::shared_ptr <pcl::PointCloud<pcl::PointXYZ>> pc_shared_ptr = pcs_points_temp[i];
                for (size_t j = 0; j < 4; ++j) {
                    const pcl::PointXYZ &point = pc_shared_ptr->points[j];
                    pc_init.emplace_back(point.x, point.y, point.z);
                }
                // 调用 initialize_with_horn 初始化 t 和 q
                initialize_with_horn(pc_init, img_init, t, q);
            }

            float alpha = 0.7f; // 相机点修正权重
            float beta  = 0.7f; // 雷达点修正权重
            std::vector <cv::Point3f> img_fused(4);
            std::vector <cv::Point3f> pc_fused(4);
            {
                ScopedTimer timer(fusion_stats);
                for (size_t k = 0; k < 4; ++k) {
                    const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                    const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                    cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);
                    cv::Point3f pc_pt_into_cam = transform_point(pc_pt_cv, t, q);

                    if (!joint_filters[k].isInitialized()) {
                        joint_filters[k].init(img_pt, pc_pt_into_cam);
                    }

                    std::pair <cv::Point3f, cv::Point3f> result = joint_filters[k].update(img_pt, pc_pt_into_cam);
                    cv::Point3f img_filtered_pt = result.first;
                    cv::Point3f pc_filtered_pt_into_cam = result.second;

                    // 融合点（相机系下）
                    cv::Point3f fused_pt;
                    fused_pt.x = (img_pt.x + pc_pt_into_cam.x) * 0.5f;
                    fused_pt.y = (img_pt.y + pc_pt_into_cam.y) * 0.5f;
                    fused_pt.z = (img_pt.z + pc_pt_into_cam.z) * 0.5f;

                    // ---- 修正相机点 ----
                    img_fused[k].x = (1 - alpha) * img_pt.x + alpha * fused_pt.x;
                    img_fused[k].y = (1 - alpha) * img_pt.y + alpha * fused_pt.y;
                    img_fused[k].z = (1 - alpha) * img_pt.z + alpha * fused_pt.z;

                    // ---- 修正雷达点 ----
                    cv::Point3f pc_pt_back = inverse_transform_point(fused_pt, t, q); // 融合点转回雷达系
                    pc_fused[k].x = (1 - beta) * pc_pt_cv.x + beta * pc_pt_back.x;
                    pc_fused[k].y = (1 - beta) * pc_pt_cv.y + beta * pc_pt_back.y;
                    pc_fused[k].z = (1 - beta) * pc_pt_cv.z + beta * pc_pt_back.z;
                }
            }

            imgs_corners_3d_fused.push_back(img_fused);
            pcs_points_fused.push_back(pc_fused);
            if (!img_fused.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_fused.size();
                center.y /= img_fused.size();
                center.z /= img_fused.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_fused) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_fused.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_fused.size();
                center.y /= pc_fused.size();
                center.z /= pc_fused.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_fused) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 20;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = i-2*num; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_fused[j][k]);
                    pcs_points_selected.push_back(pcs_points_fused[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            {
                ScopedTimer timer(extrinsic_stats);
                initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);
            }

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            //// solver全局计算外参参数t，q
//            calcul_r_t_3d(pcs_points_selected,imgs_corners_3d_selected, t, q);
//            calcul_r_t_3d_with_constraints(pcs_points_selected,imgs_corners_3d_selected, t, q);

            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = img_fused;
            pc_points_pose = pc_fused;

            {
                ScopedTimer timer(pose_stats);
                calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            }
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            ////////////////
            // 将3D点投影到图像上并绘制
            cv::Mat img_with_pose = images[i];
            // 1. 投影中心点到图像
            std::vector<cv::Point3f> points_3d = {center};
            std::vector<cv::Point2f> points_2d;
            cv::projectPoints(points_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F),
                              camera_instrinsics_, distortion_coefficients_, points_2d);
            if (!points_2d.empty()) {
                float axis_length = 0.3f; // 坐标轴长度
                float arrow_length = 0.8f; // 法向量箭头长度

                // 3. 绘制法向量箭头 - 黑色
                cv::Point3f normal_end = center + cv::Point3f(
                        normal_vector.x * arrow_length,
                        normal_vector.y * arrow_length,
                        normal_vector.z * arrow_length
                );
                std::vector<cv::Point3f> normal_3d = {normal_end};
                std::vector<cv::Point2f> normal_2d;
                cv::projectPoints(normal_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F),
                                  camera_instrinsics_, distortion_coefficients_, normal_2d);

                if (!normal_2d.empty()) {
                    cv::arrowedLine(img_with_pose, points_2d[0], normal_2d[0], cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0, 0.3);
//                    cv::putText(img_with_pose, "N", normal_2d[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
                }

                // 4. 绘制中心点 - 白色圆圈
                cv::circle(img_with_pose, points_2d[0], 6, cv::Scalar(255, 255, 255), -1);
                cv::circle(img_with_pose, points_2d[0], 8, cv::Scalar(0, 255, 0), 2);

                // 5. 可选：添加坐标文字标注
                std::string text = cv::format("C(%.2f,%.2f,%.2f) N(%.2f,%.2f,%.2f)",
                                              center.x, center.y, center.z, normal_vector.x, normal_vector.y, normal_vector.z);
                cv::putText(img_with_pose, text, cv::Point(points_2d[0].x + 15, points_2d[0].y - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 3);
            }
            // 4. 保存图像
            cv::imwrite("./src/joint_pose_estimation/results/imgs_pose/" + std::to_string(i) + ".jpg", img_with_pose);
            ////////////////

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
        const auto mode3_end = std::chrono::steady_clock::now();
        mode3_total_stats.add(std::chrono::duration<double, std::milli>(mode3_end - mode3_start).count());

        std::cout << "\n========== Runtime Summary (estimation_mode 3) ==========" << std::endl;
        print_runtime_stats("Data preparation total", data_prepare_stats);
        print_runtime_stats("Data loading", data_load_stats);
        print_runtime_stats("Image corner extraction", image_extract_stats);
        print_runtime_stats("Point cloud corner extraction", pointcloud_extract_stats);
        print_runtime_stats("Mode 3 total", mode3_total_stats);
        print_runtime_stats("Fusion/filtering", fusion_stats);
        print_runtime_stats("Extrinsic estimation", extrinsic_stats);
        print_runtime_stats("Pose estimation", pose_stats);
        std::cout << "=========================================================" << std::endl;
    }

    //// (不太行)如果先分别使用kalman滤波，再转进相机空间，加权融合，再转回相机空间；再计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 4){
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_fused;
        std::vector<std::vector<cv::Point3f>> pcs_points_fused;
        std::vector<KalmanFilter3D> img_filters(4);
        std::vector<KalmanFilter3D> pcd_filters(4);
        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::cout << "=== frame: " << i << " ==="<< std::endl;

            if (i == 0) {
                // 提取前四个角点作为初始图像点集
                std::vector <cv::Point3f> img_init(imgs_corners_3d_temp[i].begin(), imgs_corners_3d_temp[i].begin() + 4);
                std::vector <cv::Point3f> pc_init;
                boost::shared_ptr <pcl::PointCloud<pcl::PointXYZ>> pc_shared_ptr = pcs_points_temp[i];
                for (size_t j = 0; j < 4; ++j) {
                    const pcl::PointXYZ &point = pc_shared_ptr->points[j];
                    pc_init.emplace_back(point.x, point.y, point.z);
                }
                // 调用 initialize_with_horn 初始化 t 和 q
                initialize_with_horn(pc_init, img_init, t, q);
            }

            float alpha = 0.7f; // 相机点修正权重
            float beta  = 0.7f; // 雷达点修正权重
            std::vector <cv::Point3f> img_fused(4);
            std::vector <cv::Point3f> pc_fused(4);
            std::vector <cv::Point3f> img_filtered(4);
            std::vector <cv::Point3f> pc_filtered(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);

                // 初始化滤波器
                if (!img_filters[k].isInitialized()) img_filters[k].init(img_pt);
                if (!pcd_filters[k].isInitialized()) pcd_filters[k].init(pc_pt_cv);
                // 更新滤波器，跟踪数据
                img_filtered[k] = img_filters[k].update(img_pt);
                pc_filtered[k] = pcd_filters[k].update(pc_pt_cv);

                cv::Point3f &img_pt_cv = img_filtered[k];
                pc_pt_cv = pc_filtered[k];
                cv::Point3f pc_pt_into_cam = transform_point(pc_pt_cv, t, q);

                // 融合点（相机系下）
                cv::Point3f fused_pt;
                fused_pt.x = (img_pt_cv.x + pc_pt_into_cam.x) * 0.5f;
                fused_pt.y = (img_pt_cv.y + pc_pt_into_cam.y) * 0.5f;
                fused_pt.z = (img_pt_cv.z + pc_pt_into_cam.z) * 0.5f;

                // ---- 修正相机点 ----
                img_fused[k].x = (1 - alpha) * img_pt_cv.x + alpha * fused_pt.x;
                img_fused[k].y = (1 - alpha) * img_pt_cv.y + alpha * fused_pt.y;
                img_fused[k].z = (1 - alpha) * img_pt_cv.z + alpha * fused_pt.z;

                // ---- 修正雷达点 ----
                cv::Point3f pc_pt_back = inverse_transform_point(pc_pt_into_cam, t, q); // 融合点转回雷达系
                pc_fused[k].x = (1 - beta) * pc_pt_cv.x + beta * pc_pt_back.x;
                pc_fused[k].y = (1 - beta) * pc_pt_cv.y + beta * pc_pt_back.y;
                pc_fused[k].z = (1 - beta) * pc_pt_cv.z + beta * pc_pt_back.z;
            }

            imgs_corners_3d_fused.push_back(img_fused);
            pcs_points_fused.push_back(pc_fused);
            if (!img_fused.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_fused.size();
                center.y /= img_fused.size();
                center.z /= img_fused.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_fused) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_fused.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_fused) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_fused.size();
                center.y /= pc_fused.size();
                center.z /= pc_fused.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_fused) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_fused[j][k]);
                    pcs_points_selected.push_back(pcs_points_fused[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            //// solver全局计算外参参数t，q
//            calcul_r_t_3d(pcs_points_selected,imgs_corners_3d_selected, t, q);
//            calcul_r_t_3d_with_constraints(pcs_points_selected,imgs_corners_3d_selected, t, q);

            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = img_fused;
            pc_points_pose = pc_fused;

            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }
    //// 如果分别使用kalman滤波；再计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 5){
        std::vector<KalmanFilter3D> img_filters(4);
        std::vector<KalmanFilter3D> pcd_filters(4);
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_filtered;
        std::vector<std::vector<cv::Point3f>> pcs_points_filtered;

        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::vector <cv::Point3f> img_filtered(4);
            std::vector <cv::Point3f> pc_filtered(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);

                // 初始化滤波器
                if (!img_filters[k].isInitialized()) img_filters[k].init(img_pt);
                if (!pcd_filters[k].isInitialized()) pcd_filters[k].init(pc_pt_cv);
                // 更新滤波器，跟踪数据
                img_filtered[k] = img_filters[k].update(img_pt);
                pc_filtered[k] = pcd_filters[k].update(pc_pt_cv);

                std::cout << "pc_pt_cv[" << k << "] = ("
                          << pc_pt_cv.x << ", "
                          << pc_pt_cv.y << ", "
                          << pc_pt_cv.z << ")" << std::endl;
                std::cout << "pc_filtered[" << k << "] = ("
                          << pc_filtered[k].x << ", "
                          << pc_filtered[k].y << ", "
                          << pc_filtered[k].z << ")" << std::endl;
            }

            imgs_corners_3d_filtered.push_back(img_filtered);
            pcs_points_filtered.push_back(pc_filtered);
            if (!img_filtered.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_filtered.size();
                center.y /= img_filtered.size();
                center.z /= img_filtered.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_filtered) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_filtered.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_filtered.size();
                center.y /= pc_filtered.size();
                center.z /= pc_filtered.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_filtered) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices;
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0;
                size_t end_index = i;
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_filtered[j][k]);
                    pcs_points_selected.push_back(pcs_points_filtered[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = imgs_corners_3d_filtered[i];
            pc_points_pose = pcs_points_filtered[i];
            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);

            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }
    //// 如果先转进相机空间，再使用kalman滤波一起估计，再转回相机空间；再计算外参、位姿：
    if (config_node["estimation_mode"].as<int>() == 6){
        std::vector<std::vector<cv::Point3f>> imgs_corners_3d_filtered;
        std::vector<std::vector<cv::Point3f>> pcs_points_filtered;
        std::vector<KalmanFilter3D3D> joint_filters(4);
        for (size_t i = 0; i < imgs_corners_3d_temp.size(); ++i) {
            std::cout << "=== frame: " << i << " ==="<< std::endl;

            if (i == 0) {
                // 提取前四个角点作为初始图像点集
                std::vector <cv::Point3f> img_init(imgs_corners_3d_temp[i].begin(), imgs_corners_3d_temp[i].begin() + 4);
                std::vector <cv::Point3f> pc_init;
                boost::shared_ptr <pcl::PointCloud<pcl::PointXYZ>> pc_shared_ptr = pcs_points_temp[i];
                for (size_t j = 0; j < 4; ++j) {
                    const pcl::PointXYZ &point = pc_shared_ptr->points[j];
                    pc_init.emplace_back(point.x, point.y, point.z);
                }
                // 调用 initialize_with_horn 初始化 t 和 q
                initialize_with_horn(pc_init, img_init, t, q);
            }

            std::vector <cv::Point3f> img_filtered(4);
            std::vector <cv::Point3f> pc_filtered(4);
            for (size_t k = 0; k < 4; ++k) {
                const cv::Point3f &img_pt = imgs_corners_3d_temp[i][k];
                const pcl::PointXYZ &pc_pt = pcs_points_temp[i]->points[k];
                cv::Point3f pc_pt_cv(pc_pt.x, pc_pt.y, pc_pt.z);
                cv::Point3f pc_pt_into_cam = transform_point(pc_pt_cv, t, q);

                if (!joint_filters[k].isInitialized()) {
                    joint_filters[k].init(img_pt, pc_pt_into_cam);
                }

                std::pair <cv::Point3f, cv::Point3f> result = joint_filters[k].update(img_pt, pc_pt_into_cam);
                cv::Point3f img_filtered_pt = result.first;
                cv::Point3f pc_filtered_pt_into_cam = result.second;

                img_filtered[k] = img_filtered_pt;
                pc_filtered[k] = inverse_transform_point(pc_filtered_pt_into_cam, t, q);
            }

            imgs_corners_3d_filtered.push_back(img_filtered);
            pcs_points_filtered.push_back(pc_filtered);
            if (!img_filtered.empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: img_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= img_filtered.size();
                center.y /= img_filtered.size();
                center.z /= img_filtered.size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: img_filtered) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pc_filtered.empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pc_filtered) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pc_filtered.size();
                center.y /= pc_filtered.size();
                center.z /= pc_filtered.size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pc_filtered) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j: selected_indices) {
                for (size_t k = 0; k < 4; ++k) {
                    imgs_corners_3d_selected.push_back(imgs_corners_3d_filtered[j][k]);
                    pcs_points_selected.push_back(pcs_points_filtered[j][k]);
                }
            }
            std::cout << "图像点数量: " << imgs_corners_3d_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            //// 初始计算外参参数t，q
//            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.begin(), imgs_corners_3d_selected.end());
//            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.begin(), pcs_points_selected.end());
            std::vector<cv::Point3f> imgs_subset_end(imgs_corners_3d_selected.end()-4, imgs_corners_3d_selected.end());
            std::vector<cv::Point3f> pcs_subset_end(pcs_points_selected.end()-4, pcs_points_selected.end());
//            initialize_with_umeyama(pcs_subset_end, imgs_subset_end, t, q);
            initialize_with_horn(pcs_subset_end, imgs_subset_end, t, q);

            std::cout << "imgs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)imgs_subset_end.size() - 4); i < imgs_subset_end.size(); ++i) {
                const auto &pt = imgs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end (last 4):" << std::endl;
            for (size_t i = std::max(0, (int)pcs_subset_end.size() - 4); i < pcs_subset_end.size(); ++i) {
                const auto &pt = pcs_subset_end[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
            std::cout << "pcs_subset_end_transformed_horn (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_horn = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_horn.size() - 4); i < pcs_subset_end_transformed_horn.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_horn[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            //// solver全局计算外参参数t，q
//            calcul_r_t_3d(pcs_points_selected,imgs_corners_3d_selected, t, q);
//            calcul_r_t_3d_with_constraints(pcs_points_selected,imgs_corners_3d_selected, t, q);

            std::cout << "pcs_subset_end_transformed_solver (last 4):" << std::endl;
            std::vector<cv::Point3f> pcs_subset_end_transformed_solver = transform_points(pcs_subset_end, t, q);
            for (size_t i = std::max(0, (int)pcs_subset_end_transformed_solver.size() - 4); i < pcs_subset_end_transformed_solver.size(); ++i) {
                const auto &pt = pcs_subset_end_transformed_solver[i];
                std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }

            r_t_file << t[0] << ", " << t[1] << ", " << t[2] << ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose = imgs_corners_3d_filtered[i];
            pc_points_pose = pcs_points_filtered[i];

            calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_3d_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
    }

    //// 2D3D直接求外参后计算位姿：
    if (config_node["estimation_mode"].as<int>() == 9){
        RuntimeStats mode9_total_stats;
        const auto mode9_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < imgs_corners_temp.size(); ++i){
            if (!imgs_corners_3d_temp[i].empty()){
                cv::Point3f center(0, 0, 0);
                for (const auto &point: imgs_corners_3d_temp[i]) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= imgs_corners_3d_temp[i].size();
                center.y /= imgs_corners_3d_temp[i].size();
                center.z /= imgs_corners_3d_temp[i].size();
                if (cam_file.is_open()) {
                    // 先输出 4 个角点坐标
                    for (const auto &point: imgs_corners_3d_temp[i]) {
                        cam_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    // 再输出中心点坐标
                    cam_file << center.x << ", " << center.y << ", " << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open file for writing.\n";
                }
            }
            if (!pcs_points_temp[i]->empty()) {
                cv::Point3f center(0, 0, 0);
                for (const auto &point : pcs_points_temp[i]->points) {
                    center.x += point.x;
                    center.y += point.y;
                    center.z += point.z;
                }
                center.x /= pcs_points_temp[i]->size();
                center.y /= pcs_points_temp[i]->size();
                center.z /= pcs_points_temp[i]->size();
                if (lidar_file.is_open()) {
                    for (const auto &point : pcs_points_temp[i]->points) {
                        lidar_file << point.x << "," << point.y << "," << point.z << ",";
                    }
                    lidar_file << center.x << "," << center.y << "," << center.z << "\n";
                } else {
                    std::cerr << "Error: Unable to open lidar.txt for writing.\n";
                }
            }

            //// 2.1计算外参矩阵
            // ceres-solver通过pnp求解外参
            int num = 1;
            int interval = 1;
            std::set <size_t> selected_indices; // 使用 set 去重并确保索引唯一
            if (i == 0){selected_indices.insert(0);}
            if (i > 0 && i <= 1800 && i % interval == 0) {
                size_t start_index = 0; // 起始索引
                size_t end_index = i;             // 结束索引
//                if (i > 100){start_index = i - 100;}
                for (int t_ = 0; t_ < num; ++t_) {
                    float t = float(t_) / (num);  // [0, 1]的线性分布
                    size_t index = static_cast<size_t>(start_index + t * (end_index - start_index) + 0.5f);  // 平均
                    //size_t index = static_cast<size_t>(i * std::pow(t, 2)); // 平方，pow(t, 1.5)是t的1.5次方
                    if (index > 0) selected_indices.insert(index);
                }
                selected_indices.insert(i);
                std::cout << "Selected indices: ";
                for (size_t index : selected_indices) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
            }

            // 遍历所有图像和点云的角点，每组添加 4 个点
            for (size_t j : selected_indices){
                for (size_t k = 0; k < 4; ++k){
                    // 添加图像中的 2D 点
                    imgs_corners_selected.push_back(imgs_corners_temp[j][k]);
                    // 添加点云中的 3D 点
                    const pcl::PointXYZ& point = pcs_points_temp[j]->points[k];
                    pcs_points_selected.emplace_back(point.x, point.y, point.z);
                }
            }
            std::cout << "像素点数量: " << imgs_corners_selected.size() << std::endl;
            std::cout << "点云点数量: " << pcs_points_selected.size() << std::endl;

            // 计算外参参数t，q
            {
                ScopedTimer timer(extrinsic_stats);
                calcul_r_t(imgs_corners_selected,pcs_points_selected);
            }
            r_t_file << t[0] << ", " << t[1] << ", " << t[2]<< ", "
                     << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";

            //// 2.2 计算这些点的位姿
            img_corners_pose=imgs_corners_3d_temp[i];
            for (const auto& pt : pcs_points_temp[i]->points) {
                pc_points_pose.push_back(cv::Point3f(pt.x, pt.y, pt.z));
            }

            {
                ScopedTimer timer(pose_stats);
                calculate_pose(img_corners_pose, pc_points_pose, center, normal_vector);
            }
            pose_file << center.x << ", " << center.y << ", " << center.z << ", "
                      << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << "\n";

            imgs_corners_selected.clear();
            pcs_points_selected.clear();
            img_corners_pose.clear();
            pc_points_pose.clear();
        }
        const auto mode9_end = std::chrono::steady_clock::now();
        mode9_total_stats.add(std::chrono::duration<double, std::milli>(mode9_end - mode9_start).count());

        std::cout << "\n========== Runtime Summary (estimation_mode 9) ==========" << std::endl;
        print_runtime_stats("Data preparation total", data_prepare_stats);
        print_runtime_stats("Data loading", data_load_stats);
        print_runtime_stats("Image corner extraction", image_extract_stats);
        print_runtime_stats("Point cloud corner extraction", pointcloud_extract_stats);
        print_runtime_stats("Mode 9 total", mode9_total_stats);
        print_runtime_stats("2D-3D extrinsic estimation", extrinsic_stats);
        print_runtime_stats("Pose estimation", pose_stats);
        std::cout << "=========================================================" << std::endl;
    }

    /////////////////////////////////////////////////////////////

    r_t_file.close();
    pose_file.close();
    pixel_file.close();
    cam_file.close();
    lidar_file.close();

    return 0;
}
