#ifndef WS_POSE_ESTIMATION_CALCULATE_R_T_H
#define WS_POSE_ESTIMATION_CALCULATE_R_T_H

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
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
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/impl/project_inliers.hpp>
#include <pcl/common/intersections.h>
#include <iostream>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include <math.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

Eigen::Matrix3d inner;
Eigen::Matrix<double,1,5> distortion;
cv::Mat camera_instrinsics_ = cv::Mat(3, 3, CV_64F);
cv::Mat distortion_coefficients_ = cv::Mat(1, 5, CV_64F);
cv::Mat rvec_c_l(3, 1, cv::DataType<double>::type);
cv::Mat tvec_c_l(3, 1, cv::DataType<double>::type);
double t[3] = {0., 2., 3.};
double q[4] = {1., 0., .0, 0.};

struct calibrate_cam_lidar_data {
    Eigen::Vector3d lidar_point;
    Eigen::Vector2d pixel_point;
};
std::vector<calibrate_cam_lidar_data> calibrate_data_input;

// 模板函数
template <class T>
void point_reproject(const T* q, const T* t, const Eigen::Matrix<T, 3, 1> point_at_world, T reproject_point[2]){
    Eigen::Quaternion<T> q_last(q[0], q[1], q[2], q[3]);
    q_last.normalize();
    Eigen::Matrix<T, 3, 1> t_last(t[0], t[1], t[2]);

    Eigen::Matrix<T, 3, 1> point_at_camera;
    point_at_camera = q_last * point_at_world + t_last;

    Eigen::Matrix<T, 2, 1> pixel_at_image;

    double intrinsic[4] = {inner(0,0), inner(0,2), inner(1,1), inner(1,2)};

    pixel_at_image[0] =
            intrinsic[0] * point_at_camera[0] / point_at_camera[2] +
            intrinsic[1];
    pixel_at_image[1] =
            intrinsic[2] * point_at_camera[1] / point_at_camera[2] +
            intrinsic[3];

    T u = (pixel_at_image[0] - intrinsic[1]) /
          intrinsic[0];
    T v = (pixel_at_image[1] - intrinsic[3]) /
          intrinsic[2];

    T r_2 = u * u + v * v;

    double k1 = distortion[0];
    double k2 = distortion[1];
    double p1 = distortion[2];
    double p2 = distortion[3];
    double k3 = distortion[4];

    v = v * (1. + k1 * r_2 + k2 * r_2 * r_2 + k3*r_2 * r_2*r_2 ) +  p1 * (r_2 + 2. * pow(v,2)) + 2. * p2 * u * v;
    u = u * (1. + k1 * r_2 + k2 * r_2 * r_2 + k3*r_2 * r_2*r_2) +  2. * p1 * u * v + p2 * (r_2 + 2. * pow(u,2)) ;


    reproject_point[0] =
            u * intrinsic[0] + intrinsic[1];
    reproject_point[1] =
            v * intrinsic[2] + intrinsic[3];
}
template <typename _T>
struct pixel_cost_function {
    // obtained from the point reproject to image
    Eigen::Matrix<_T, 3, 1> point_from_world_;
    // obtained directly from image
    Eigen::Matrix<_T, 2, 1> pixel_from_image_;

    // construct function
    pixel_cost_function(const Eigen::Matrix<_T, 3, 1>& point_from_world,const Eigen::Matrix<_T, 2, 1>& pixel_from_image)
            :point_from_world_(point_from_world),pixel_from_image_(pixel_from_image)
    {}

    // operator() function
    template <typename T>
    bool operator()(const T* t, const T* q, T* residual) const {
        Eigen::Matrix<T, 3, 1> point_at_world(static_cast<T>(point_from_world_(0)),
                                              static_cast<T>(point_from_world_(1)),
                                              static_cast<T>(point_from_world_(2)));
        T reproject_point[2];

        point_reproject( q,  t, point_at_world, reproject_point);

        Eigen::Matrix<T, 2, 1> residual_matrix(pixel_from_image_[0] - reproject_point[0],
                                               pixel_from_image_[1] - reproject_point[1]);

        residual[0] = residual_matrix[0];
        residual[1] = residual_matrix[1];

        return true;
    }

    // param[in] weight:
    static ceres::CostFunction* Create(
            const Eigen::Matrix<_T, 3, 1> point_from_world,
            const Eigen::Matrix<_T, 2, 1> point_from_pixel
    ) {
        return (
                new ceres::AutoDiffCostFunction<pixel_cost_function, 2,3,4>(new pixel_cost_function(
                        point_from_world, point_from_pixel)));
    }
};

// 函数声明
void param_init(YAML::Node config_node);
void CalibrateCamera(const std::vector<calibrate_cam_lidar_data>& calibrate_data, const int& start_index, const int& total_num);
void calcul_r_t(std::vector<cv::Point2f> img_corners,std::vector<cv::Point3f> board_corner_ps);

// 函数实现
void param_init(YAML::Node config_node)
{
    // 获取 camera_intrinsic 节点
    YAML::Node camera_intrinsic = config_node["camera_intrinsic"];

    vector<double> inner_list;
    vector<double> dist_list;
    // 提取 inner 参数
    if (camera_intrinsic["inner"] && camera_intrinsic["inner"].IsSequence()) {
        for (const auto& value : camera_intrinsic["inner"]) {
            inner_list.push_back(value.as<double>());
        }
    }
    // 提取 dist 参数
    if (camera_intrinsic["dist"] && camera_intrinsic["dist"].IsSequence()) {
        for (const auto& value : camera_intrinsic["dist"]) {
            dist_list.push_back(value.as<double>());
        }
    }
    //std::cout << "inner_list size: " << inner_list.size() << std::endl;
    //std::cout << "dist_list size: " << dist_list.size() << std::endl;

    Eigen::Matrix3d inner_transpose;
    inner_transpose = Eigen::Map<Eigen::MatrixXd>(inner_list.data(), 3, 3);
    inner = inner_transpose.transpose();
    distortion = Eigen::Map<Eigen::Matrix<double,1,5>>(dist_list.data(), 1, 5);

    cv::eigen2cv(inner,camera_instrinsics_);
    cv::eigen2cv(distortion,distortion_coefficients_);
}
void CalibrateCamera(const std::vector<calibrate_cam_lidar_data>& calibrate_data, const int& start_index, const int& total_num) {

    t[0] = tvec_c_l.at<double>(0);
    t[1] = tvec_c_l.at<double>(1);
    t[2] = tvec_c_l.at<double>(2);

    cv::Mat cv_rotationMatrix(3, 3, CV_64F);
    Eigen::Matrix3d mat3x3;

    Rodrigues (rvec_c_l, cv_rotationMatrix);//rootation_vector -> cv_mat
    cv2eigen(cv_rotationMatrix, mat3x3);//cv_mat -> eigen_mat
    Eigen::Quaterniond quaternion(mat3x3);//eigen_mat -> quat
    quaternion.normalize();
    q[0] = quaternion.w();
    q[1] = quaternion.x();
    q[2] = quaternion.y();
    q[3] = quaternion.z();

    ceres::LocalParameterization* q_parameterization =
            new ceres::EigenQuaternionParameterization();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    for (int i = start_index; i < total_num + start_index; ++i) {

        problem.AddResidualBlock(
                pixel_cost_function<double>::Create(calibrate_data[i].lidar_point.cast<double>(),
                                                    calibrate_data[i].pixel_point.cast<double>()
                ),
                new ceres::HuberLoss(0.5), t, q);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.function_tolerance = 1e-18;
    options.parameter_tolerance = 1e-18;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    Eigen::Quaterniond q_final(q[0], q[1], q[2], q[3]);
    Eigen::Vector3d euler_angle = q_final.toRotationMatrix().eulerAngles(2, 1, 0);
    Eigen::Matrix3d rotation = q_final.toRotationMatrix();

    cout<<"打印transform: "<<  t[0] <<" "<< t[1] <<" "<< t[2] <<endl;
    cout<<"打印下四元数 "<< q[0] <<" "<< q[1] <<" "<< q[2] <<" "<< q[3] <<endl;

    vector<double> vec_t  = {t[0],t[1],t[2]};
    cv::Mat Trans(vec_t);

    cout<<"euler_angle:  "<<endl;
    cout<<  euler_angle <<endl;
    vector<double> vec_eul  = {euler_angle[0],euler_angle[1],euler_angle[2]};
    cv::Mat euler(vec_eul);

    std::vector<double> mat_r= {
            rotation(0, 0), rotation(0, 1), rotation(0, 2),
            rotation(1, 0), rotation(1, 1), rotation(1, 2),
            rotation(2, 0), rotation(2, 1), rotation(2, 2)
    };
}
void calcul_r_t( std::vector<cv::Point2f> img_corners,std::vector<cv::Point3f> board_corner_ps)
{
    // 1st step: rough calculation of extrinsic using pnp
    std::vector<cv::Point3f> board_corner_ps_tough_cali;
    std::vector<cv::Point2f> img_corners_tough_cali;
    for(int i = 0;i<4;i++)
    {
        board_corner_ps_tough_cali.push_back(board_corner_ps[i]);
        img_corners_tough_cali.push_back(img_corners[i]);
    }

    // 因为每帧是4对点，所以这里用4对3D-2D点结果才会合理，选择PnP算法为cv::SOLVEPNP_EPNP
    cv::solvePnP(board_corner_ps_tough_cali, img_corners_tough_cali, camera_instrinsics_, distortion_coefficients_,
                 rvec_c_l, tvec_c_l,false, cv::SOLVEPNP_EPNP);
    cout<<"rvec_c_l:"<<endl;
    cout<<rvec_c_l<<endl;
    cout<<"tvec_c_l:"<<endl;
    cout<<tvec_c_l<<endl;
    // 2nd step: send all point-pairs and rough extrinsic to ceres

    for(int i = 0; i<img_corners.size(); i++)
    {
        calibrate_cam_lidar_data calib_tmp;
        cv::Point3f tmp_lidar = board_corner_ps[i];
        cv::Point2f tmp_img   = img_corners[i];

        calib_tmp.lidar_point.x() = tmp_lidar.x;
        calib_tmp.lidar_point.y() = tmp_lidar.y;
        calib_tmp.lidar_point.z() = tmp_lidar.z;

        calib_tmp.pixel_point.x() = tmp_img.x;
        calib_tmp.pixel_point.y() = tmp_img.y;

        calibrate_data_input.push_back(calib_tmp);
    }

    int start_index = 0;
    int total_num = calibrate_data_input.size();

    CalibrateCamera(calibrate_data_input,start_index,total_num);

}

#endif //WS_POSE_ESTIMATION_CALCULATE_R_T_H
