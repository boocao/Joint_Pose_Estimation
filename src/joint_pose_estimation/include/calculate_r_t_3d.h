#ifndef WS_POSE_ESTIMATION_CALCULATE_R_T_3D_H
#define WS_POSE_ESTIMATION_CALCULATE_R_T_3D_H

//// 用umeyama或horn等方法计算外参。或可作为初始值
void initialize_with_umeyama(const std::vector<cv::Point3f>& src_points, const std::vector<cv::Point3f>& dst_points,
                             double t_out[3], double q_out[4]);
void initialize_with_umeyama(const std::vector<cv::Point3f>& src_points, const std::vector<cv::Point3f>& dst_points,
                             double t_out[3], double q_out[4]) {

    if (src_points.size() != dst_points.size() || src_points.size() < 3) {
        std::cerr << "Error: Point sets must have the same size and at least 3 points" << std::endl;
        return;
    }

    int n = src_points.size();

    // 转换为Eigen矩阵
    Eigen::Matrix3Xd P(3, n);  // source points (点云坐标系)
    Eigen::Matrix3Xd Q(3, n);  // destination points (图像坐标系)

    for (int i = 0; i < n; ++i) {
        P.col(i) << src_points[i].x, src_points[i].y, src_points[i].z;
        Q.col(i) << dst_points[i].x, dst_points[i].y, dst_points[i].z;
    }

    // 计算质心
    Eigen::Vector3d centroid_P = P.rowwise().mean();
    Eigen::Vector3d centroid_Q = Q.rowwise().mean();

    // 中心化点集
    Eigen::Matrix3Xd P_centered = P.colwise() - centroid_P;
    Eigen::Matrix3Xd Q_centered = Q.colwise() - centroid_Q;

    // 计算协方差矩阵 H = P_centered * Q_centered^T
    Eigen::Matrix3d H = P_centered * Q_centered.transpose();

    // SVD分解: H = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 计算旋转矩阵 R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();

    // 确保旋转矩阵的行列式为正（右手坐标系）
    if (R.determinant() < 0) {
        V.col(2) *= -1;  // 翻转V的最后一列
        R = V * U.transpose();
    }

    // 计算平移向量: t = centroid_Q - R * centroid_P
    Eigen::Vector3d t = centroid_Q - R * centroid_P;

    // 转换旋转矩阵为四元数
    Eigen::Quaterniond q(R);
    q.normalize();

    // 输出到数组格式 - [w, x, y, z] 顺序
    t_out[0] = t.x();
    t_out[1] = t.y();
    t_out[2] = t.z();

    q_out[0] = q.w();  // w
    q_out[1] = q.x();  // x
    q_out[2] = q.y();  // y
    q_out[3] = q.z();  // z

    std::cout << "Umeyama算法结果:" << std::endl;
    std::cout << "Translation: (" << t_out[0] << ", " << t_out[1] << ", " << t_out[2] << ")" << std::endl;
    std::cout << "Quaternion (wxyz): (" << q_out[0] << ", " << q_out[1] << ", " << q_out[2] << ", " << q_out[3] << ")" << std::endl;

    // 计算拟合误差
    double total_error = 0.0;
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p_transformed = R * P.col(i) + t;
        Eigen::Vector3d error = p_transformed - Q.col(i);
        double error_norm = error.norm();
        total_error += error_norm;
        max_error = std::max(max_error, error_norm);
    }

    std::cout << "使用点对数量: " << n << std::endl;
    std::cout << "平均拟合误差: " << total_error / n << std::endl;
    std::cout << "最大拟合误差: " << max_error << std::endl;
    std::cout << "RMS误差: " << std::sqrt(total_error * total_error / n) << std::endl;
}
void initialize_with_horn(const std::vector<cv::Point3f>& src_points, const std::vector<cv::Point3f>& dst_points,
                          double t_out[3], double q_out[4]);
void initialize_with_horn(const std::vector<cv::Point3f>& src_points, const std::vector<cv::Point3f>& dst_points,
                          double t_out[3], double q_out[4]) {
    // 检查输入点的数量是否一致且非空
    if (src_points.size() != dst_points.size() || src_points.empty()) {
        std::cerr << "Error: The number of points in the two sets must be equal and non-zero." << std::endl;
        return;
    }

    size_t num_points = src_points.size();

    // 计算质心
    cv::Point3f centroid_src = {0, 0, 0};
    cv::Point3f centroid_dst = {0, 0, 0};
    for (size_t i = 0; i < num_points; ++i) {
        centroid_src += src_points[i];
        centroid_dst += dst_points[i];
    }
    centroid_src /= static_cast<float>(num_points);
    centroid_dst /= static_cast<float>(num_points);

    // 中心化点
    std::vector<cv::Point3f> src_centered = src_points;
    std::vector<cv::Point3f> dst_centered = dst_points;
    for (size_t i = 0; i < num_points; ++i) {
        src_centered[i] -= centroid_src;
        dst_centered[i] -= centroid_dst;
    }

    // 构建协方差矩阵 H
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < num_points; ++i) {
        Eigen::Vector3d src_vec(src_centered[i].x, src_centered[i].y, src_centered[i].z);
        Eigen::Vector3d dst_vec(dst_centered[i].x, dst_centered[i].y, dst_centered[i].z);
        H += src_vec * dst_vec.transpose();
    }

    // 构造对称矩阵 M
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M(0, 0) = H(0, 0) + H(1, 1) + H(2, 2);
    M(0, 1) = H(1, 2) - H(2, 1);
    M(0, 2) = H(2, 0) - H(0, 2);
    M(0, 3) = H(0, 1) - H(1, 0);

    M(1, 0) = M(0, 1);
    M(1, 1) = H(0, 0) - H(1, 1) - H(2, 2);
    M(1, 2) = H(0, 1) + H(1, 0);
    M(1, 3) = H(2, 0) + H(0, 2);

    M(2, 0) = M(0, 2);
    M(2, 1) = M(1, 2);
    M(2, 2) = -H(0, 0) + H(1, 1) - H(2, 2);
    M(2, 3) = H(1, 2) + H(2, 1);

    M(3, 0) = M(0, 3);
    M(3, 1) = M(1, 3);
    M(3, 2) = M(2, 3);
    M(3, 3) = -H(0, 0) - H(1, 1) + H(2, 2);

    // 求解最大特征值对应的特征向量（四元数）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(M);
    Eigen::VectorXd max_eigenvector = eigen_solver.eigenvectors().col(3); // 最大特征值对应的特征向量

    // 确保四元数的实部为正（消除符号歧义）
    if (max_eigenvector(0) < 0) {
        max_eigenvector = -max_eigenvector;
    }

    // 提取四元数 - 注意：Horn方法的特征向量顺序是 [w, x, y, z]
    Eigen::Quaterniond q(max_eigenvector(0), max_eigenvector(1), max_eigenvector(2), max_eigenvector(3));
    q.normalize(); // 归一化四元数

    // 将四元数转换为旋转矩阵
    Eigen::Matrix3d R = q.toRotationMatrix();

    // 计算平移向量 t
    Eigen::Vector3d centroid_src_eigen(centroid_src.x, centroid_src.y, centroid_src.z);
    Eigen::Vector3d centroid_dst_eigen(centroid_dst.x, centroid_dst.y, centroid_dst.z);
    Eigen::Vector3d t = centroid_dst_eigen - R * centroid_src_eigen;

    // 将结果写入输出数组 - 统一使用 [w, x, y, z] 顺序
    t_out[0] = t(0);
    t_out[1] = t(1);
    t_out[2] = t(2);

    q_out[0] = q.w();  // w
    q_out[1] = q.x();  // x
    q_out[2] = q.y();  // y
    q_out[3] = q.z();  // z
}

// 定义误差项
template <typename T>
struct PointAlignmentError {
    PointAlignmentError(const Eigen::Matrix<T, 3, 1>& point_src, const Eigen::Matrix<T, 3, 1>& point_dst)
            : point_src_(point_src), point_dst_(point_dst) {}

    template <typename U>
    bool operator()(const U* const t, const U* const q, U* residuals) const {
        // 将四元数转换为旋转矩阵，四元数顺序: [w, x, y, z] - 与Ceres QuaternionParameterization一致
        Eigen::Quaternion<U> quat(q[0], q[1], q[2], q[3]); // w, x, y, z
        Eigen::Matrix<U, 3, 3> R = quat.toRotationMatrix();

        // 计算变换后的点: dst = R * src + t
        Eigen::Matrix<U, 3, 1> transformed_point =
                R * point_src_.template cast<U>() + Eigen::Matrix<U, 3, 1>(t[0], t[1], t[2]);

        // 计算残差:
        residuals[0] = point_dst_[0]-transformed_point[0];
        residuals[1] = point_dst_[1]-transformed_point[1];
        residuals[2] = point_dst_[2]-transformed_point[2];

        return true;
    }

private:
    const Eigen::Matrix<T, 3, 1> point_src_; // 点云坐标系中的点
    const Eigen::Matrix<T, 3, 1> point_dst_; // 相机坐标系中的点
};
// 改进的共面约束
template <typename T>
struct CoplanarConstraint {
    CoplanarConstraint(const std::vector<Eigen::Matrix<T, 3, 1>>& points_src,
                       const std::vector<Eigen::Matrix<T, 3, 1>>& points_dst) {
        for (int i = 0; i < 4; ++i) {
            points_src_[i] = points_src[i];
            points_dst_[i] = points_dst[i];
        }
    }

    template <typename U>
    bool operator()(const U* const t, const U* const q, U* residuals) const {
        Eigen::Quaternion<U> quat(q[0], q[1], q[2], q[3]);  // w, x, y, z
        Eigen::Matrix<U, 3, 3> R = quat.toRotationMatrix();

        Eigen::Matrix<U, 3, 1> transformed_points[4];
        for (int i = 0; i < 4; ++i) {
            transformed_points[i] = R * points_src_[i].template cast<U>() +
                                    Eigen::Matrix<U, 3, 1>(t[0], t[1], t[2]);
        }

        Eigen::Matrix<U, 3, 1> v1 = transformed_points[1] - transformed_points[0];
        Eigen::Matrix<U, 3, 1> v2 = transformed_points[2] - transformed_points[0];
        Eigen::Matrix<U, 3, 1> normal = v1.cross(v2);
        U normal_norm = normal.norm();

        if (normal_norm < U(1e-8)) {
            v1 = transformed_points[2] - transformed_points[0];
            v2 = transformed_points[3] - transformed_points[0];
            normal = v1.cross(v2);
            normal_norm = normal.norm();
        }

        if (normal_norm > U(1e-8)) {
            normal /= normal_norm;
        } else {
            normal = Eigen::Matrix<U, 3, 1>(U(0), U(0), U(1));
        }

        U d = normal.dot(transformed_points[0]);

        residuals[0] = normal.dot(transformed_points[3]) - d;

        for (int i = 0; i < 4; ++i) {
            Eigen::Matrix<U, 3, 1> img_point = points_dst_[i].template cast<U>();
            residuals[1 + i] = normal.dot(img_point) - d;
        }

        return true;
    }

private:
    Eigen::Matrix<T, 3, 1> points_src_[4];
    Eigen::Matrix<T, 3, 1> points_dst_[4];
};
// 改进的矩形约束
template <typename T>
struct RectangleConstraint {
    RectangleConstraint(const std::vector<Eigen::Matrix<T, 3, 1>>& points_src,
                        const std::vector<Eigen::Matrix<T, 3, 1>>& points_dst) {
        for (int i = 0; i < 4; ++i) {
            points_src_[i] = points_src[i];
            points_dst_[i] = points_dst[i];
        }

        original_src_distances_[0] = (points_src[1] - points_src[0]).norm();
        original_src_distances_[1] = (points_src[2] - points_src[1]).norm();
        original_src_distances_[2] = (points_src[3] - points_src[2]).norm();
        original_src_distances_[3] = (points_src[0] - points_src[3]).norm();

        original_dst_distances_[0] = (points_dst[1] - points_dst[0]).norm();
        original_dst_distances_[1] = (points_dst[2] - points_dst[1]).norm();
        original_dst_distances_[2] = (points_dst[3] - points_dst[2]).norm();
        original_dst_distances_[3] = (points_dst[0] - points_dst[3]).norm();
    }

    template <typename U>
    bool operator()(const U* const t, const U* const q, U* residuals) const {
        Eigen::Quaternion<U> quat(q[0], q[1], q[2], q[3]);  // w, x, y, z
        Eigen::Matrix<U, 3, 3> R = quat.toRotationMatrix();

        Eigen::Matrix<U, 3, 1> transformed[4];
        for (int i = 0; i < 4; ++i) {
            transformed[i] = R * points_src_[i].template cast<U>() +
                             Eigen::Matrix<U, 3, 1>(t[0], t[1], t[2]);
        }

        residuals[0] = (transformed[1] - transformed[0]).norm() - U(original_src_distances_[0]);
        residuals[1] = (transformed[2] - transformed[1]).norm() - U(original_src_distances_[1]);
        residuals[2] = (transformed[3] - transformed[2]).norm() - U(original_src_distances_[2]);
        residuals[3] = (transformed[0] - transformed[3]).norm() - U(original_src_distances_[3]);

        Eigen::Matrix<U, 3, 1> img_points[4];
        for (int i = 0; i < 4; ++i)
            img_points[i] = points_dst_[i].template cast<U>();

        U img_dists[4] = {
                (img_points[1] - img_points[0]).norm(),
                (img_points[2] - img_points[1]).norm(),
                (img_points[3] - img_points[2]).norm(),
                (img_points[0] - img_points[3]).norm()
        };

        residuals[4] = (transformed[1] - transformed[0]).norm() - img_dists[0];
        residuals[5] = (transformed[2] - transformed[1]).norm() - img_dists[1];
        residuals[6] = (transformed[3] - transformed[2]).norm() - img_dists[2];
        residuals[7] = (transformed[0] - transformed[3]).norm() - img_dists[3];

        return true;
    }

private:
    Eigen::Matrix<T, 3, 1> points_src_[4];
    Eigen::Matrix<T, 3, 1> points_dst_[4];
    T original_src_distances_[4];
    T original_dst_distances_[4];
};

void calcul_r_t_3d(const std::vector<cv::Point3f>& pcs_points_selected, const std::vector<cv::Point3f>& imgs_corners_3d_selected,
                   double t_out[3], double q_out[4]);
void calcul_r_t_3d(const std::vector<cv::Point3f>& pcs_points_selected, const std::vector<cv::Point3f>& imgs_corners_3d_selected,
                   double t_out[3], double q_out[4]) {
    // 检查输入点的数量是否一致且非空
    if (pcs_points_selected.size() != imgs_corners_3d_selected.size() || imgs_corners_3d_selected.empty()) {
        std::cerr << "Error: The number of points in the two sets must be equal and non-zero." << std::endl;
        return;
    }

    // 构建优化问题
    ceres::Problem problem;
    // 加入鲁棒核函数 HuberLoss（新增）
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    for (size_t i = 0; i < imgs_corners_3d_selected.size(); ++i) {
        Eigen::Vector3d point_src(pcs_points_selected[i].x, pcs_points_selected[i].y, pcs_points_selected[i].z);
        Eigen::Vector3d point_dst(imgs_corners_3d_selected[i].x, imgs_corners_3d_selected[i].y, imgs_corners_3d_selected[i].z);

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PointAlignmentError<double>, 3, 3, 4>(
                        new PointAlignmentError<double>(point_src, point_dst)),
                loss_function, t_out, q_out
        );
    }

    // 单位四元数约束 - Ceres的QuaternionParameterization期望 [w, x, y, z] 顺序
    ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
    problem.SetParameterization(q_out, quaternion_parameterization);

    // 配置求解器
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.minimizer_progress_to_stdout = false;  // 打印每轮优化过程
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "迭代次数: " << summary.num_successful_steps + summary.num_unsuccessful_steps << std::endl;
    std::cout << "最终代价: " << summary.final_cost << std::endl;

    // 确保输出的四元数是单位四元数
    Eigen::Quaterniond q_final(q_out[0], q_out[1], q_out[2], q_out[3]);
    q_final.normalize();
    q_out[0] = q_final.w();
    q_out[1] = q_final.x();
    q_out[2] = q_final.y();
    q_out[3] = q_final.z();
}

void calcul_r_t_3d_with_constraints(const std::vector<cv::Point3f>& pcs_points_selected,
                                    const std::vector<cv::Point3f>& imgs_corners_3d_selected,
                                    double t_out[3], double q_out[4],
                                    bool use_coplanar_constraint,
                                    bool use_rectangle_constraint );
void calcul_r_t_3d_with_constraints(const std::vector<cv::Point3f>& pcs_points_selected,
                                    const std::vector<cv::Point3f>& imgs_corners_3d_selected,
                                    double t_out[3], double q_out[4],
                                    bool use_coplanar_constraint = true,
                                    bool use_rectangle_constraint = true,
                                    double coplanar_weight = 1.0,
                                    double rectangle_weight = 1.0) {
    // 检查输入点的数量是否一致且非空
    if (pcs_points_selected.size() != imgs_corners_3d_selected.size() || imgs_corners_3d_selected.empty()) {
        std::cerr << "Error: The number of points in the two sets must be equal and non-zero." << std::endl;
        return;
    }

    // 只有在使用约束时才检查是否为4的倍数
    if ((use_coplanar_constraint || use_rectangle_constraint) && imgs_corners_3d_selected.size() % 4 != 0) {
        std::cerr << "Error: The number of points must be a multiple of 4 for rectangle/coplanar constraints." << std::endl;
        return;
    }

    // 初始化输出参数（如果尚未初始化）
    // 建议在调用此函数前先初始化这些参数

    // 构建优化问题
    ceres::Problem problem;

    // 添加点对齐的残差块
    for (size_t i = 0; i < imgs_corners_3d_selected.size(); ++i) {
        Eigen::Vector3d pt_src(pcs_points_selected[i].x, pcs_points_selected[i].y, pcs_points_selected[i].z);
        Eigen::Vector3d pt_dst(imgs_corners_3d_selected[i].x, imgs_corners_3d_selected[i].y, imgs_corners_3d_selected[i].z);

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PointAlignmentError<double>, 3, 3, 4>(
                        new PointAlignmentError<double>(pt_src, pt_dst)),
                nullptr, t_out, q_out
        );
    }

    // 添加约束（只有在点数为4的倍数且启用约束时）
    if ((use_coplanar_constraint || use_rectangle_constraint) && imgs_corners_3d_selected.size() % 4 == 0) {
        int num_frames = imgs_corners_3d_selected.size() / 4;

        for (int frame = 0; frame < num_frames; ++frame) {
            int idx = frame * 4;
            std::vector<Eigen::Vector3d> pcs, imgs;

            for (int j = 0; j < 4; ++j) {
                pcs.emplace_back(pcs_points_selected[idx + j].x,
                                 pcs_points_selected[idx + j].y,
                                 pcs_points_selected[idx + j].z);
                imgs.emplace_back(imgs_corners_3d_selected[idx + j].x,
                                  imgs_corners_3d_selected[idx + j].y,
                                  imgs_corners_3d_selected[idx + j].z);
            }

            if (use_coplanar_constraint) {
                ceres::LossFunction* coplanar_loss = nullptr;
                if (coplanar_weight != 1.0) {
                    coplanar_loss = new ceres::ScaledLoss(nullptr, coplanar_weight, ceres::TAKE_OWNERSHIP);
                }

                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<CoplanarConstraint<double>, 5, 3, 4>(
                                new CoplanarConstraint<double>(pcs, imgs)),
                        coplanar_loss,
                        t_out, q_out
                );
            }

            if (use_rectangle_constraint) {
                ceres::LossFunction* rectangle_loss = nullptr;
                if (rectangle_weight != 1.0) {
                    rectangle_loss = new ceres::ScaledLoss(nullptr, rectangle_weight, ceres::TAKE_OWNERSHIP);
                }

                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<RectangleConstraint<double>, 8, 3, 4>(
                                new RectangleConstraint<double>(pcs, imgs)),
                        rectangle_loss,
                        t_out, q_out
                );
            }
        }
    }

    // 设置四元数参数化（注意：在新版本Ceres中可能需要使用Manifold）
    problem.SetParameterization(q_out, new ceres::QuaternionParameterization());

    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false; // 建议设为false避免过多输出
    options.function_tolerance = 1e-6;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << "Ceres optimization result: " << summary.BriefReport() << std::endl;
    if (!summary.IsSolutionUsable()) {
        std::cerr << "Warning: Optimization did not converge to a usable solution!" << std::endl;
        std::cerr << "Full report: " << summary.FullReport() << std::endl;
    }

    // 规范化四元数
    Eigen::Quaterniond q_final(q_out[0], q_out[1], q_out[2], q_out[3]);
    q_final.normalize();
    q_out[0] = q_final.w();
    q_out[1] = q_final.x();
    q_out[2] = q_final.y();
    q_out[3] = q_final.z();

    // 输出最终结果
    std::cout << "Final translation: [" << t_out[0] << ", " << t_out[1] << ", " << t_out[2] << "]" << std::endl;
    std::cout << "Final quaternion (w,x,y,z): [" << q_out[0] << ", " << q_out[1] << ", " << q_out[2] << ", " << q_out[3] << "]" << std::endl;
}

void calculate_pose(const std::vector<cv::Point3f>& imgs_corners_3d, const std::vector<cv::Point3f>& pc_points,
                    cv::Point3f& center, cv::Point3f& normal_vector);
void calculate_pose(const std::vector<cv::Point3f>& img_corners_3d, const std::vector<cv::Point3f>& pc_points,
                    cv::Point3f& center, cv::Point3f& normal_vector){
    // 转换为 Eigen 格式
    Eigen::Vector3d translation(t[0], t[1], t[2]);
    Eigen::Quaterniond quaternion(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d rotation = quaternion.toRotationMatrix();

    std::vector<cv::Point3f> all_points;

    // 添加相机角点
    std::cout << "imgs_subset_pose:" << std::endl;
    for (const auto& point : img_corners_3d) {
        all_points.emplace_back(point.x, point.y, point.z);
        std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
    }

    // 雷达点
    std::cout << "pcs_subset_pose:" << std::endl;
    for (const auto& point : pc_points) {
        std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
    }

    // 雷达点转换到相机坐标系
    std::cout << "pcs_subset_pose_transformed:" << std::endl;
    for (const auto& pt : pc_points) {
        Eigen::Vector3d point_lidar(pt.x, pt.y, pt.z);
        Eigen::Vector3d point_cam = rotation * point_lidar + translation;
        all_points.emplace_back(point_cam.x(), point_cam.y(), point_cam.z());
        std::cout << "(" << point_cam.x() << ", " << point_cam.y() << ", " << point_cam.z() << ")" << std::endl;
    }

    // 计算几何中心
    center = cv::Point3f(0, 0, 0);
    for (const auto& p : all_points) {
        center.x += p.x;
        center.y += p.y;
        center.z += p.z;
    }
    center.x /= static_cast<float>(all_points.size());
    center.y /= static_cast<float>(all_points.size());
    center.z /= static_cast<float>(all_points.size());

    // PCA 提取方向
    cv::Mat data(static_cast<int>(all_points.size()), 3, CV_64F);
    for (size_t i = 0; i < all_points.size(); ++i) {
        data.at<double>(i, 0) = all_points[i].x;
        data.at<double>(i, 1) = all_points[i].y;
        data.at<double>(i, 2) = all_points[i].z;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    normal_vector = cv::Point3f(
            static_cast<float>(pca.eigenvectors.at<double>(2, 0)),
            static_cast<float>(pca.eigenvectors.at<double>(2, 1)),
            static_cast<float>(pca.eigenvectors.at<double>(2, 2))
    );

    // 归一化
    float norm = std::sqrt(
            normal_vector.x * normal_vector.x +
            normal_vector.y * normal_vector.y +
            normal_vector.z * normal_vector.z
    );
    normal_vector.x /= norm;
    normal_vector.y /= norm;
    normal_vector.z /= norm;

    // 对齐 X+ 方向
    cv::Point3f ref_vector(1, 0, 0);
    if (normal_vector.x * ref_vector.x +
        normal_vector.y * ref_vector.y +
        normal_vector.z * ref_vector.z < 0) {
        normal_vector.x *= -1;
        normal_vector.y *= -1;
        normal_vector.z *= -1;
    }
    // 打印位姿
    std::cout << "Vehicle Center: (" << center.x << ", " << center.y << ", " << center.z << ")\n";
    std::cout << "Normal Vector: (" << normal_vector.x << ", " << normal_vector.y << ", " << normal_vector.z << ")\n";
}

// 使用平移向量和四元数对点进行变换
std::vector<cv::Point3f> transform_points(const std::vector<cv::Point3f>& points, const double t[3], const double q[4]);
std::vector<cv::Point3f> transform_points(const std::vector<cv::Point3f>& points, const double t[3], const double q[4]) {
    std::vector<cv::Point3f> transformed_points;
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = quat.toRotationMatrix();

    for (const auto& pt : points) {
        Eigen::Vector3d pt_eigen(pt.x, pt.y, pt.z);
        Eigen::Vector3d transformed_pt = R * pt_eigen + Eigen::Vector3d(t[0], t[1], t[2]);

        transformed_points.emplace_back(transformed_pt[0], transformed_pt[1], transformed_pt[2]);
    }

    return transformed_points;
}
cv::Point3f transform_point(const cv::Point3f& pt, const double t[3], const double q[4]);
cv::Point3f transform_point(const cv::Point3f& pt, const double t[3], const double q[4]) {
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = quat.toRotationMatrix();

    Eigen::Vector3d pt_eigen(pt.x, pt.y, pt.z);
    Eigen::Vector3d transformed_pt = R * pt_eigen + Eigen::Vector3d(t[0], t[1], t[2]);

    return cv::Point3f(transformed_pt[0], transformed_pt[1], transformed_pt[2]);
}
cv::Point3f inverse_transform_point(const cv::Point3f& pt, const double t[3], const double q[4]);
cv::Point3f inverse_transform_point(const cv::Point3f& pt, const double t[3], const double q[4]) {
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = quat.toRotationMatrix();
    Eigen::Matrix3d R_inv = R.transpose();

    Eigen::Vector3d pt_eigen(pt.x, pt.y, pt.z);
    Eigen::Vector3d inverse_transformed_pt = R_inv * (pt_eigen - Eigen::Vector3d(t[0], t[1], t[2]));

    return cv::Point3f(inverse_transformed_pt[0], inverse_transformed_pt[1], inverse_transformed_pt[2]);
}

// 定义 3D 卡尔曼滤波器类（简单运动模型）
class KalmanFilter3D {
private:
    cv::KalmanFilter kf;
    cv::Mat state;       // [x, y, z, vx, vy, vz]
    cv::Mat measurement; // [x, y, z]
    bool initialized;    // 初始化状态

public:
    KalmanFilter3D()
            : kf(6, 3, 0),
              state(6, 1, CV_32F),
              measurement(3, 1, CV_32F),
              initialized(false)
    {
        // 状态转移矩阵
        kf.transitionMatrix = (cv::Mat_<float>(6, 6) <<
                                                     1,0,0,1,0,0,
                0,1,0,0,1,0,
                0,0,1,0,0,1,
                0,0,0,1,0,0,
                0,0,0,0,1,0,
                0,0,0,0,0,1);

        cv::setIdentity(kf.measurementMatrix); // H
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3)); // Q
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2)); // R
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1)); // P
    }

    // 初始化滤波器
    void init(const cv::Point3f& pt) {
        state.at<float>(0) = pt.x;
        state.at<float>(1) = pt.y;
        state.at<float>(2) = pt.z;
        state.at<float>(3) = 0;
        state.at<float>(4) = 0;
        state.at<float>(5) = 0;
        kf.statePost = state.clone();
        initialized = true;
    }

    // 更新滤波器状态
    cv::Point3f update(const cv::Point3f& pt) {
        if (!initialized) {
            init(pt);
            return pt;
        }
        kf.predict();
        measurement.at<float>(0) = pt.x;
        measurement.at<float>(1) = pt.y;
        measurement.at<float>(2) = pt.z;
        cv::Mat estimated = kf.correct(measurement);
        return cv::Point3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
    }

    // 判断是否已经初始化
    bool isInitialized() const {
        return initialized;
    }
};
class KalmanFilter3D3D {
private:
    cv::KalmanFilter kf;
    cv::Mat state;        // [x_cam, y_cam, z_cam, x_lidar, y_lidar, z_lidar, vx, vy, vz]
    cv::Mat measurement;  // [x_cam, y_cam, z_cam, x_lidar, y_lidar, z_lidar]
    bool initialized;

public:
    KalmanFilter3D3D()
            : kf(9, 6, 0),
              state(9, 1, CV_32F),
              measurement(6, 1, CV_32F),
              initialized(false)
    {
        // 状态转移矩阵 A（9x9）
        kf.transitionMatrix = (cv::Mat_<float>(9, 9) <<
                1,0,0, 0,0,0, 1,0,0,
                0,1,0, 0,0,0, 0,1,0,
                0,0,1, 0,0,0, 0,0,1,
                0,0,0, 1,0,0, 1,0,0,
                0,0,0, 0,1,0, 0,1,0,
                0,0,0, 0,0,1, 0,0,1,
                0,0,0, 0,0,0, 1,0,0,
                0,0,0, 0,0,0, 0,1,0,
                0,0,0, 0,0,0, 0,0,1);

        // 观测矩阵 H（6x9），提取前6个变量（图像点和点云点）
        kf.measurementMatrix = cv::Mat::zeros(6, 9, CV_32F);
        for (int i = 0; i < 6; ++i)
            kf.measurementMatrix.at<float>(i, i) = 1.0f;

        // 噪声协方差矩阵
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));     // Q
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2)); // R
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));           // P
    }

    void init(const cv::Point3f& cam_pt, const cv::Point3f& lidar_pt) {
        state.at<float>(0) = cam_pt.x;
        state.at<float>(1) = cam_pt.y;
        state.at<float>(2) = cam_pt.z;
        state.at<float>(3) = lidar_pt.x;
        state.at<float>(4) = lidar_pt.y;
        state.at<float>(5) = lidar_pt.z;
        state.at<float>(6) = 0;
        state.at<float>(7) = 0;
        state.at<float>(8) = 0;
        kf.statePost = state.clone();
        initialized = true;
    }

    // 更新滤波器状态，返回估计后的两个3D点
    std::pair<cv::Point3f, cv::Point3f> update(const cv::Point3f& cam_pt, const cv::Point3f& lidar_pt) {
        if (!initialized) {
            init(cam_pt, lidar_pt);
            return {cam_pt, lidar_pt};
        }

        kf.predict();
        measurement.at<float>(0) = cam_pt.x;
        measurement.at<float>(1) = cam_pt.y;
        measurement.at<float>(2) = cam_pt.z;
        measurement.at<float>(3) = lidar_pt.x;
        measurement.at<float>(4) = lidar_pt.y;
        measurement.at<float>(5) = lidar_pt.z;

        cv::Mat estimated = kf.correct(measurement);
        cv::Point3f cam_est(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
        cv::Point3f lidar_est(estimated.at<float>(3), estimated.at<float>(4), estimated.at<float>(5));
        return {cam_est, lidar_est};
    }

    bool isInitialized() const {
        return initialized;
    }
};
class KalmanFilter3D3Dva {
private:
    cv::KalmanFilter kf;
    cv::Mat state;        // [x_cam, y_cam, z_cam, x_lidar, y_lidar, z_lidar, vx, vy, vz, ax, ay, az]
    cv::Mat measurement;  // [x_cam, y_cam, z_cam, x_lidar, y_lidar, z_lidar]
    bool initialized;

public:
    KalmanFilter3D3Dva()
            : kf(12, 6, 0),
              state(12, 1, CV_32F),
              measurement(6, 1, CV_32F),
              initialized(false)
    {
        float dt = 0.1;  // 时间步长

        // 状态转移矩阵 A（12x12）
        kf.transitionMatrix = (cv::Mat_<float>(12, 12) <<
                                                       1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5*dt*dt, 0,        0,
                0, 1, 0, 0, 0, 0, 0,  dt, 0, 0,        0.5*dt*dt, 0,
                0, 0, 1, 0, 0, 0, 0,  0,  dt, 0,        0,        0.5*dt*dt,
                0, 0, 0, 1, 0, 0, dt, 0,  0,  0.5*dt*dt, 0,        0,
                0, 0, 0, 0, 1, 0, 0,  dt, 0,  0,        0.5*dt*dt, 0,
                0, 0, 0, 0, 0, 1, 0,  0,  dt, 0,        0,        0.5*dt*dt,
                0, 0, 0, 0, 0, 0, 1,  0,  0,  dt,       0,        0,
                0, 0, 0, 0, 0, 0, 0,  1,  0,  0,        dt,       0,
                0, 0, 0, 0, 0, 0, 0,  0,  1,  0,        0,        dt,
                0, 0, 0, 0, 0, 0, 0,  0,  0,  1,        0,        0,
                0, 0, 0, 0, 0, 0, 0,  0,  0,  0,        1,        0,
                0, 0, 0, 0, 0, 0, 0,  0,  0,  0,        0,        1);

        // 观测矩阵 H（6x12）
        kf.measurementMatrix = cv::Mat::zeros(6, 12, CV_32F);
        for (int i = 0; i < 6; ++i)
            kf.measurementMatrix.at<float>(i, i) = 1.0f;

        // 噪声协方差矩阵
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));     // Q
        kf.processNoiseCov.at<float>(9, 9) = 1e-2;  // 加速度噪声
        kf.processNoiseCov.at<float>(10, 10) = 1e-2;
        kf.processNoiseCov.at<float>(11, 11) = 1e-2;
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2)); // R
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));           // P
    }

    void init(const cv::Point3f& cam_pt, const cv::Point3f& lidar_pt) {
        state.at<float>(0) = cam_pt.x;
        state.at<float>(1) = cam_pt.y;
        state.at<float>(2) = cam_pt.z;
        state.at<float>(3) = lidar_pt.x;
        state.at<float>(4) = lidar_pt.y;
        state.at<float>(5) = lidar_pt.z;
        state.at<float>(6) = 0;  // 初始速度
        state.at<float>(7) = 0;
        state.at<float>(8) = 0;
        state.at<float>(9) = 0;  // 初始加速度
        state.at<float>(10) = 0;
        state.at<float>(11) = 0;
        kf.statePost = state.clone();
        initialized = true;
    }

    std::pair<cv::Point3f, cv::Point3f> update(const cv::Point3f& cam_pt, const cv::Point3f& lidar_pt) {
        if (!initialized) {
            init(cam_pt, lidar_pt);
            return {cam_pt, lidar_pt};
        }

        kf.predict();
        measurement.at<float>(0) = cam_pt.x;
        measurement.at<float>(1) = cam_pt.y;
        measurement.at<float>(2) = cam_pt.z;
        measurement.at<float>(3) = lidar_pt.x;
        measurement.at<float>(4) = lidar_pt.y;
        measurement.at<float>(5) = lidar_pt.z;

        cv::Mat estimated = kf.correct(measurement);
        cv::Point3f cam_est(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
        cv::Point3f lidar_est(estimated.at<float>(3), estimated.at<float>(4), estimated.at<float>(5));
        return {cam_est, lidar_est};
    }

    bool isInitialized() const {
        return initialized;
    }
};


#endif //WS_POSE_ESTIMATION_CALCULATE_R_T_3D_H

