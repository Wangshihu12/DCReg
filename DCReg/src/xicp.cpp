#include "xicp.h"


namespace XICP {

// SingleDimensionConstraint implementation
    SingleDimensionConstraint::SingleDimensionConstraint(int dim) : dimension(dim) {}

    template<typename T>
    bool SingleDimensionConstraint::operator()(const T *const x, T *residual) const {
        residual[0] = x[dimension];
        return true;
    }

// Point2PlaneLinearCostFunctor implementation
    Point2PlaneLinearCostFunctor::Point2PlaneLinearCostFunctor(const Eigen::Matrix<double, 6, 6> &A_,
                                                               const Eigen::Matrix<double, 6, 1> &b_) : A(A_), b(b_) {}

    template<typename T>
    bool Point2PlaneLinearCostFunctor::operator()(const T *const x, T *residual) const {
        // 计算线性系统的残差: r = Ax - b
        // 这样最小化 ||r||^2 等价于求解 Ax = b
        for (int i = 0; i < 6; ++i) {
            residual[i] = T(0);
            for (int j = 0; j < 6; ++j) {
                residual[i] += T(A(i, j)) * x[j];
            }
            residual[i] -= T(b(i));
        }
        return true;
    }

// Point2PlaneResidualAutoDiff implementation
    Point2PlaneResidualAutoDiff::Point2PlaneResidualAutoDiff(const Eigen::Vector3d &src,
                                                             const Eigen::Vector3d &tgt,
                                                             const Eigen::Vector3d &n)
            : src_point(src), tgt_point(tgt), normal(n) {}

    template<typename T>
    bool Point2PlaneResidualAutoDiff::operator()(const T *const delta, T *residual) const {
        // delta = [omega, v] 其中 omega是旋转增量，v是平移增量
        Eigen::Matrix<T, 3, 1> omega;
        Eigen::Matrix<T, 3, 1> v;
        for (int i = 0; i < 3; ++i) {
            omega(i) = delta[i];
            v(i) = delta[i + 3];
        }

        // 应用增量变换（注意这里是左乘形式）
        // p_new = p + omega × p + v
        Eigen::Matrix<T, 3, 1> src_T;
        for (int i = 0; i < 3; ++i) {
            src_T(i) = T(src_point(i));
        }

        // 计算 omega × p （叉积）
        Eigen::Matrix<T, 3, 1> omega_cross_p;
        omega_cross_p(0) = omega(1) * src_T(2) - omega(2) * src_T(1);
        omega_cross_p(1) = omega(2) * src_T(0) - omega(0) * src_T(2);
        omega_cross_p(2) = omega(0) * src_T(1) - omega(1) * src_T(0);

        // 变换后的点
        Eigen::Matrix<T, 3, 1> transformed_point = src_T + omega_cross_p + v;

        // 点到平面距离
        T distance = T(0);
        for (int i = 0; i < 3; ++i) {
            distance += (transformed_point(i) - T(tgt_point(i))) * T(normal(i));
        }

        residual[0] = distance;
        return true;
    }

// Point2PlaneResidualNumeric implementation
    Point2PlaneResidualNumeric::Point2PlaneResidualNumeric(const Eigen::Vector3d &src,
                                                           const Eigen::Vector3d &tgt,
                                                           const Eigen::Vector3d &n)
            : src_point(src), tgt_point(tgt), normal(n) {}

    bool Point2PlaneResidualNumeric::operator()(const double *const delta, double *residual) const {
        // 构建变换矩阵（小角度近似）
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R(0, 1) = -delta[2];
        R(0, 2) = delta[1];
        R(1, 0) = delta[2];
        R(1, 2) = -delta[0];
        R(2, 0) = -delta[1];
        R(2, 1) = delta[0];

        Eigen::Vector3d t(delta[3], delta[4], delta[5]);

        // 变换点
        Eigen::Vector3d transformed = R * src_point + t;

        // 计算点到平面距离
        residual[0] = normal.dot(transformed - tgt_point);
        return true;
    }

// DirectionConstraint implementation
    DirectionConstraint::DirectionConstraint(const Eigen::VectorXd &direction, double target_value)
            : direction_(direction), target_value_(target_value) {}

    template<typename T>
    bool DirectionConstraint::operator()(const T *const x, T *residual) const {
        *residual = T(0.0);
        for (int i = 0; i < direction_.size(); ++i) {
            *residual += x[i] * T(direction_(i));
        }
        *residual -= T(target_value_);
        return true;
    }

// InequalityDirectionConstraint implementation
    InequalityDirectionConstraint::InequalityDirectionConstraint(const Eigen::VectorXd &direction, double bound)
            : direction_(direction), bound_(bound) {}

    template<typename T>
    bool InequalityDirectionConstraint::operator()(const T *const x, T *residual) const {
        T projection = T(0.0);
        for (int i = 0; i < direction_.size(); ++i) {
            projection += x[i] * T(direction_(i));
        }

        // 不等式约束：如果投影在界限内，残差为0
        // 否则，残差是超出界限的部分
        T abs_projection = ceres::abs(projection);
        if (abs_projection <= T(bound_)) {
            *residual = T(0.0);
        } else {
            *residual = abs_projection - T(bound_);
        }
        return true;
    }

// XICPCore implementation
    template<typename T>
    void XICPCore<T>::setParameters(const DegeneracyDetectionParameters<T> &params) {
        params_ = params;
    }

    template<typename T>
    const DegeneracyDetectionParameters<T> &XICPCore<T>::getParameters() const {
        return params_;
    }

    /**
     * [功能描述]：退化检测主入口函数，根据配置的方法选择相应的退化检测策略
     * 该函数作为统一接口，将退化检测请求分派给不同的具体实现方法
     * 
     * @param sourcePoints：源点云坐标矩阵（3xN，每列为一个点的[x,y,z]坐标）
     * @param targetPoints：目标点云坐标矩阵（3xN）
     * @param targetNormals：目标点云法向量矩阵（3xN，每列为单位法向量）
     * @param hessian：点到平面ICP的Hessian矩阵（6x6信息矩阵，衡量6自由度的可观测性）
     * @param results：输出参数，存储退化检测的完整分析结果
     * 
     * @return 是否检测到退化（true表示存在至少一个不可定位的自由度）
     */
    template<typename T>
    bool XICPCore<T>::detectDegeneracy(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 更新点数统计信息，记录参与退化检测的总点数
        params_.numberOfPoints = sourcePoints.cols();

        // 根据配置的退化感知方法选择相应的检测策略
        switch (params_.degeneracyAwarenessMethod) {
            // 优化等式约束方法：快速检测，适用于实时应用
            case DegeneracyAwarenessMethod::kOptimizedEqualityConstraints:
                return detectLocalizabilityOptimized(sourcePoints, targetNormals, hessian, results);
            
            // 等式约束方法：通过采样高贡献点计算精确约束值
            case DegeneracyAwarenessMethod::kEqualityConstraints:
            // 不等式约束方法：使用边界约束处理退化方向
            case DegeneracyAwarenessMethod::kInequalityConstraints:
                // 两种方法共用三元检测逻辑，区别在于后续如何应用约束
                return detectLocalizabilityTernary(sourcePoints, targetPoints, targetNormals, hessian, results);
            
            // Solution Remapping方法：通过投影矩阵重映射解空间
            case DegeneracyAwarenessMethod::kSolutionRemapping:
                return detectLocalizabilitySolutionRemapping(hessian, results);

            // 默认情况（kNone）：不执行退化检测
            default:
                return false;
        }
    }

    template<typename T>
    typename XICPCore<T>::Matrix6
    XICPCore<T>::getDegenerateDirections(const LocalizabilityAnalysisResults<T> &results) const {
        Matrix6 directions = Matrix6::Zero();

        // 检查旋转方向
        for (int i = 0; i < 3; ++i) {
            if (results.localizabilityRpy_(i) == static_cast<T>(LocalizabilityCategory::kNonLocalizable)) {
                directions.col(i).head(3) = results.rotationEigenvectors_.col(i);
            }
        }

        // 检查平移方向
        for (int i = 0; i < 3; ++i) {
            if (results.localizabilityXyz_(i) == static_cast<T>(LocalizabilityCategory::kNonLocalizable)) {
                directions.col(i + 3).tail(3) = results.translationEigenvectors_.col(i);
            }
        }

        return directions;
    }

    template<typename T>
    typename XICPCore<T>::Vector6
    XICPCore<T>::getConstraintValues(const LocalizabilityAnalysisResults<T> &results) const {
        Vector6 constraintValues;
        constraintValues.head(3) = results.localizabilityConstraints_.rotationConstraintValues_;
        constraintValues.tail(3) = results.localizabilityConstraints_.translationConstraintValues_;
        return constraintValues;
    }

    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeres(
            const Eigen::Matrix<double, 6, 6> &hessian,
            const Eigen::Matrix<double, 6, 1> &b,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution) {

        // 初始化解
        double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // 创建Ceres问题
        ceres::Problem problem;

        // 使用正确的线性系统代价函数
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<Point2PlaneLinearCostFunctor, 6, 6>(
                        new Point2PlaneLinearCostFunctor(hessian, b)), nullptr, x);


        // 获取退化方向和约束值
        Eigen::Matrix<double, 6, 6> degenerateDirections = getDegenerateDirections(xicpResults);
        Eigen::Matrix<double, 6, 1> constraintValues = getConstraintValues(xicpResults);
        // 判断是等式约束还是不等式约束
        bool is_inequality = (getParameters().degeneracyAwarenessMethod ==
                              XICP::DegeneracyAwarenessMethod::kInequalityConstraints);
        double inequalityBoundMultiplier = getParameters().inequalityBoundMultiplier;

        // std::cout << "inequalityBoundMultiplier: " << inequalityBoundMultiplier << std::endl;

        // 正确的逻辑：只处理退化方向
        int num_constraints = 0;

        // 处理旋转方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);

            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                Eigen::VectorXd direction = degenerateDirections.col(i);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                }
            }
        }

        // 处理平移方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i + 3);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                double constraint_val = constraintValues(i + 3);  // 平移约束值在后3个
                Eigen::VectorXd direction = degenerateDirections.col(i + 3);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;

                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                }
            }
        }

        // 设置Ceres求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 1;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;

        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 复制结果
        for (int i = 0; i < 6; ++i) {
            solution(i) = x[i];
        }

        if (!summary.IsSolutionUsable()) {
            std::cerr << "[XICP-Ceres] Solver failed: " << summary.message << std::endl;
            std::cout << "[XICP-Ceres] Falling back to standard SVD solution..." << std::endl;

            // 如果Ceres失败，使用标准方法求解
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double singular_threshold = 1e-6;
            Eigen::Matrix<double, 6, 1> singular_values = svd.singularValues();
            Eigen::Matrix<double, 6, 1> inv_singular_values = Eigen::Matrix<double, 6, 1>::Zero();

            for (int i = 0; i < 6; ++i) {
                if (singular_values(i) > singular_threshold) {
                    inv_singular_values(i) = 1.0 / singular_values(i);
                }
            }

            solution = svd.matrixV() * inv_singular_values.asDiagonal() * svd.matrixU().transpose() * b;
            std::cout << "[XICP-Ceres] Fallback solution: " << solution.transpose() << std::endl;
        }
    }

    /**
     * [功能描述]：使用Ceres优化器的自动微分功能求解带约束的退化ICP问题
     * 该方法直接从点对和法向量构建优化问题，支持自动微分或数值微分计算雅可比矩阵
     * 相比于基于预计算Hessian的方法，该方法更灵活，但计算量稍大
     * 
     * 优化目标：
     * min sum_i ||n_i^T * (R*p_i + t - q_i)||^2
     * 约束条件：direction^T * x = constraint_val (等式) 或 |direction^T * x| <= bound (不等式)
     * 
     * @param valid_src：有效源点坐标向量（每个元素为3x1的Eigen::Vector3d）
     * @param valid_tgt：有效目标点坐标向量（与valid_src一一对应）
     * @param valid_normals：有效目标点法向量向量（单位向量，与valid_src一一对应）
     * @param xicpResults：XICP退化分析结果，包含退化方向、约束值等信息
     * @param solution：输出参数，6维解向量（[delta_roll, delta_pitch, delta_yaw, delta_x, delta_y, delta_z]）
     * @param useNumericDiff：是否使用数值微分（false则使用自动微分，自动微分更快更准确）
     */
    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeresAutoDiff(
            const std::vector <Eigen::Vector3d> &valid_src,
            const std::vector <Eigen::Vector3d> &valid_tgt,
            const std::vector <Eigen::Vector3d> &valid_normals,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution,
            bool useNumericDiff) {

        // ===== 步骤1：初始化优化变量 =====
        // 6维增量参数：[delta_roll, delta_pitch, delta_yaw, delta_x, delta_y, delta_z]
        // 初始猜测为零（假设当前变换已经接近最优）
        double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // ===== 步骤2：创建Ceres优化问题 =====
        ceres::Problem problem;

        // ===== 步骤3：添加点到平面残差项（数据项） =====
        // 根据选择使用数值微分或自动微分
        if (useNumericDiff) {
            // 数值微分方法：通过有限差分近似计算雅可比矩阵
            // 优点：实现简单，稳定性好；缺点：计算慢，精度较低
            for (size_t i = 0; i < valid_src.size(); ++i) {
                ceres::CostFunction *cost_function =
                        new ceres::NumericDiffCostFunction<Point2PlaneResidualNumeric,
                                ceres::CENTRAL,  // 使用中心差分（更准确）
                                1,               // 残差维度：1（标量点到平面距离）
                                6>(              // 参数维度：6（6自由度增量）
                                new Point2PlaneResidualNumeric(valid_src[i], valid_tgt[i], valid_normals[i]));
                problem.AddResidualBlock(cost_function, nullptr, x);
            }
        } else {
            // 自动微分方法：通过模板元编程自动计算精确的雅可比矩阵
            // 优点：快速且精确；缺点：编译时间较长
            for (size_t i = 0; i < valid_src.size(); ++i) {
                ceres::CostFunction *cost_function =
                        new ceres::AutoDiffCostFunction<Point2PlaneResidualAutoDiff, 
                                1,  // 残差维度：1
                                6>( // 参数维度：6
                                new Point2PlaneResidualAutoDiff(valid_src[i], valid_tgt[i], valid_normals[i]));
                problem.AddResidualBlock(cost_function, nullptr, x);
            }
        }

        // ===== 步骤4：获取退化信息 =====
        // degenerateDirections: 6x6矩阵，每列为一个退化方向（特征向量）
        Eigen::Matrix<double, 6, 6> degenerateDirections = getDegenerateDirections(xicpResults);
        // constraintValues: 6x1向量，对应每个退化方向的约束值
        Eigen::Matrix<double, 6, 1> constraintValues = getConstraintValues(xicpResults);

        // ===== 步骤5：确定约束类型和参数 =====
        // 判断使用等式约束还是不等式约束
        bool is_inequality = (getParameters().degeneracyAwarenessMethod ==
                              XICP::DegeneracyAwarenessMethod::kInequalityConstraints);
        // 不等式边界乘数：用于调整约束的松紧程度
        double inequalityBoundMultiplier = getParameters().inequalityBoundMultiplier;

        int num_constraints = 0;  // 统计添加的约束数量
        
        // ===== 步骤6：处理旋转自由度的退化方向 =====
        for (int i = 0; i < 3; ++i) {
            // 获取第i个旋转方向的约束值（roll/pitch/yaw）
            double constraint_val = constraintValues(i);
            // 计算约束权重：约束值越接近1，说明该方向越不确定，权重越小
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);

            // 检查该旋转自由度是否不可定位（退化）
            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                // 提取退化方向向量（6x1，在6自由度空间中的方向）
                Eigen::VectorXd direction = degenerateDirections.col(i);

                if (is_inequality) {
                    // ------ 不等式约束模式 ------
                    // 约束形式：|direction^T * x| <= constraint_val
                    // 含义：限制解在退化方向上的投影不超过边界
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    // 添加带权重的残差块（权重控制约束的强度）
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                } else {
                    // ------ 等式约束模式 ------
                    // 约束形式：direction^T * x = constraint_val
                    // 含义：强制解在退化方向上的投影等于约束值（通常接近0）
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                }
            }
        }

        // ===== 步骤7：处理平移自由度的退化方向 =====
        for (int i = 0; i < 3; ++i) {
            // 平移约束值在constraintValues的后3个位置（索引3-5）
            double constraint_val = constraintValues(i + 3);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);
            
            // 检查该平移自由度是否不可定位（退化）
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                double constraint_val = constraintValues(i + 3);  // 平移约束值在后3个元素
                // 提取退化方向向量（对应x/y/z方向）
                Eigen::VectorXd direction = degenerateDirections.col(i + 3);

                if (is_inequality) {
                    // ------ 不等式约束模式 ------
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                } else {
                    // ------ 等式约束模式 ------
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;

                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                }
            }
        }

        // ===== 步骤8：配置求解器参数 =====
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;  // 使用稠密QR分解（适合小规模问题）
        options.minimizer_type = ceres::TRUST_REGION;  // 信赖域方法（更稳定）
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;  // LM算法
        options.max_num_iterations = 1;  // 只进行一次迭代（ICP每次迭代只需一步优化）
        options.function_tolerance = 1e-6;   // 目标函数变化容差
        options.gradient_tolerance = 1e-10;  // 梯度容差
        options.parameter_tolerance = 1e-8;  // 参数变化容差

        // ===== 步骤9：求解优化问题 =====
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // ===== 步骤10：提取求解结果 =====
        // 将C数组结果复制到Eigen向量
        for (int i = 0; i < 6; ++i) {
            solution(i) = x[i];
        }
    }

    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeresKKT(
            const Eigen::Matrix<double, 6, 6> &hessian,
            const Eigen::Matrix<double, 6, 1> &b,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution) {

        // 收集退化的方向索引（模仿原始实现的readLocalizabilityFlags）
        std::vector<int> degenerateIndices;

        // 先检查旋转子空间
        for (int i = 0; i < 3; ++i) {
            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                degenerateIndices.push_back(i);
            }
        }

        // 再检查平移子空间
        for (int i = 0; i < 3; ++i) {
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                degenerateIndices.push_back(i + 3);
            }
        }

        const int numberOfConstraints = degenerateIndices.size();

        if (numberOfConstraints == 0) {
            // 没有约束，直接求解
            std::cout << "[XICP-KKT] No constraints, solving unconstrained system" << std::endl;
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double tolerance = 1e-6;
            Eigen::Matrix<double, 6, 1> singularValues = svd.singularValues();
            Eigen::Matrix<double, 6, 1> invSingularValues = Eigen::Matrix<double, 6, 1>::Zero();

            for (int i = 0; i < 6; ++i) {
                if (singularValues(i) > tolerance) {
                    invSingularValues(i) = 1.0 / singularValues(i);
                }
            }

            solution = svd.matrixV() * invSingularValues.asDiagonal() * svd.matrixU().transpose() * b;
            return;
        }

        std::cout << "[XICP-KKT] Solving with " << numberOfConstraints << " constraints" << std::endl;

        // 构建增广系统 - 这是关键修正
        // 增广矩阵的结构：[A C^T; C 0]
        Eigen::MatrixXd augmentedA = Eigen::MatrixXd::Zero(6 + numberOfConstraints,
                                                           6 + numberOfConstraints);
        Eigen::VectorXd augmentedb = Eigen::VectorXd::Zero(6 + numberOfConstraints);

        // 填充原始系统 A 和 b
        augmentedA.topLeftCorner(6, 6) = hessian;
        augmentedb.head(6) = b;

        // 构建约束矩阵 C 和约束向量 c
        // 这是最重要的修正部分
        int constraintCounter = 0;
        const int translationIndexOffset = 3;

        for (const auto &index : degenerateIndices) {
            const bool inRotationSubspace = (index < translationIndexOffset);

            // 构建6维约束向量
            Eigen::VectorXd constraintVector = Eigen::VectorXd::Zero(6);

            if (inRotationSubspace) {
                // 旋转约束：约束向量的前3个分量是对应的特征向量
                constraintVector.head(3) = xicpResults.rotationEigenvectors_.col(index);

                // 根据ICP.cpp，等式约束的约束值通常是0
                // 但如果计算出了特定值，也可以使用
                augmentedb(6 + constraintCounter) =
                        xicpResults.localizabilityConstraints_.rotationConstraintValues_(index);
            } else {
                // 平移约束：约束向量的后3个分量是对应的特征向量
                int transIndex = index - translationIndexOffset;
                constraintVector.tail(3) = xicpResults.translationEigenvectors_.col(transIndex);

                augmentedb(6 + constraintCounter) =
                        xicpResults.localizabilityConstraints_.translationConstraintValues_(transIndex);
            }

            // 填充约束矩阵（对称填充）
            // C矩阵的第constraintCounter行
            augmentedA.block(6 + constraintCounter, 0, 1, 6) = constraintVector.transpose();
            // C^T矩阵的第constraintCounter列
            augmentedA.block(0, 6 + constraintCounter, 6, 1) = constraintVector;

            constraintCounter++;
        }

        // 调试输出
        if (getParameters().isPrintingEnabled) {
            std::cout << "[XICP-KKT] Augmented system size: " << augmentedA.rows() << "x" << augmentedA.cols()
                      << std::endl;
            std::cout << "[XICP-KKT] Constraint values: ";
            for (int i = 0; i < numberOfConstraints; ++i) {
                std::cout << augmentedb(6 + i) << " ";
            }
            std::cout << std::endl;
        }

        // 求解增广系统
        Eigen::VectorXd augmentedSolution(6 + numberOfConstraints);

        // 首先尝试QR分解（遵循ICP.cpp的做法）
        Eigen::HouseholderQR <Eigen::MatrixXd> qr(augmentedA);
        augmentedSolution = qr.solve(augmentedb);

        // 检查求解质量
        double residualNorm = (augmentedA * augmentedSolution - augmentedb).norm();

        if (residualNorm > 1e-3 || augmentedSolution.hasNaN()) {
            std::cout << "[XICP-KKT] QR decomposition unstable (residual: " << residualNorm
                      << "), using SVD" << std::endl;

            // 使用SVD作为备选方案
            Eigen::JacobiSVD <Eigen::MatrixXd> svd(augmentedA,
                                                   Eigen::ComputeFullU | Eigen::ComputeFullV);

            // 使用相对容差
            double tolerance = 1e-10 * svd.singularValues()(0);

            Eigen::VectorXd singularValues = svd.singularValues();
            Eigen::VectorXd invSingularValues = Eigen::VectorXd::Zero(singularValues.size());

            int rank = 0;
            for (int i = 0; i < singularValues.size(); ++i) {
                if (singularValues(i) > tolerance) {
                    invSingularValues(i) = 1.0 / singularValues(i);
                    rank++;
                }
            }

            std::cout << "[XICP-KKT] System rank: " << rank << "/" << augmentedA.rows() << std::endl;

            augmentedSolution = svd.matrixV() * invSingularValues.asDiagonal() *
                                svd.matrixU().transpose() * augmentedb;
        }

        // 提取前6个分量作为解
        solution = augmentedSolution.head(6);

        // 输出拉格朗日乘子（用于调试）
        if (numberOfConstraints > 0) {
            Eigen::VectorXd lambda = augmentedSolution.tail(numberOfConstraints);
            std::cout << "[XICP-KKT] Lagrange multipliers: " << lambda.transpose() << std::endl;
        }

        // 验证约束满足情况
        double constraintResidual = 0.0;
        for (int i = 0; i < numberOfConstraints; ++i) {
            double violation = 0.0;
            int index = degenerateIndices[i];

            if (index < 3) {
                // 旋转约束
                Eigen::Vector3d constraint_dir = xicpResults.rotationEigenvectors_.col(index);
                violation = constraint_dir.dot(solution.head(3)) -
                            xicpResults.localizabilityConstraints_.rotationConstraintValues_(index);
            } else {
                // 平移约束
                int transIndex = index - 3;
                Eigen::Vector3d constraint_dir = xicpResults.translationEigenvectors_.col(transIndex);
                violation = constraint_dir.dot(solution.tail(3)) -
                            xicpResults.localizabilityConstraints_.translationConstraintValues_(transIndex);
            }

            constraintResidual += violation * violation;
            std::cout << "[XICP-KKT] Constraint " << i << " violation: " << violation << std::endl;
        }
        constraintResidual = std::sqrt(constraintResidual);
        std::cout << "[XICP-KKT] Total constraint residual: " << constraintResidual << std::endl;

        // 验证解的有效性
        if (solution.hasNaN()) {
            std::cerr << "[XICP-KKT] Invalid solution detected, falling back to unconstrained solution"
                      << std::endl;

            // 回退到无约束解
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double tolerance = 1e-6;
            Eigen::Matrix<double, 6, 1> singularValues = svd.singularValues();
            Eigen::Matrix<double, 6, 1> invSingularValues = Eigen::Matrix<double, 6, 1>::Zero();

            for (int j = 0; j < 6; ++j) {
                if (singularValues(j) > tolerance) {
                    invSingularValues(j) = 1.0 / singularValues(j);
                }
            }

            solution = svd.matrixV() * invSingularValues.asDiagonal() *
                       svd.matrixU().transpose() * b;
        }

        std::cout << "[XICP-KKT] Final solution: " << solution.transpose() << std::endl;
    }

    template<typename T>
    void XICPCore<T>::eigenAnalysis3x3(const Matrix6 &hessian, LocalizabilityAnalysisResults<T> &results) {
        // 旋转部分特征分析
        Eigen::JacobiSVD <Matrix3> svd_rot(hessian.template topLeftCorner<3, 3>(),
                                           Eigen::ComputeFullU | Eigen::ComputeFullV);
        results.rotationEigenvectors_ = svd_rot.matrixU();

        // 平移部分特征分析
        Eigen::JacobiSVD <Matrix3> svd_trans(hessian.template bottomRightCorner<3, 3>(),
                                             Eigen::ComputeFullU | Eigen::ComputeFullV);
        results.translationEigenvectors_ = svd_trans.matrixU();
    }

    template<typename T>
    bool XICPCore<T>::detectLocalizabilityOptimized(
            const Matrix &sourcePoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 3x3特征分析
        eigenAnalysis3x3(hessian, results);

        size_t numPoints = sourcePoints.cols();

        // 计算交叉积（用于旋转对齐）
        Matrix crosses(3, numPoints);
        for (size_t i = 0; i < numPoints; ++i) {
            Vector3 src_pt = sourcePoints.col(i).template head<3>();
            Vector3 tgt_normal = targetNormals.col(i).template head<3>();
            Vector3 cross = src_pt.cross(tgt_normal);
            T norm = cross.norm();
            crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
        }

        // 检测每个特征向量的局部化性
        for (int i = 0; i < 3; ++i) {
            // 检测旋转方向
            T rotContribution = 0.0;
            T rotHighContribution = 0.0;
            bool rotLocalizable = detectDirectionLocalizability(
                    results.rotationEigenvectors_.col(i),
                    crosses,
                    rotContribution,
                    rotHighContribution);

            results.localizabilityRpy_(i) = rotLocalizable ?
                                            static_cast<T>(LocalizabilityCategory::kLocalizable) :
                                            static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            results.localizabilityConstraints_.rotationConstraintValues_(i) = rotLocalizable ? 1.0 : 0.0;

            if (params_.isPrintingEnabled && !rotLocalizable) {
                std::cout << "Rotation axis " << i << " is non-localizable. "
                          << "Combined: " << rotContribution << "/" << params_.enoughInformationThreshold
                          << ", High: " << rotHighContribution << "/" << params_.insufficientInformationThreshold
                          << std::endl;
            }

            // 检测平移方向
            T transContribution = 0.0;
            T transHighContribution = 0.0;
            bool transLocalizable = detectDirectionLocalizability(
                    results.translationEigenvectors_.col(i),
                    targetNormals,
                    transContribution,
                    transHighContribution);

            results.localizabilityXyz_(i) = transLocalizable ?
                                            static_cast<T>(LocalizabilityCategory::kLocalizable) :
                                            static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            results.localizabilityConstraints_.translationConstraintValues_(i) = transLocalizable ? 1.0 : 0.0;

            if (params_.isPrintingEnabled && !transLocalizable) {
                std::cout << "Translation axis " << i << " is non-localizable. "
                          << "Combined: " << transContribution << "/" << params_.enoughInformationThreshold
                          << ", High: " << transHighContribution << "/" << params_.insufficientInformationThreshold
                          << std::endl;
            }
        }

        // 旋转特征向量回到map坐标系
        Matrix3 rot_to_map = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
        for (int i = 0; i < 3; ++i) {
            results.rotationEigenvectors_.col(i) = rot_to_map * results.rotationEigenvectors_.col(i);
            results.translationEigenvectors_.col(i) = rot_to_map * results.translationEigenvectors_.col(i);
        }

        return true;
    }

    /**
     * [功能描述]：三元级别的退化检测方法，用于等式约束和不等式约束模式
     * 该方法通过详细分析点云对每个特征向量方向的贡献，将局部化程度分为三个等级：
     * 1. 可定位（有足够的几何约束）
     * 2. 部分定位（需要从高贡献点采样计算约束）
     * 3. 不可定位（退化方向）
     * 
     * 核心思想：对旋转和平移的6个自由度分别分析，计算每个方向上点云的对齐程度和贡献度
     * 
     * @param sourcePoints：源点云坐标矩阵（3xN或更高维度，每列为一个点）
     * @param targetPoints：目标点云坐标矩阵（3xN）
     * @param targetNormals：目标点云法向量矩阵（3xN，每列为单位法向量）
     * @param hessian：6x6 Hessian矩阵（信息矩阵）
     * @param results：输出参数，存储局部化分析结果、特征向量、约束值等
     * 
     * @return 总是返回true（表示完成检测）
     */
    template<typename T>
    bool XICPCore<T>::detectLocalizabilityTernary(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // ===== 步骤1：进行3x3特征分析 =====
        // 将6x6 Hessian分解为旋转和平移两个3x3子块，分别进行特征值分解
        // 得到6个主方向（特征向量）和对应的信息量（特征值）
        eigenAnalysis3x3(hessian, results);

        // ===== 步骤2：计算源点云的几何中心 =====
        // 中心点用于后续计算相对位置和力矩
        Vector3 center = Vector3::Zero();  // 初始化中心为零向量（3x1）
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            center += sourcePoints.col(i).template head<3>();  // 累加每个点的xyz坐标
        }
        center /= static_cast<T>(sourcePoints.cols());  // 求平均得到中心点

        // ===== 步骤3：计算交叉积矩阵用于旋转对齐分析 =====
        // 交叉积 = (源点 - 中心) × 目标法向量，衡量旋转自由度的约束
        // 物理意义：表示点云几何对旋转运动的敏感度
        Matrix crosses(3, sourcePoints.cols());  // 存储N个交叉积向量（3xN）
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            Vector3 src_pt = sourcePoints.col(i).template head<3>() - center;  // 相对中心的位置（3x1）
            Vector3 tgt_normal = targetNormals.col(i).template head<3>();      // 目标点法向量（3x1）
            Vector3 cross = src_pt.cross(tgt_normal);  // 计算交叉积（3x1）
            T norm = cross.norm();  // 交叉积的模长
            // 归一化处理：模长小于1保持原值（避免放大噪声），否则归一化为单位向量
            crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
        }

        // ===== 步骤4：计算点对差值向量 =====
        // deltas = source - target，用于平移对齐分析和约束计算
        Matrix deltas(sourcePoints.rows(), sourcePoints.cols());  // 与sourcePoints同形状
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            deltas.col(i) = sourcePoints.col(i) - targetPoints.col(i);  // 每对点的差值向量
        }

        // ===== 步骤5：创建对齐列表容器 =====
        // 对齐列表用于存储点的索引和对齐度，后续用于排序和采样高贡献点
        std::vector <std::pair<Eigen::Index, T>> alignmentList;
        alignmentList.reserve(sourcePoints.cols());  // 预分配空间提高效率

        // ===== 步骤6：对6个自由度（3个旋转 + 3个平移）进行三元级别检测 =====
        for (int eigIdx = 0; eigIdx < 3; ++eigIdx) {
            // ------ 6.1 旋转子空间检测 ------
            // 分析第eigIdx个旋转特征向量方向的可定位性
            // 使用crosses（交叉积）作为对齐向量，因为它衡量旋转约束
            detectSubspaceLocalizabilityTernary(
                    sourcePoints, targetPoints, targetNormals, 
                    crosses,  // 对齐向量：使用交叉积分析旋转约束
                    deltas,
                    results.rotationEigenvectors_.col(eigIdx),  // 当前旋转特征向量（3x1）
                    alignmentList, eigIdx, 
                    true,  // isRotationSubspace = true
                    results);

            // 记录旋转方向的高贡献点数（用于统计和调试）
            params_.highlyContributingNumberOfPoints_rot = params_.highlyContributingNumberOfPoints;

            // ------ 6.2 平移子空间检测 ------
            // 分析第eigIdx个平移特征向量方向的可定位性
            // 使用targetNormals作为对齐向量，因为法向量直接约束平移
            detectSubspaceLocalizabilityTernary(
                    sourcePoints, targetPoints, targetNormals, 
                    targetNormals,  // 对齐向量：使用法向量分析平移约束
                    deltas,
                    results.translationEigenvectors_.col(eigIdx),  // 当前平移特征向量（3x1）
                    alignmentList, eigIdx, 
                    false,  // isRotationSubspace = false
                    results);
            
            // 记录平移方向的高贡献点数
            params_.highlyContributingNumberOfPoints_trans = params_.highlyContributingNumberOfPoints;
        }

        // ===== 步骤7：坐标系变换（如果需要） =====
        // 将特征向量从当前坐标系旋转回到优化坐标系
        // 这确保约束和特征向量在正确的参考系下表达
        if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
            // 提取旋转矩阵（4x4齐次变换的左上角3x3部分）
            Matrix3 rot_to_opt = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
            for (int i = 0; i < 3; ++i) {
                // 旋转每个特征向量：v_opt = R * v_current
                results.rotationEigenvectors_.col(i) = rot_to_opt * results.rotationEigenvectors_.col(i);
                results.translationEigenvectors_.col(i) = rot_to_opt * results.translationEigenvectors_.col(i);
            }
        }

        // ===== 步骤8：调试信息输出 =====
        if (params_.isPrintingEnabled) {
            // 输出平移和旋转的可定位性状态（3x1向量，0表示可定位，1表示不可定位）
            std::cout << "Translation localizability: " << results.localizabilityXyz_.transpose() << std::endl;
            std::cout << "Rotation localizability: " << results.localizabilityRpy_.transpose() << std::endl;
            // 输出高贡献点数量统计（平移和旋转）
            std::cout << "highlyContributingPoints size: " << params_.highlyContributingNumberOfPoints_trans << " "
                      << params_.highlyContributingNumberOfPoints_rot << std::endl;
        }

        return true;  // 检测完成
    }

    template<typename T>
    bool XICPCore<T>::detectLocalizabilitySolutionRemapping(
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 6x6特征分析
        Eigen::JacobiSVD <Matrix6> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vector6 eigenvalues = svd.singularValues();
        Matrix6 eigenvectors = svd.matrixU();

        // 计算条件数
        T conditionNumber = eigenvalues(0) / eigenvalues(5);

        if (params_.isPrintingEnabled) {
            std::cout << "[Solution Remapping] Condition number: " << conditionNumber
                      << ", threshold: " << params_.solutionRemappingThreshold << std::endl;
            std::cout << "Eigenvalues: " << eigenvalues.transpose() << std::endl;
        }

        // 构建投影矩阵 - 遵循ICP.cpp的实现
        results.solutionRemappingProjectionMatrix_ = Matrix6::Zero();

        // 根据特征值阈值决定保留哪些方向
        T eigenValueThreshold = params_.solutionRemappingThreshold;

        for (int i = 0; i < 6; ++i) {
            if (eigenvalues(i) >= eigenValueThreshold) {
                // 保留这个方向
                results.solutionRemappingProjectionMatrix_ +=
                        eigenvectors.col(i) * eigenvectors.col(i).transpose();
            } else {
                // 这是一个退化方向 - 更新局部化状态
                if (i < 3) {
                    // 对应旋转的某个轴
                    // 需要将6D特征向量映射到3D子空间
                    Vector3 rot_component = eigenvectors.col(i).head(3);
                    T rot_norm = rot_component.norm();
                    if (rot_norm > 0.5) {  // 主要是旋转退化
                        // 找到最大分量对应的轴
                        int max_idx = 0;
                        T max_val = std::abs(rot_component(0));
                        for (int j = 1; j < 3; ++j) {
                            if (std::abs(rot_component(j)) > max_val) {
                                max_val = std::abs(rot_component(j));
                                max_idx = j;
                            }
                        }
                        results.localizabilityRpy_(max_idx) =
                                static_cast<T>(LocalizabilityCategory::kNonLocalizable);
                    }
                } else {
                    // 对应平移的某个轴
                    Vector3 trans_component = eigenvectors.col(i).tail(3);
                    T trans_norm = trans_component.norm();
                    if (trans_norm > 0.5) {  // 主要是平移退化
                        // 找到最大分量对应的轴
                        int max_idx = 0;
                        T max_val = std::abs(trans_component(0));
                        for (int j = 1; j < 3; ++j) {
                            if (std::abs(trans_component(j)) > max_val) {
                                max_val = std::abs(trans_component(j));
                                max_idx = j;
                            }
                        }
                        results.localizabilityXyz_(max_idx) =
                                static_cast<T>(LocalizabilityCategory::kNonLocalizable);
                    }
                }
            }
        }

        // 如果投影矩阵是零矩阵，设为单位矩阵（避免完全失败）
        if (results.solutionRemappingProjectionMatrix_.norm() < 1e-6) {
            results.solutionRemappingProjectionMatrix_ = Matrix6::Identity();
            if (params_.isPrintingEnabled) {
                std::cout << "[Solution Remapping] Warning: Projection matrix was zero, using identity"
                          << std::endl;
            }
        }

        return true;
    }

    template<typename T>
    bool XICPCore<T>::detectDirectionLocalizability(
            const Vector3 &eigenvector,
            const Matrix &alignmentVectors,
            T &combinedContribution,
            T &highContribution) {

        combinedContribution = 0.0;
        highContribution = 0.0;
        bool informationIsEnough = false;

        for (Eigen::Index i = 0; i < alignmentVectors.cols() && !informationIsEnough; ++i) {
            Vector3 alignVec = alignmentVectors.col(i).template head<3>();
            T alignment = std::abs(alignVec.dot(eigenvector));

            if (alignment >= params_.point2NormalMinimalAlignmentCosineThreshold) {
                combinedContribution += alignment;
            }

            if (alignment >= params_.point2NormalStrongAlignmentCosineThreshold) {
                highContribution += alignment;
            }

            // 检查是否有足够信息（遵循ICP.cpp的逻辑）
            informationIsEnough = (combinedContribution >= params_.enoughInformationThreshold) ||
                                  (highContribution >= params_.insufficientInformationThreshold);
        }

        return informationIsEnough;
    }

    template<typename T>
    bool XICPCore<T>::compareAlignmentList(const std::pair <Eigen::Index, T> &p1,
                                           const std::pair <Eigen::Index, T> &p2) {
        return p1.second > p2.second;
    }

    template<typename T>
    void XICPCore<T>::detectSubspaceLocalizabilityTernary(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix &alignmentVectors,
            const Matrix &deltas,
            const Vector3 &eigenvector,
            std::vector <std::pair<Eigen::Index, T>> &alignmentList,
            int index,
            bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        // 清空对齐列表和重置计数器
        alignmentList.clear();
        params_.contributingNumberOfPoints = 0;
        params_.highlyContributingNumberOfPoints = 0;
        params_.combinedContribution = 0.0;
        params_.highContribution = 0.0;

        // 计算每个点的对齐贡献（使用对齐值排序，遵循ICP.cpp）
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            Vector3 alignVec = alignmentVectors.col(i).template head<3>();
            T alignment = std::abs(alignVec.dot(eigenvector));

            // 存储索引和对齐值（不是贡献值）
            alignmentList.push_back(std::make_pair(i, alignment));

            if (alignment >= params_.point2NormalMinimalAlignmentCosineThreshold) {
                params_.combinedContribution += alignment;
                params_.contributingNumberOfPoints++;
            }

            if (alignment >= params_.point2NormalStrongAlignmentCosineThreshold) {
                params_.highContribution += alignment;
                params_.highlyContributingNumberOfPoints++;
            }
        }

        // 决定局部化级别
        LocalizabilitySamplingType samplingType = decideLocalizabilityLevelTernary(
                index, isRotationSubspace, results);

        // 确定采样点数
        Eigen::Index pointsToSample = 0;
        switch (samplingType) {
            case LocalizabilitySamplingType::kHighContributionPoints:
                pointsToSample = params_.highlyContributingNumberOfPoints;
                break;
            case LocalizabilitySamplingType::kMixedContributionPoints:
                pointsToSample = params_.contributingNumberOfPoints;
                break;
            case LocalizabilitySamplingType::kUnnecessary:
            case LocalizabilitySamplingType::kInsufficientPoints:
                return;  // 不需要采样
        }

        if (params_.isPrintingEnabled) {
//                std::cout << "highlyContributingPoints size: " << params_.highlyContributingNumberOfPoints_trans << " "
//                          << params_.highlyContributingNumberOfPoints_rot << std::endl;
        }

        // 如果需要采样，执行采样和约束计算
        if (pointsToSample > 0) {
            // 确保采样数量合理
            pointsToSample = std::min(pointsToSample, sourcePoints.cols());
            pointsToSample = std::max(pointsToSample,
                                      static_cast<Eigen::Index>(params_.insufficientInformationThreshold));

            // 部分排序获取贡献最大的点
            std::partial_sort(alignmentList.begin(),
                              alignmentList.begin() + pointsToSample,
                              alignmentList.end(),
                              compareAlignmentList);

            // 执行约束求解
            solvePartialConstraints(sourcePoints, targetPoints, targetNormals, deltas,
                                    alignmentList, pointsToSample, eigenvector, index,
                                    isRotationSubspace, results);

//                std::cout << "constriant: " << results.localizabilityConstraints_.transpose() << std::endl;
        }
    }

    template<typename T>
    LocalizabilitySamplingType XICPCore<T>::decideLocalizabilityLevelTernary(
            int index, bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        auto &localizability = isRotationSubspace ?
                               results.localizabilityRpy_(index) : results.localizabilityXyz_(index);
        auto &constraintValue = isRotationSubspace ?
                                results.localizabilityConstraints_.rotationConstraintValues_(index) :
                                results.localizabilityConstraints_.translationConstraintValues_(index);

        // 完全可定位
        if (params_.combinedContribution >= params_.highInformationThreshold ||
            params_.highContribution >= params_.enoughInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kLocalizable);
            constraintValue = 1.0;
            return LocalizabilitySamplingType::kUnnecessary;
        }

        // 部分可定位 - 混合贡献
        if (params_.combinedContribution >= params_.enoughInformationThreshold &&
            params_.combinedContribution < params_.highInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            // 关键区别：等式约束 vs 不等式约束
            if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
                // 等式约束：二值化为0（退化方向）
                constraintValue = 0.0;
            } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
                // 不等式约束：连续值
                constraintValue = params_.inequalityBoundMultiplier *
                                  (params_.combinedContribution / params_.highInformationThreshold);
                constraintValue = std::min(constraintValue, T(1.0));
            }
            return LocalizabilitySamplingType::kMixedContributionPoints;
        }

        // 部分可定位 - 高贡献点
        if (params_.highContribution >= params_.insufficientInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
                // 等式约束：二值化为0
                constraintValue = 0.0;
            } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
                // 不等式约束：部分约束
                constraintValue = params_.inequalityBoundMultiplier * 0.5;
                constraintValue = std::min(constraintValue, T(1.0));
            }
            return LocalizabilitySamplingType::kHighContributionPoints;
        }

        // 不可定位
        localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);
        constraintValue = 0.0;  // 对所有方法都是0
        return LocalizabilitySamplingType::kInsufficientPoints;
    }

    template<typename T>
    void XICPCore<T>::solvePartialConstraints(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix &deltas,
            const std::vector <std::pair<Eigen::Index, T>> &alignmentList,
            Eigen::Index pointsToSample,
            const Vector3 &eigenvector,
            int index,
            bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        // 创建采样点云和相应的法向量、增量
        Matrix sampledPoints(sourcePoints.rows(), pointsToSample);
        Matrix sampledNormals(targetNormals.rows(), pointsToSample);
        Matrix sampledDeltas(deltas.rows(), pointsToSample);

        for (Eigen::Index i = 0; i < pointsToSample; ++i) {
            Eigen::Index idx = alignmentList[i].first;
            sampledPoints.col(i) = sourcePoints.col(idx);
            sampledNormals.col(i) = targetNormals.col(idx);
            sampledDeltas.col(i) = deltas.col(idx);
        }

        // 计算约束值
        T constraintValue = 0.0;
        const T thr = T(1e-5);

        if (!isRotationSubspace) {
            // 平移约束
            Matrix3 partial_A = sampledNormals.topRows(3) * sampledNormals.topRows(3).transpose();

            // 计算点积 dot = dot(deltas, normals)
            Matrix dotProd = Matrix::Zero(1, sampledNormals.cols());
            for (Eigen::Index i = 0; i < sampledNormals.rows(); ++i) {
                dotProd += (sampledDeltas.row(i).array() * sampledNormals.row(i).array()).matrix();
            }

            Vector3 partial_b = -(sampledNormals.topRows(3) * dotProd.transpose());

            // 求解
            Vector3 x_partial;
            if (partial_A.determinant() > thr) {
                x_partial = partial_A.ldlt().solve(partial_b);
            } else {
                // 使用SVD进行更稳定的求解
                Eigen::JacobiSVD <Matrix3> svd(partial_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                x_partial = svd.solve(partial_b);
            }

            // 将特征向量旋转到优化坐标系
            Vector3 rotatedEigenVector = eigenvector;
            if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
                Matrix3 T_op = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
                rotatedEigenVector = T_op * eigenvector;
            }

            constraintValue = rotatedEigenVector.transpose() * x_partial;
        } else {
            // 旋转约束
            // 重新计算采样点的交叉积
            Matrix crosses(3, pointsToSample);
            Vector3 center = sampledPoints.topRows(3).rowwise().mean();

            for (Eigen::Index i = 0; i < pointsToSample; ++i) {
                Vector3 pt = sampledPoints.col(i).head(3) - center;
                Vector3 normal = sampledNormals.col(i).head(3);
                Vector3 cross = pt.cross(normal);
                T norm = cross.norm();
                crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
            }

            Matrix3 partial_A = crosses * crosses.transpose();

            // 计算点积
            Matrix dotProd = Matrix::Zero(1, sampledNormals.cols());
            for (Eigen::Index i = 0; i < sampledNormals.rows(); ++i) {
                dotProd += (sampledDeltas.row(i).array() * sampledNormals.row(i).array()).matrix();
            }

            Vector3 partial_b = -(crosses * dotProd.transpose());

            // 使用更稳定的求解方法（遵循ICP.cpp的实现）
            Vector3 x_partial;
            Vector3 y;
            if (partial_A.determinant() > thr) {
                // LU分解
                Eigen::PartialPivLU <Matrix3> lu(partial_A);
                Matrix3 l = Matrix3::Identity();
                l.template triangularView<Eigen::StrictlyLower>() = lu.matrixLU();
                Matrix3 u = lu.matrixLU().template triangularView<Eigen::Upper>();

                Matrix3 new_A = l.transpose() * l;
                Vector3 new_b = l.transpose() * (lu.permutationP() * partial_b);

                // 使用double精度进行SVD求解
                y = new_A.template cast<double>().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(
                        new_b.template cast<double>()).
                        template cast<T>();
                x_partial = u.inverse() * y;
            } else {
                x_partial = Vector3::Zero();
            }

            // 旋转特征向量
            Vector3 rotatedEigenVector = eigenvector;
            if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
                Matrix3 T_op = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
                rotatedEigenVector = T_op * eigenvector;
            }

            constraintValue = rotatedEigenVector.transpose() * x_partial;
        }

        // 更新约束值
        auto &finalConstraintValue = isRotationSubspace ?
                                     results.localizabilityConstraints_.rotationConstraintValues_(index) :
                                     results.localizabilityConstraints_.translationConstraintValues_(index);

        // 对于等式约束，约束值已经在decideLocalizabilityLevelTernary中设置
        // 只有当需要采样求解时才会调用这个函数，此时约束值应该保持为0
        if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
            // 等式约束：保持二值（已在decide函数中设置为0）
            // 不需要更新，因为退化方向的约束值应该是0
            if (params_.isPrintingEnabled) {
                std::cout << "[Partial Constraints] "
                          << (isRotationSubspace ? "Rotation" : "Translation")
                          << " axis " << index
                          << " constraint value: " << finalConstraintValue
                          << " (Equality constraint)" << std::endl;
            }
        } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
            // 不等式约束：使用计算出的值并缩放
            finalConstraintValue = std::min(std::abs(constraintValue) * params_.inequalityBoundMultiplier, T(1.0));
            if (params_.isPrintingEnabled) {
                std::cout << "[Partial Constraints] "
                          << (isRotationSubspace ? "Rotation" : "Translation")
                          << " axis " << index
                          << " constraint value: " << finalConstraintValue
                          << " (Inequality constraint)" << std::endl;
            }
        } else {
            // 对于等式约束，保留原始约束值
            finalConstraintValue = constraintValue;
        }
    }

// 模板函数的显式实例化，用于自动微分
    template bool SingleDimensionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool SingleDimensionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool Point2PlaneLinearCostFunctor::operator()<double>(const double *const x, double *residual) const;

    template bool Point2PlaneLinearCostFunctor::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool Point2PlaneResidualAutoDiff::operator()<double>(const double *const delta, double *residual) const;

    template bool Point2PlaneResidualAutoDiff::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const delta, ceres::Jet<double, 6>
    *residual) const;

    template bool DirectionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool DirectionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool InequalityDirectionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool InequalityDirectionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

// 显式实例化XICPCore类模板
//    template class XICPCore<float>;
    template
    class XICPCore<double>;

} // namespace XICP