#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <ceres/ceres.h>

/**
 * XICP命名空间：扩展ICP（eXtended Iterative Closest Point）算法实现
 * 主要功能：处理退化场景下的点云配准问题，通过检测和处理几何退化来提高配准鲁棒性
 */
namespace XICP {

/**
 * 退化感知方法枚举：定义处理ICP退化问题的不同策略
 */
    enum class DegeneracyAwarenessMethod {
        kNone = 0,                          // 不使用退化处理
        kSolutionRemapping = 1,             // 解映射方法：通过投影矩阵重映射解空间
        kEqualityConstraints = 2,           // 等式约束方法：对退化方向施加等式约束
        kOptimizedEqualityConstraints = 3,  // 优化的等式约束方法：改进的约束策略
        kInequalityConstraints = 4          // 不等式约束方法：使用不等式约束处理退化
    };

    /**
     * 局部化类别枚举：描述某个自由度是否可定位
     */
    enum class LocalizabilityCategory {
        kLocalizable = 0,        // 可定位：该自由度有足够的几何约束
        kNonLocalizable = 1      // 不可定位：该自由度缺乏几何约束（退化方向）
    };

    /**
     * 局部化采样类型枚举：描述点云对某个自由度的贡献程度
     */
    enum class LocalizabilitySamplingType {
        kHighContributionPoints = 0,    // 高贡献点：点云对该方向提供强约束
        kMixedContributionPoints = 1,   // 混合贡献点：点云提供中等约束
        kUnnecessary = 2,               // 不需要：该方向已足够约束
        kInsufficientPoints = 3         // 点数不足：缺少足够的约束点
    };

    /**
     * 退化检测参数结构体：包含所有用于退化检测和处理的参数
     * @tparam T 数值类型（通常为double或float）
     */
    template<typename T>
    struct DegeneracyDetectionParameters {
        // ===== 阈值参数 - 用于判断信息矩阵的质量 =====
        T enoughInformationThreshold = 100.0;       // 足够信息阈值：特征值高于此值认为该方向可定位
        T insufficientInformationThreshold = 10.0;  // 不足信息阈值：特征值低于此值认为信息不足
        T highInformationThreshold = 1000.0;        // 高信息阈值：特征值高于此值认为该方向约束很强
        T solutionRemappingThreshold = 120.0;       // Solution Remapping条件数阈值：条件数超过此值启用重映射
        T point2NormalMinimalAlignmentCosineThreshold = 0.866;  // 最小对齐余弦阈值：cos(30°)，用于判断点与法向量的对齐程度
        T point2NormalStrongAlignmentCosineThreshold = 0.966;   // 强对齐余弦阈值：cos(15°)，表示强烈对齐
        T inequalityBoundMultiplier = 0.5;          // 不等式边界乘数：用于计算不等式约束的边界范围

        // ===== 统计参数 - 记录点云的贡献信息 =====
        size_t numberOfPoints = 0;                          // 总点数：参与配准的点云总数
        size_t contributingNumberOfPoints = 0;              // 有贡献的点数：对配准有效贡献的点数
        size_t highlyContributingNumberOfPoints = 0;        // 高贡献点总数
        size_t highlyContributingNumberOfPoints_trans = 0;  // 对平移自由度高贡献的点数
        size_t highlyContributingNumberOfPoints_rot = 0;    // 对旋转自由度高贡献的点数
        T combinedContribution = 0.0;                       // 组合贡献度：所有点的综合贡献
        T highContribution = 0.0;                           // 高贡献度：高贡献点的贡献总和

        // ===== 控制参数 =====
        DegeneracyAwarenessMethod degeneracyAwarenessMethod = DegeneracyAwarenessMethod::kNone;  // 退化处理方法
        Eigen::Matrix4d transformationToOptimizationFrame = Eigen::Matrix4d::Identity();         // 到优化坐标系的变换矩阵（4x4齐次变换）
        bool isPrintingEnabled = false;                     // 是否启用调试打印输出
    };

    /**
     * 局部化约束结果结构体：存储对退化方向施加的约束值
     * @tparam T 数值类型
     */
    template<typename T>
    struct LocalizabilityConstraints {
        Eigen::Matrix<T, 3, 1> rotationConstraintValues_ = Eigen::Matrix<T, 3, 1>::Zero();     // 旋转约束值：3x1向量，对应roll、pitch、yaw三个方向的约束
        Eigen::Matrix<T, 3, 1> translationConstraintValues_ = Eigen::Matrix<T, 3, 1>::Zero();  // 平移约束值：3x1向量，对应x、y、z三个方向的约束
    };

    /**
     * 局部化分析结果结构体：存储退化检测的完整分析结果
     * @tparam T 数值类型
     */
    template<typename T>
    struct LocalizabilityAnalysisResults {
        // ===== 特征向量 - 描述信息矩阵的主方向 =====
        Eigen::Matrix<T, 3, 3> rotationEigenvectors_ = Eigen::Matrix<T, 3, 3>::Zero();     // 旋转信息矩阵的特征向量（3x3），列向量为特征向量
        Eigen::Matrix<T, 3, 3> translationEigenvectors_ = Eigen::Matrix<T, 3, 3>::Zero();  // 平移信息矩阵的特征向量（3x3），列向量为特征向量

        // ===== 局部化状态 - 标记每个自由度是否可定位 =====
        Eigen::Matrix<T, 3, 1> localizabilityRpy_ = Eigen::Matrix<T, 3, 1>::Zero();  // 旋转自由度的可定位性（3x1），0表示可定位，1表示退化
        Eigen::Matrix<T, 3, 1> localizabilityXyz_ = Eigen::Matrix<T, 3, 1>::Zero();  // 平移自由度的可定位性（3x1），0表示可定位，1表示退化

        // ===== 约束值 =====
        LocalizabilityConstraints<T> localizabilityConstraints_;  // 对退化方向的约束值

        // ===== Solution remapping投影矩阵 =====
        Eigen::Matrix<T, 6, 6> solutionRemappingProjectionMatrix_ = Eigen::Matrix<T, 6, 6>::Identity();  // 6x6投影矩阵，用于解空间重映射
    };

    /**
     * 单维度约束类：用于Ceres优化中对单个维度施加约束
     * 功能：限制优化变量的某一个维度为固定值
     */
    struct SingleDimensionConstraint {
        int dimension;  // 要约束的维度索引（0-5，对应6自由度）

        /**
         * [功能描述]：构造函数，指定要约束的维度
         * @param dim：维度索引（0-5）
         */
        SingleDimensionConstraint(int dim);

        /**
         * [功能描述]：计算残差的函数对象重载，用于Ceres优化
         * @param x：优化变量指针（6维向量）
         * @param residual：残差输出指针
         * @return 是否成功计算残差
         */
        template<typename T>
        bool operator()(const T *const x, T *residual) const;
    };

    /**
     * 点到平面线性代价函数：基于预计算的Hessian矩阵和梯度向量构建代价函数
     * 功能：将点到平面ICP问题转化为线性最小二乘问题 min ||Ax - b||^2
     */
    struct Point2PlaneLinearCostFunctor {
        Eigen::Matrix<double, 6, 6> A;  // 线性系统矩阵（6x6 Hessian矩阵）
        Eigen::Matrix<double, 6, 1> b;  // 线性系统右端项（6x1梯度向量）

        /**
         * [功能描述]：构造函数，初始化线性系统参数
         * @param A_：Hessian矩阵（6x6）
         * @param b_：梯度向量（6x1）
         */
        Point2PlaneLinearCostFunctor(const Eigen::Matrix<double, 6, 6> &A_,
                                     const Eigen::Matrix<double, 6, 1> &b_);

        /**
         * [功能描述]：计算残差的函数对象重载
         * @param x：优化变量指针（6维增量：[delta_roll, delta_pitch, delta_yaw, delta_x, delta_y, delta_z]）
         * @param residual：残差输出指针
         * @return 是否成功计算残差
         */
        template<typename T>
        bool operator()(const T *const x, T *residual) const;
    };

    /**
     * 点到平面残差自动微分版本：使用Ceres的自动微分功能计算雅可比矩阵
     * 功能：直接从点对和法向量计算点到平面距离，支持自动微分
     */
    struct Point2PlaneResidualAutoDiff {
        Eigen::Vector3d src_point;  // 源点坐标（3x1，在源点云坐标系下）
        Eigen::Vector3d tgt_point;  // 目标点坐标（3x1，在目标点云坐标系下）
        Eigen::Vector3d normal;     // 目标点处的平面法向量（3x1，单位向量）

        /**
         * [功能描述]：构造函数，初始化点对和法向量
         * @param src：源点坐标（3x1向量）
         * @param tgt：目标点坐标（3x1向量）
         * @param n：平面法向量（3x1单位向量）
         */
        Point2PlaneResidualAutoDiff(const Eigen::Vector3d &src,
                                    const Eigen::Vector3d &tgt,
                                    const Eigen::Vector3d &n);

        /**
         * [功能描述]：计算点到平面距离残差，支持自动微分
         * @param delta：6维增量参数（[delta_roll, delta_pitch, delta_yaw, delta_x, delta_y, delta_z]）
         * @param residual：输出残差（标量，表示点到平面距离）
         * @return 是否成功计算残差
         */
        template<typename T>
        bool operator()(const T *const delta, T *residual) const;
    };

    /**
     * 点到平面残差数值微分版本：使用数值微分方法计算雅可比矩阵
     * 功能：与AutoDiff版本功能相同，但使用数值微分，精度较低但更稳定
     */
    struct Point2PlaneResidualNumeric {
        Eigen::Vector3d src_point;  // 源点坐标（3x1）
        Eigen::Vector3d tgt_point;  // 目标点坐标（3x1）
        Eigen::Vector3d normal;     // 平面法向量（3x1）

        /**
         * [功能描述]：构造函数，初始化点对和法向量
         * @param src：源点坐标
         * @param tgt：目标点坐标
         * @param n：平面法向量
         */
        Point2PlaneResidualNumeric(const Eigen::Vector3d &src,
                                   const Eigen::Vector3d &tgt,
                                   const Eigen::Vector3d &n);

        /**
         * [功能描述]：计算点到平面距离残差，用于数值微分
         * @param delta：6维增量参数
         * @param residual：输出残差
         * @return 是否成功计算残差
         */
        bool operator()(const double *const delta, double *residual) const;
    };

    /**
     * 方向约束类：用于Ceres优化中对特定方向施加等式约束
     * 功能：约束优化变量在某个方向上的投影等于目标值（direction^T * x = target_value）
     */
    struct DirectionConstraint {
        /**
         * [功能描述]：构造函数，指定约束方向和目标值
         * @param direction：约束方向向量（6维）
         * @param target_value：目标值（标量）
         */
        DirectionConstraint(const Eigen::VectorXd &direction, double target_value);

        /**
         * [功能描述]：计算方向约束残差
         * @param x：优化变量指针（6维）
         * @param residual：残差输出（标量，等于 direction^T * x - target_value）
         * @return 是否成功计算残差
         */
        template<typename T>
        bool operator()(const T *const x, T *residual) const;

    private:
        Eigen::VectorXd direction_;  // 约束方向向量（6维单位向量）
        double target_value_;        // 目标值（通常为0）
    };

    /**
     * 不等式方向约束类：用于Ceres优化中对特定方向施加不等式约束
     * 功能：约束优化变量在某个方向上的投影不超过边界值（|direction^T * x| <= bound）
     */
    struct InequalityDirectionConstraint {
        /**
         * [功能描述]：构造函数，指定约束方向和边界
         * @param direction：约束方向向量（6维）
         * @param bound：边界值（正标量）
         */
        InequalityDirectionConstraint(const Eigen::VectorXd &direction, double bound);

        /**
         * [功能描述]：计算不等式约束残差，使用软约束方式
         * @param x：优化变量指针（6维）
         * @param residual：残差输出（当超出边界时非零）
         * @return 是否成功计算残差
         */
        template<typename T>
        bool operator()(const T *const x, T *residual) const;

    private:
        Eigen::VectorXd direction_;  // 约束方向向量（6维单位向量）
        double bound_;               // 约束边界值（正数）
    };

    /**
     * XICP核心类：实现扩展ICP算法的退化检测和处理功能
     * @tparam T 数值类型（通常为double）
     * 
     * 主要功能：
     * 1. 检测ICP配准中的几何退化（通过分析信息矩阵）
     * 2. 识别退化方向和可定位方向
     * 3. 提供多种退化处理策略（Solution Remapping、等式/不等式约束）
     * 4. 与Ceres优化器集成求解带约束的ICP问题
     */
    template<typename T>
    class XICPCore {
    public:
        // ===== 类型别名定义 =====
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;  // 动态大小矩阵
        using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;                // 动态大小列向量
        using Matrix6 = Eigen::Matrix<T, 6, 6>;                            // 6x6矩阵（对应6自由度）
        using Vector6 = Eigen::Matrix<T, 6, 1>;                            // 6x1向量
        using Matrix3 = Eigen::Matrix<T, 3, 3>;                            // 3x3矩阵
        using Vector3 = Eigen::Matrix<T, 3, 1>;                            // 3x1向量
        using Matrix4 = Eigen::Matrix<T, 4, 4>;                            // 4x4齐次变换矩阵

        /**
         * [功能描述]：默认构造函数
         */
        XICPCore() = default;

        /**
         * [功能描述]：设置退化检测参数
         * @param params：退化检测参数结构体
         */
        void setParameters(const DegeneracyDetectionParameters<T> &params);

        /**
         * [功能描述]：获取当前的退化检测参数
         * @return 退化检测参数的常量引用
         */
        const DegeneracyDetectionParameters<T> &getParameters() const;

        /**
         * [功能描述]：主要的退化检测接口函数，分析点云配准的退化情况
         * @param sourcePoints：源点云坐标矩阵（3xN或Nx3）
         * @param targetPoints：目标点云坐标矩阵（3xN或Nx3）
         * @param targetNormals：目标点云法向量矩阵（3xN或Nx3）
         * @param hessian：点到平面ICP的Hessian矩阵（6x6信息矩阵）
         * @param results：输出的局部化分析结果
         * @return 是否检测到退化（true表示存在退化方向）
         */
        bool detectDegeneracy(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：获取退化方向矩阵
         * @param results：局部化分析结果
         * @return 6x6矩阵，每列为一个退化方向（对应的特征向量）
         */
        Matrix6 getDegenerateDirections(const LocalizabilityAnalysisResults<T> &results) const;

        /**
         * [功能描述]：获取对退化方向施加的约束值
         * @param results：局部化分析结果
         * @return 6x1向量，包含对每个自由度的约束值
         */
        Vector6 getConstraintValues(const LocalizabilityAnalysisResults<T> &results) const;

        /**
         * [功能描述]：使用Ceres优化器求解带约束的ICP问题（基于预计算的Hessian）
         * @param hessian：6x6 Hessian矩阵（信息矩阵）
         * @param b：6x1梯度向量
         * @param xicpResults：XICP退化分析结果，包含约束信息
         * @param solution：输出的6维解向量（增量变换参数）
         */
        void solveDegenerateSystemWithCeres(
                const Eigen::Matrix<double, 6, 6> &hessian,
                const Eigen::Matrix<double, 6, 1> &b,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution);

        /**
         * [功能描述]：使用Ceres的自动微分功能求解带约束的ICP问题
         * @param valid_src：有效源点坐标向量
         * @param valid_tgt：有效目标点坐标向量
         * @param valid_normals：有效目标点法向量向量
         * @param xicpResults：XICP退化分析结果
         * @param solution：输出的6维解向量
         * @param useNumericDiff：是否使用数值微分（false则使用自动微分）
         */
        void solveDegenerateSystemWithCeresAutoDiff(
                const std::vector <Eigen::Vector3d> &valid_src,
                const std::vector <Eigen::Vector3d> &valid_tgt,
                const std::vector <Eigen::Vector3d> &valid_normals,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution,
                bool useNumericDiff);

        /**
         * [功能描述]：使用KKT条件（Karush-Kuhn-Tucker）求解带等式约束的线性系统
         * 通过增广系统 [A C^T; C 0][x; lambda] = [b; d] 求解约束优化问题
         * @param hessian：6x6 Hessian矩阵
         * @param b：6x1梯度向量
         * @param xicpResults：XICP退化分析结果，包含约束矩阵
         * @param solution：输出的6维解向量
         */
        void solveDegenerateSystemWithCeresKKT(
                const Eigen::Matrix<double, 6, 6> &hessian,
                const Eigen::Matrix<double, 6, 1> &b,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution);

    private:
        DegeneracyDetectionParameters<T> params_;  // 退化检测参数

        /**
         * [功能描述]：对6x6 Hessian矩阵进行3x3分块特征分析
         * 将Hessian分为旋转和平移两个3x3子块，分别进行特征值分解
         * @param hessian：6x6 Hessian矩阵（通常为 [R 0; 0 T] 块对角形式）
         * @param results：输出的特征向量和特征值分析结果
         */
        void eigenAnalysis3x3(const Matrix6 &hessian, LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：优化的退化检测方法（用于OptimizedEqualityConstraints模式）
         * 通过分析点云与特征向量的对齐关系快速判断退化
         * @param sourcePoints：源点云坐标矩阵
         * @param targetNormals：目标点云法向量矩阵
         * @param hessian：6x6 Hessian矩阵
         * @param results：输出的退化分析结果
         * @return 是否检测到退化
         */
        bool detectLocalizabilityOptimized(
                const Matrix &sourcePoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：三元退化检测方法（用于Equality和Inequality约束模式）
         * 将局部化程度分为三个等级：可定位、部分定位、不可定位
         * @param sourcePoints：源点云坐标矩阵
         * @param targetPoints：目标点云坐标矩阵
         * @param targetNormals：目标点云法向量矩阵
         * @param hessian：6x6 Hessian矩阵
         * @param results：输出的退化分析结果
         * @return 是否检测到退化
         */
        bool detectLocalizabilityTernary(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：Solution Remapping方法进行退化检测
         * 通过构建投影矩阵将解映射到可观测子空间
         * @param hessian：6x6 Hessian矩阵
         * @param results：输出的投影矩阵等结果
         * @return 是否检测到退化（条件数是否超过阈值）
         */
        bool detectLocalizabilitySolutionRemapping(
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：检测某个特征向量方向的可定位性
         * 通过计算点云对该方向的贡献度判断是否可定位
         * @param eigenvector：特征向量（3维，表示旋转或平移的某个主方向）
         * @param alignmentVectors：对齐向量矩阵（点云相关的几何向量）
         * @param combinedContribution：输出的综合贡献度
         * @param highContribution：输出的高贡献度
         * @return 该方向是否可定位（true表示可定位）
         */
        bool detectDirectionLocalizability(
                const Vector3 &eigenvector,
                const Matrix &alignmentVectors,
                T &combinedContribution,
                T &highContribution);

        /**
         * [功能描述]：比较函数，用于对对齐列表排序（按对齐度从大到小）
         * @param p1：第一个对齐对（索引，对齐度）
         * @param p2：第二个对齐对（索引，对齐度）
         * @return p1的对齐度是否大于p2
         */
        static bool compareAlignmentList(const std::pair <Eigen::Index, T> &p1,
                                         const std::pair <Eigen::Index, T> &p2);

        /**
         * [功能描述]：三元级别的子空间局部化检测
         * 针对某个特征向量方向，分析点云的贡献分布，决定采样策略
         * @param sourcePoints：源点云坐标矩阵
         * @param targetPoints：目标点云坐标矩阵
         * @param targetNormals：目标点云法向量矩阵
         * @param alignmentVectors：对齐向量矩阵
         * @param deltas：点对差值矩阵（target - source）
         * @param eigenvector：当前分析的特征向量（3维）
         * @param alignmentList：输出的对齐列表（按对齐度排序的点索引）
         * @param index：自由度索引（0-2）
         * @param isRotationSubspace：是否为旋转子空间（false表示平移子空间）
         * @param results：更新的局部化分析结果
         */
        void detectSubspaceLocalizabilityTernary(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix &alignmentVectors,
                const Matrix &deltas,
                const Vector3 &eigenvector,
                std::vector <std::pair<Eigen::Index, T>> &alignmentList,
                int index,
                bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：决定三元局部化的采样类型
         * 根据点云对某个自由度的贡献情况，决定需要何种采样策略
         * @param index：自由度索引（0-2）
         * @param isRotationSubspace：是否为旋转子空间
         * @param results：局部化分析结果
         * @return 采样类型（高贡献/混合/不需要/点数不足）
         */
        LocalizabilitySamplingType decideLocalizabilityLevelTernary(
                int index, bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);

        /**
         * [功能描述]：求解部分约束值（用于等式约束方法）
         * 从高对齐度的点子集中计算对退化方向的约束值
         * @param sourcePoints：源点云坐标矩阵
         * @param targetPoints：目标点云坐标矩阵
         * @param targetNormals：目标点云法向量矩阵
         * @param deltas：点对差值矩阵
         * @param alignmentList：按对齐度排序的点索引列表
         * @param pointsToSample：要采样的点数
         * @param eigenvector：退化方向的特征向量（3维）
         * @param index：自由度索引（0-2）
         * @param isRotationSubspace：是否为旋转子空间
         * @param results：更新约束值到该结果中
         */
        void solvePartialConstraints(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix &deltas,
                const std::vector <std::pair<Eigen::Index, T>> &alignmentList,
                Eigen::Index pointsToSample,
                const Vector3 &eigenvector,
                int index,
                bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);
    };

} // namespace XICP