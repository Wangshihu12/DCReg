#include "icp_test_runner.h"

#include <numeric>
#include <algorithm>
#include <cmath>

namespace ICPRunner {

// Constructor
    TestRunner::TestRunner(const Config &config) : config_(config) {
        source_cloud_.reset(new pcl::PointCloud <PointT>);
        target_cloud_.reset(new pcl::PointCloud <PointT>);

        //use ptr
        dcreg_ = std::make_shared<DCReg>();
        dcreg_->setConfig(config);
    }

// Load configuration from YAML file
    bool loadConfig(const std::string &filename, Config &config) {
        try {
            YAML::Node yaml = YAML::LoadFile(filename);

            // Test configuration
            if (yaml["test"]) {
                config.num_runs = yaml["test"]["num_runs"].as<int>();
                config.save_pcd = yaml["test"]["save_pcd"].as<bool>();
                config.save_error_pcd = yaml["test"]["save_error_pcd"].as<bool>();
                config.visualize = yaml["test"]["visualize"].as<bool>();
            }

            // File paths
            if (yaml["paths"]) {
                config.folder_path = yaml["paths"]["folder_path"].as<std::string>();
                config.source_pcd = yaml["paths"]["source_pcd"].as<std::string>();
                config.target_pcd = yaml["paths"]["target_pcd"].as<std::string>();
                config.output_folder = yaml["paths"]["output_folder"].as<std::string>();
            }

            // ICP parameters
            if (yaml["icp"]) {
                config.search_radius = yaml["icp"]["search_radius"].as<double>();
                config.max_iterations = yaml["icp"]["max_iterations"].as<int>();
                config.normal_nn = yaml["icp"]["normal_nn"].as<int>();
                config.error_threshold = yaml["icp"]["error_threshold"].as<double>();
                config.CONVERGENCE_THRESH_TRANS = yaml["icp"]["CONVERGENCE_THRESH_TRANS"].as<double>();
                config.CONVERGENCE_THRESH_ROT = yaml["icp"]["CONVERGENCE_THRESH_ROT"].as<double>();
                std::cout << "CONVERGENCE_THRESH_TRANS: " << config.CONVERGENCE_THRESH_TRANS << std::endl;
                std::cout << "CONVERGENCE_THRESH_ROT: " << config.CONVERGENCE_THRESH_ROT << std::endl;
            }

            // Initial noise
            if (yaml["initial_noise"]) {
                config.initial_noise.x = yaml["initial_noise"]["x"].as<double>();
                config.initial_noise.y = yaml["initial_noise"]["y"].as<double>();
                config.initial_noise.z = yaml["initial_noise"]["z"].as<double>();
                config.initial_noise.roll = pcl::deg2rad(yaml["initial_noise"]["roll_deg"].as<double>());
                config.initial_noise.pitch = pcl::deg2rad(yaml["initial_noise"]["pitch_deg"].as<double>());
                config.initial_noise.yaw = pcl::deg2rad(yaml["initial_noise"]["yaw_deg"].as<double>());
                config.initial_matrix = Pose6D2Matrix(config.initial_noise);
            }

            // Ground truth
            if (yaml["gt_pose"]) {
                config.gt_pose.x = yaml["gt_pose"]["x"].as<double>();
                config.gt_pose.y = yaml["gt_pose"]["y"].as<double>();
                config.gt_pose.z = yaml["gt_pose"]["z"].as<double>();
                config.gt_pose.roll = pcl::deg2rad(yaml["gt_pose"]["roll_deg"].as<double>());
                config.gt_pose.pitch = pcl::deg2rad(yaml["gt_pose"]["pitch_deg"].as<double>());
                config.gt_pose.yaw = pcl::deg2rad(yaml["gt_pose"]["yaw_deg"].as<double>());
                config.gt_matrix = Pose6D2Matrix(config.gt_pose);
            }

            // Degeneracy thresholds
            if (yaml["degeneracy"]) {
                config.icp_params.DEGENERACY_THRES_COND = yaml["degeneracy"]["condition_threshold"].as<double>();
                config.icp_params.DEGENERACY_THRES_EIG = yaml["degeneracy"]["eigenvalue_threshold"].as<double>();
            }

            // Method parameters
            if (yaml["method_params"]) {
                if (yaml["method_params"]["adaptive_reg"]) {
                    config.icp_params.ADAPTIVE_REG_ALPHA = yaml["method_params"]["adaptive_reg"]["alpha"].as<double>();
                }
                if (yaml["method_params"]["standard_reg"]) {
                    config.icp_params.STD_REG_GAMMA = yaml["method_params"]["standard_reg"]["gamma"].as<double>();
                }
                if (yaml["method_params"]["pcg"]) {
                    config.icp_params.KAPPA_TARGET = yaml["method_params"]["pcg"]["kappa_target"].as<double>();
                    config.icp_params.PCG_TOLERANCE = yaml["method_params"]["pcg"]["tolerance"].as<double>();
                    config.icp_params.PCG_MAX_ITER = yaml["method_params"]["pcg"]["max_iter"].as<int>();
                }
                if (yaml["method_params"]["tsvd"]) {
                    config.icp_params.TSVD_SINGULAR_THRESH = yaml["method_params"]["tsvd"]["singular_threshold"].as<double>();
                }
                if (yaml["method_params"]["solution_remapping"]) {
                    config.icp_params.LOAM_EIGEN_THRESH = yaml["method_params"]["solution_remapping"]["eigen_threshold"].as<double>();
                }
            }

            // Load ICP parameters
            if (yaml["icp_params"]) {
                auto icp = yaml["icp_params"];

                // Load X-ICP parameters
                if (icp["XICP_ENOUGH_INFO_THRESHOLD"]) {
                    config.icp_params.XICP_ENOUGH_INFO_THRESHOLD = icp["XICP_ENOUGH_INFO_THRESHOLD"].as<double>();
                }
                if (icp["XICP_INSUFFICIENT_INFO_THRESHOLD"]) {
                    config.icp_params.XICP_INSUFFICIENT_INFO_THRESHOLD = icp["XICP_INSUFFICIENT_INFO_THRESHOLD"].as<double>();
                }
                if (icp["XICP_HIGH_INFO_THRESHOLD"]) {
                    config.icp_params.XICP_HIGH_INFO_THRESHOLD = icp["XICP_HIGH_INFO_THRESHOLD"].as<double>();
                }
                if (icp["XICP_SOLUTION_REMAPPING_THRESHOLD"]) {
                    config.icp_params.XICP_SOLUTION_REMAPPING_THRESHOLD = icp["XICP_SOLUTION_REMAPPING_THRESHOLD"].as<double>();
                }
                if (icp["XICP_MINIMAL_ALIGNMENT_ANGLE"]) {
                    config.icp_params.XICP_MINIMAL_ALIGNMENT_ANGLE = icp["XICP_MINIMAL_ALIGNMENT_ANGLE"].as<double>();
                }
                if (icp["XICP_STRONG_ALIGNMENT_ANGLE"]) {
                    config.icp_params.XICP_STRONG_ALIGNMENT_ANGLE = icp["XICP_STRONG_ALIGNMENT_ANGLE"].as<double>();
                }
                if (icp["XICP_INEQUALITY_BOUND_MULTIPLIER"]) {
                    config.icp_params.XICP_INEQUALITY_BOUND_MULTIPLIER = icp["XICP_INEQUALITY_BOUND_MULTIPLIER"].as<double>();
                }
            }

            // Test methods
            if (yaml["test_methods"]) {
                for (auto it = yaml["test_methods"].begin(); it != yaml["test_methods"].end(); ++it) {
                    std::string method_name = it->first.as<std::string>();
                    auto methods = it->second.as < std::vector < std::string >> ();
                    config.test_methods[method_name] = std::make_pair(methods[0], methods[1]);
                }
            }

            // Print loaded parameters for verification
            std::cout << "\n=== Loaded Configuration ===" << std::endl;
            std::cout << "STD_REG_GAMMA: " << config.icp_params.STD_REG_GAMMA << std::endl;
            std::cout << "ADAPTIVE_REG_ALPHA: " << config.icp_params.ADAPTIVE_REG_ALPHA << std::endl;
            std::cout << "KAPPA_TARGET: " << config.icp_params.KAPPA_TARGET << std::endl;
            std::cout << "DEGENERACY_THRES_COND: " << config.icp_params.DEGENERACY_THRES_COND << std::endl;
            std::cout << "DEGENERACY_THRES_EIG: " << config.icp_params.DEGENERACY_THRES_EIG << std::endl;
            std::cout << "USE_SO3 ICP: " << config.use_so3_parameterization << std::endl;
            std::cout << "==========================\n" << std::endl;

            return true;
        } catch (const YAML::Exception &e) {
            std::cerr << "Error loading YAML config: " << e.what() << std::endl;
            return false;
        }
    }

/**
 * [功能描述]：加载ICP算法所需的源点云和目标点云数据。
 * @return bool：加载成功返回true，失败返回false
 */
bool TestRunner::loadPointClouds() {
    // 构建源点云文件的完整路径：配置文件夹路径 + 源点云文件名
    std::string source_path = config_.folder_path + config_.source_pcd;
    // 构建目标点云文件的完整路径：配置文件夹路径 + 目标点云文件名  
    std::string target_path = config_.folder_path + config_.target_pcd;

    // 使用PCL库加载源点云文件到source_cloud_成员变量中
    // 返回-1表示加载失败，输出错误信息并返回false
    if (pcl::io::loadPCDFile<PointT>(source_path, *source_cloud_) == -1) {
        std::cerr << "Failed to load source cloud: " << source_path << std::endl;
        return false;
    }
    
    // 使用PCL库加载目标点云文件到target_cloud_成员变量中
    // 返回-1表示加载失败，输出错误信息并返回false
    if (pcl::io::loadPCDFile<PointT>(target_path, *target_cloud_) == -1) {
        std::cerr << "Failed to load target cloud: " << target_path << std::endl;
        return false;
    }
    
    // 检查加载的点云是否为空（无点数据）
    // 空点云无法进行ICP配准，输出错误信息并返回false
    if (source_cloud_->empty() || target_cloud_->empty()) {
        std::cerr << "Error: Loaded point cloud is empty: " << source_path << std::endl;
        return false;
    }
    
    // 输出成功加载的点云信息，包括源点云和目标点云的点数量
    std::cout << "Loaded point clouds - Source: " << source_cloud_->size()
              << " points, Target: " << target_cloud_->size() << " points" << std::endl;

    return true;  // 所有检查通过，点云加载成功
}

// String to enum conversions
    DetectionMethod TestRunner::stringToDetectionMethod(const std::string &str) {
        static std::map <std::string, DetectionMethod> map = {
                {"NONE_DETE",               DetectionMethod::NONE_DETE},
                {"SCHUR_CONDITION_NUMBER",  DetectionMethod::SCHUR_CONDITION_NUMBER},
                {"FULL_EVD_MIN_EIGENVALUE", DetectionMethod::FULL_EVD_MIN_EIGENVALUE},
                {"EVD_SUB_CONDITION",       DetectionMethod::EVD_SUB_CONDITION},
                {"FULL_SVD_CONDITION",      DetectionMethod::FULL_SVD_CONDITION},
                {"O3D",                     DetectionMethod::O3D},
                {"SUPERLOC",                DetectionMethod::SUPERLOC},  // 新增SUPERLOC
                {"XICP_OPTIMIZED_EQUALITY", DetectionMethod::XICP_OPTIMIZED_EQUALITY},  // 新增SUPERLOC
                {"XICP_INEQUALITY",         DetectionMethod::XICP_INEQUALITY},  // 新增SUPERLOC
                {"XICP_EQUALITY",           DetectionMethod::XICP_EQUALITY},  // 新增SUPERLOC
                {"XICP_SOLUTION_REMAPPING", DetectionMethod::XICP_SOLUTION_REMAPPING}  // 新增SUPERLOC
        };
        return map[str];
    }

    HandlingMethod TestRunner::stringToHandlingMethod(const std::string &str) {
        static std::map <std::string, HandlingMethod> map = {
                {"NONE_HAND",               HandlingMethod::NONE_HAND},
                {"STANDARD_REGULARIZATION", HandlingMethod::STANDARD_REGULARIZATION},
                {"ADAPTIVE_REGULARIZATION", HandlingMethod::ADAPTIVE_REGULARIZATION},
                {"PRECONDITIONED_CG",       HandlingMethod::PRECONDITIONED_CG},
                {"SOLUTION_REMAPPING",      HandlingMethod::SOLUTION_REMAPPING},
                {"TRUNCATED_SVD",           HandlingMethod::TRUNCATED_SVD},
                {"O3D",                     HandlingMethod::O3D},
                {"SUPERLOC",                HandlingMethod::SUPERLOC},  // 新增SUPERLOC
                {"XICP_CONSTRAINT",         HandlingMethod::XICP_CONSTRAINT},  // 新增SUPERLOC
                {"XICP_PROJECTION",         HandlingMethod::XICP_PROJECTION}  // 新增SUPERLOC
        };
        auto it = map.find(str);
        if (it != map.end()) {
        } else {
            std::cerr << "Unknown handling method: " << str << std::endl;
        }
        return map[str];
    }

/**
     * [功能描述]：打印当前ICP配准方法的参数配置信息
     * 该函数用于输出当前使用的退化检测方法和处理方法，以及相应的阈值参数，
     * 便于调试和分析不同配置下的算法表现。
     * 
     * @param method_name：方法名称，用于标识当前使用的整体配准策略
     * @param detection：退化检测方法枚举值，指定用于检测点云配准退化情况的算法
     * @param handling：退化处理方法枚举值，指定在检测到退化时采用的缓解策略
     * @return 无返回值
     */
    void TestRunner::printCurrentParameters(const std::string &method_name,
                                            DetectionMethod detection,
                                            HandlingMethod handling) {
        // 打印方法名称标题
        std::cout << "\n=== Method: " << method_name << " ===" << std::endl;
        
        // 打印退化检测方法及其参数
        std::cout << "退化检测方法: ";
        switch (detection) {
            case DetectionMethod::NONE_DETE:
                // 不使用任何退化检测
                std::cout << "NONE";
                break;
            case DetectionMethod::SCHUR_CONDITION_NUMBER:
                // 使用Schur补条件数检测，通过条件数阈值判断矩阵病态程度
                std::cout << "SCHUR_CONDITION_NUMBER (threshold=" << config_.icp_params.DEGENERACY_THRES_COND << ")";
                break;
            case DetectionMethod::FULL_EVD_MIN_EIGENVALUE:
                // 使用完整特征值分解，通过最小特征值判断退化
                std::cout << "FULL_EVD_MIN_EIGENVALUE (threshold=" << config_.icp_params.DEGENERACY_THRES_EIG << ")";
                break;
            case DetectionMethod::EVD_SUB_CONDITION:
                // 使用特征值分解的子条件数检测
                std::cout << "EVD_SUB_CONDITION (threshold=" << config_.icp_params.DEGENERACY_THRES_COND << ")";
                break;
            case DetectionMethod::FULL_SVD_CONDITION:
                // 使用完整奇异值分解的条件数检测
                std::cout << "FULL_SVD_CONDITION (threshold=" << config_.icp_params.DEGENERACY_THRES_COND << ")";
                break;
            case DetectionMethod::O3D:
                // Open3D方法，不进行退化检测
                std::cout << "O3D (No detection)";
                break;
            case DetectionMethod::SUPERLOC:
                // SuperLoc方法，基于特征可观测性和协方差分析
                std::cout << "SUPERLOC (Feature Observability + Covariance)";
                break;
            case DetectionMethod::XICP_OPTIMIZED_EQUALITY:
                // XICP优化等式约束方法
                std::cout << "XICP_OPTIMIZED_EQUALITY";
                break;
            case DetectionMethod::XICP_INEQUALITY:
                // XICP不等式约束方法
                std::cout << "XICP_INEQUALITY";
                break;
            case DetectionMethod::XICP_EQUALITY:
                // XICP等式约束方法
                std::cout << "XICP_EQUALITY";
                break;
            case DetectionMethod::XICP_SOLUTION_REMAPPING:
                // XICP解重映射方法
                std::cout << "XICP_SOLUTION_REMAPPING";
                break;
        }
        std::cout << std::endl;

        // 打印退化处理方法及其参数
        std::cout << "退化处理方法: ";
        switch (handling) {
            case HandlingMethod::NONE_HAND:
                // 不使用任何退化处理策略
                std::cout << "NONE";
                break;
            case HandlingMethod::STANDARD_REGULARIZATION:
                // 标准正则化方法，添加固定正则化项gamma * I
                std::cout << "STANDARD_REGULARIZATION (gamma=" << config_.icp_params.STD_REG_GAMMA << ")";
                break;
            case HandlingMethod::ADAPTIVE_REGULARIZATION:
                // 自适应正则化方法，根据矩阵条件动态调整正则化强度
                std::cout << "ADAPTIVE_REGULARIZATION (alpha=" << config_.icp_params.ADAPTIVE_REG_ALPHA << ")";
                break;
            case HandlingMethod::PRECONDITIONED_CG:
                // 预条件共轭梯度法，通过预条件器改善条件数kappa
                std::cout << "PRECONDITIONED_CG (kappa=" << config_.icp_params.KAPPA_TARGET << ", tol="
                          << config_.icp_params.PCG_TOLERANCE << ")";
                break;
            case HandlingMethod::SOLUTION_REMAPPING:
                // 解重映射方法，仅在可靠特征方向上求解（类似LOAM）
                std::cout << "SOLUTION_REMAPPING (eigen_thresh=" << config_.icp_params.LOAM_EIGEN_THRESH << ")";
                break;
            case HandlingMethod::TRUNCATED_SVD:
                // 截断奇异值分解，去除小奇异值对应的分量
                std::cout << "TRUNCATED_SVD (singular_thresh=" << config_.icp_params.TSVD_SINGULAR_THRESH << ")";
                break;
            case HandlingMethod::O3D:
                // Open3D方法，不进行特殊的退化缓解处理
                std::cout << "O3D (No Mitigation)";
                break;
            case HandlingMethod::SUPERLOC:
                // SuperLoc方法，使用基于Ceres的SE(3)流形优化
                std::cout << "SUPERLOC (Ceres-based SE3 optimization)";
                break;
            case HandlingMethod::XICP_CONSTRAINT:
                // XICP约束方法，在优化中加入约束条件
                std::cout << "XICP_CONSTRAINT";
                break;
            case HandlingMethod::XICP_PROJECTION:
                // XICP投影方法，将解投影到可靠子空间
                std::cout << "XICP_PROJECTION";
                break;
        }
        std::cout << std::endl;
        
        // 打印结束分隔线
        std::cout << "============================" << std::endl;
    }

/**
 * [功能描述]：执行所有配置的ICP测试方法，进行性能测试和算法比较。
 * @return bool：所有测试成功执行返回true，任何步骤失败返回false
 */
bool TestRunner::runAllTests() {
    // 首先加载点云数据，只需加载一次供所有测试方法使用
    // 如果点云加载失败，直接返回false终止所有测试
    if (!loadPointClouds()) {
        return false;
    }

    // 遍历配置文件中定义的所有测试方法
    // config_.test_methods是一个map，包含方法名和对应的检测/处理方法对
    for (const auto&[method_name, method_pair] : config_.test_methods) {
        // 将字符串格式的检测方法转换为枚举类型
        // method_pair.first存储检测方法名称（如"NONE_DETE", "FCN_SR"等）
        auto detection = stringToDetectionMethod(method_pair.first);
        // 将字符串格式的处理方法转换为枚举类型  
        // method_pair.second存储处理方法名称（如"NONE_HAND", "ME_SR"等）
        auto handling = stringToHandlingMethod(method_pair.second);

        // 打印当前测试方法的标题信息
        std::cout << "\n--- Testing method: " << method_name << " ---" << std::endl;
        // 打印当前方法的详细参数配置，便于结果分析和调试
        printCurrentParameters(method_name, detection, handling);

        // 执行具体的测试方法，包含多次运行以获得统计数据
        // runMethod会根据config_.num_runs参数重复运行指定次数
        if (!runMethod(method_name, detection, handling)) {
            std::cerr << "Failed to run method: " << method_name << std::endl;
            return false;  // 任何方法失败都会终止整个测试流程
        }
    }

    // 完成所有方法测试后，整理和计算最终统计数据
    // 包括平均值、标准差、成功率等统计指标
    finalizeStatistics();

    // 保存统计摘要数据到文件（如CSV格式的汇总结果）
    saveStatistics();
    // 保存详细的测试结果数据（如每次运行的具体数据）
    saveDetailedResults();

    return true;  // 所有测试方法成功完成
}

/**
 * [功能描述]：对指定的ICP方法进行多次运行测试，收集统计数据并保存结果。
 * @param method_name：测试方法的名称（如"FCN-SR", "ME-TReg"等）
 * @param detection：退化检测方法的枚举类型
 * @param handling：退化处理方法的枚举类型
 * @return bool：测试成功完成返回true，失败返回false
 */
bool TestRunner::runMethod(const std::string &method_name,
                           DetectionMethod detection,
                           HandlingMethod handling) {
    // 为当前测试方法初始化统计数据结构
    // statistics_是一个map，存储每个方法的统计信息
    statistics_[method_name] = MethodStatistics();
    statistics_[method_name].method_name = method_name;

    // 根据配置进行多次运行测试以获得统计学上有效的数据
    // config_.num_runs定义了每个方法的运行次数
    for (int run = 0; run < config_.num_runs; ++run) {
        // 如果需要多次运行，每10次显示一次进度信息
        // 避免输出过多信息影响可读性
        if (config_.num_runs > 1 && run % 10 == 0) {
            std::cout << "  Run " << run + 1 << "/" << config_.num_runs << std::endl;
        }

        // 执行单次ICP测试，返回包含配准结果的TestResult对象
        // runSingleTest是核心测试函数，执行具体的ICP算法
        TestResult result = runSingleTest(method_name, detection, handling);

        // 将单次测试结果保存到详细结果数组中
        // detailed_results_存储每次运行的完整数据，用于后续分析
        detailed_results_[method_name].push_back(result);
        // 更新该方法的累积统计信息（平均值、标准差等）
        updateStatistics(method_name, result);

        // 仅在第一次运行时保存点云文件和可视化结果，避免重复保存
        // 检查配置是否需要保存PCD文件、错误可视化或交互式可视化
        if (run == 0 && (config_.save_pcd || config_.save_error_pcd || config_.visualize)) {
            // 使用ICP配准得到的最终变换矩阵对源点云进行变换
            // 生成配准后的对齐点云，用于可视化和保存
            pcl::PointCloud<PointT>::Ptr aligned_cloud(new pcl::PointCloud <PointT>);
            pcl::transformPointCloud(*source_cloud_, *aligned_cloud, result.final_transform);

            // 获取初始噪声位姿，用于生成初始状态点云
            Pose6D initial_pose = config_.initial_noise;
            pcl::PointCloud<PointT>::Ptr initial_cloud_(new pcl::PointCloud<PointT>());
            // 使用配置的初始变换矩阵生成初始位置的点云
            pcl::transformPointCloud(*source_cloud_, *initial_cloud_, config_.initial_matrix);

            // 如果配置要求保存点云文件，则保存各种状态的点云数据
            if (config_.save_pcd) {
                // 保存带颜色的对齐点云可视化文件（源点云红色，目标点云绿色）
                std::string aligned_filename = config_.output_folder + method_name + "_aligned_clouds.pcd";
                saveAlignedClouds(aligned_cloud, target_cloud_, aligned_filename);
                std::cout << "Saved aligned clouds for " << method_name << " to " << aligned_filename << std::endl;

                // 保存单独的配准后点云文件（二进制格式，文件更小）
                std::string aligned_filename_single =
                        config_.output_folder + method_name + "_aligned_clouds_sig.pcd";
                pcl::io::savePCDFileBinary(aligned_filename_single, *aligned_cloud);
                
                // 保存初始状态的点云文件，用于对比分析
                std::string initial_filename = config_.output_folder + "initial_clouds.pcd";
                pcl::io::savePCDFileBinary(initial_filename, *initial_cloud_);
                
                // 保存原始源点云文件（注意：这里保存的是source_cloud_作为target）
                std::string target_filename = config_.output_folder + "target_clouds.pcd";
                pcl::io::savePCDFileBinary(target_filename, *source_cloud_);
            }
            
            // 如果配置要求保存错误可视化，生成基于配准误差的彩色点云
            // 使用jet色彩映射显示每个点的配准误差大小
            if (config_.save_error_pcd) {
                std::string error_filename = config_.output_folder + method_name + "_error.pcd";
                saveErrorPointCloud(aligned_cloud, target_cloud_, error_filename);
                std::cout << "Saved error visualization for " << method_name << " to " << error_filename
                          << std::endl;
            }
            
            // 如果配置要求交互式可视化，启动3D可视化窗口
            // 用户可以交互式地查看配准结果
            if (run == 0 && config_.visualize) {
                visualizeResults(aligned_cloud, target_cloud_, method_name, result);
            }
        }
    }

    return true;  // 所有运行完成，测试成功
}

/**
 * [功能描述]：执行单次ICP配准测试，根据方法名称调用相应的ICP算法实现。
 * @param method_name：ICP方法名称（如"O3D", "SuperLoc", "XICP", "ME-SR"等）
 * @param detection：退化检测方法枚举类型
 * @param handling：退化处理方法枚举类型
 * @return TestResult：包含配准结果、性能指标和详细数据的测试结果对象
 */
TestResult TestRunner::runSingleTest(const std::string &method_name,
                                     DetectionMethod detection,
                                     HandlingMethod handling) {
    // 初始化测试结果对象，存储本次测试的所有输出数据
    TestResult result;
    result.method_name = method_name;

    // 设置初始位姿：从配置文件读取的初始噪声作为ICP的起始位姿
    // initial_pose使用6D表示（3个平移 + 3个欧拉角）
    Pose6D initial_pose = config_.initial_noise;
    // initial_state使用SE3群表示（旋转矩阵 + 平移向量），用于SO3参数化的ICP
    MathUtils::SE3State initial_state(
            config_.initial_matrix.block<3, 3>(0, 0),  // 提取旋转矩阵部分
            config_.initial_matrix.block<3, 1>(0, 3)   // 提取平移向量部分
    );
    MathUtils::SE3State optimized_state;  // 存储优化后的SE3状态

    // 创建ICP上下文对象，用于存储配准过程中的中间数据和参数
    // 计算目标点云的法向量，法向量对point-to-plane ICP至关重要
    ICPContext context;
    context.setTargetCloud(target_cloud_, config_.normal_nn);  // normal_nn是计算法向量的近邻点数

    // 根据方法名称选择相应的ICP算法实现
    if (method_name == "O3D") {
        // Open3D库的ICP实现：使用第三方库提供的标准ICP算法
        // 调用后仍进入统一的测试流程进行误差计算和结果分析
        runOpen3DICP(method_name, result);

    } else if (method_name == "SuperLoc") {
        // SuperLoc方法：基于Ceres非线性优化库的SE(3)参数化ICP
        std::cout << "\n[SuperLoc] 开始SuperLoc ICP方法..." << std::endl;
        std::cout << "[SuperLoc] 使用Ceres-based SE(3)优化" << std::endl;
        std::cout << "[SuperLoc] 参数: 最大迭代次数=" << config_.max_iterations
                  << ", 搜索半径=" << config_.search_radius << std::endl;

        // 调用SuperLoc的ICP实现，使用Ceres进行位姿优化
        SuperLocICP::runSuperLocICP(method_name, config_, result, context, source_cloud_, target_cloud_);

    } else if (method_name == "XICP-1" || method_name == "XICP-EQ" || method_name == "XICP-INQ" ||
               method_name == "XICP-4" || method_name == "XICP-OP" || method_name == "XICP") {
        // XICP方法系列：具有退化感知能力的ICP算法
        // 支持多种退化检测和处理策略（等式约束、不等式约束、解映射等）
        auto start = std::chrono::high_resolution_clock::now();  // 开始计时
        
        // 调用SO3参数化的point-to-plane ICP，使用TBB并行加速
        result.converged = Point2PlaneICP_SO3_tbb_XICP(
                source_cloud_, target_cloud_, initial_state,     // 输入点云和初始状态
                config_.search_radius, detection, handling,     // 搜索半径和退化处理策略
                config_.max_iterations, context, result, optimized_state);  // 迭代参数和输出
        
        auto end = std::chrono::high_resolution_clock::now();    // 结束计时
        
        // 记录执行时间（毫秒）和迭代次数
        result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.iterations = context.final_iterations_;
        result.final_transform = optimized_state.matrix();      // 转换SE3状态为变换矩阵
        
    } else if (method_name == "Ours" || method_name == "NONE" ||
               method_name == "ME-SR" || method_name == "FCN-SR" ||
               method_name == "ME-TSVD" || method_name == "ME-TReg") {
        // DCReg论文中的方法系列：包含多种退化检测和处理策略
        // ME-SR: Minimum Eigenvalue - Singular Ratio, FCN-SR: Frobenius Condition Number - Singular Ratio
        // ME-TSVD: Truncated SVD, ME-TReg: Tikhonov Regularization
        
        auto start = std::chrono::high_resolution_clock::now();  // 开始计时
        
        if (config_.use_so3_parameterization) {
            // 使用SO(3)李群参数化：旋转用流形表示，避免奇异性
            // OpenMP并行加速的point-to-plane ICP实现
            result.converged = Point2PlaneICP_SO3_OpenMP(
                    source_cloud_, target_cloud_, initial_state,     // 输入数据
                    config_.search_radius, detection, handling,     // 搜索和处理参数
                    config_.max_iterations, context, result, optimized_state);  // 输出参数
            result.final_transform = optimized_state.matrix();
        } else {
            // 使用欧拉角参数化：传统的6D位姿表示方法
            // 兼容原始LOAM实现，保持与基线方法的一致性
            Pose6D optimized_pose;
            result.converged = Point2PlaneICP(
                    source_cloud_, target_cloud_, initial_pose,      // 使用6D位姿表示
                    config_.search_radius, detection, handling,     // 配准参数
                    config_.max_iterations, context, result, optimized_pose    // 输出结果
            );
            result.final_transform = Pose6D2Matrix(optimized_pose);  // 转换为4x4变换矩阵
        }
        auto end = std::chrono::high_resolution_clock::now();    // 结束计时
        
        // 记录性能指标
        result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.iterations = context.final_iterations_;
        
    } else {
        // 未识别的方法名称：输出错误信息并返回空结果
        std::cout << "Can not recognize the method!!!!! pls check your yaml!!!" << std::endl;
        return result;
    }

    // === 后处理：保存迭代数据和计算误差指标 ===
    
    // 保存完整的迭代过程数据，用于详细分析和可视化
    // iteration_log_data_包含每次迭代的RMSE、fitness、变换矩阵等信息
    result.iteration_data = context.iteration_log_data_;

    // 提取并保存迭代历史数据，用于收敛性分析和性能评估
    if (!context.iteration_log_data_.empty()) {
        // 遍历每次迭代的数据，构建历史记录数组
        for (const auto &iter_data : context.iteration_log_data_) {
            result.iter_rmse_history.push_back(iter_data.rmse);                    // RMSE收敛曲线
            result.iter_fitness_history.push_back(iter_data.fitness);              // 匹配率收敛曲线  
            result.iter_corr_num_history.push_back(iter_data.corr_num);            // 对应点数量变化
            result.iter_transform_history.push_back(iter_data.transform_matrix);   // 变换矩阵演进
            result.iter_trans_error_history.push_back(iter_data.trans_error_vs_gt); // 平移误差历史
            result.iter_rot_error_history.push_back(iter_data.rot_error_vs_gt);    // 旋转误差历史
        }

        // 从最后一次迭代中提取最终结果和退化信息
        const auto &last_iter = context.iteration_log_data_.back();
        result.final_rmse = last_iter.rmse;                    // 最终配准精度
        result.final_transform = last_iter.transform_matrix;   // 最终变换矩阵
        result.final_fitness = last_iter.fitness;             // 最终匹配率
        result.corr_num = last_iter.corr_num;                 // 最终对应点数量
        
        // 保存条件数信息，用于退化分析
        // cond_schur_rot/trans: Schur complement的条件数, cond_full_svd: 完整矩阵的SVD条件数
        result.condition_numbers = {last_iter.cond_schur_rot, last_iter.cond_schur_trans, last_iter.cond_full_svd};
        
        // 保存Hessian矩阵的特征值，用于退化检测和分析
        result.eigenvalues.resize(6);  // 6DOF位姿对应6个特征值
        for (int i = 0; i < 6; ++i) {
            result.eigenvalues[i] = last_iter.eigenvalues_full(i);
        }
        result.degenerate_mask = last_iter.degenerate_mask;   // 退化维度掩码
    }

    // 计算相对于真值的位姿误差（假设真值为单位矩阵，即完美对齐）
    PoseError error = calculatePoseError(config_.gt_matrix, result.final_transform, true);
    result.trans_error_m = error.translation_error;          // 平移误差（米）
    result.rot_error_deg = error.rotation_error;            // 旋转误差（度）

    // 计算point-to-point几何误差指标
    // 将源点云用最终变换矩阵进行变换，得到配准后的点云
    pcl::PointCloud<PointT>::Ptr aligned_cloud(new pcl::PointCloud <PointT>);
    pcl::transformPointCloud(*source_cloud_, *aligned_cloud, result.final_transform);
    
    // 计算配准后点云与目标点云之间的几何误差
    calculatePointToPointError(aligned_cloud, target_cloud_,
                               result.p2p_rmse,           // Point-to-Point RMSE
                               result.p2p_fitness,        // 匹配点的比例
                               result.chamfer_distance,   // Chamfer距离（双向最近点距离）
                               result.corr_num,           // 有效对应点数量
                               config_.error_threshold);   // 误差阈值

    // 输出本次测试的关键结果信息
    error.printPoseError();  // 打印位姿误差详细信息
    std::cout << "P2P RMSE: " << result.p2p_rmse << ", Chamfer: " << result.chamfer_distance << std::endl;

    return result;  // 返回完整的测试结果
}


    // Save aligned clouds with different colors
    void TestRunner::saveAlignedClouds(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                                       const pcl::PointCloud<PointT>::Ptr &target_cloud,
                                       const std::string &filename) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud <pcl::PointXYZRGB>);

        // Add aligned cloud in red
        for (const auto &point : aligned_cloud->points) {
            pcl::PointXYZRGB colored_point;
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;
            colored_point.r = 245;
            colored_point.g = 121;
            colored_point.b = 0;
            combined_cloud->push_back(colored_point);
        }

        // Add target cloud in green
        for (const auto &point : target_cloud->points) {
            pcl::PointXYZRGB colored_point;
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;
            colored_point.r = 144;
            colored_point.g = 159;
            colored_point.b = 207;
            combined_cloud->push_back(colored_point);
        }

        combined_cloud->width = combined_cloud->points.size();
        combined_cloud->height = 1;
        combined_cloud->is_dense = false;

        pcl::io::savePCDFileBinary(filename, *combined_cloud);
        //        pcl::io::savePCDFileBinary(filename, *target_cloud);
    }

    // Save error visualization point cloud
    void TestRunner::saveErrorPointCloud(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                                         const pcl::PointCloud<PointT>::Ptr &target_cloud,
                                         const std::string &filename) {
        auto error_cloud = createErrorPointCloud(aligned_cloud, target_cloud, config_.error_threshold);
        pcl::io::savePCDFileBinary(filename, *error_cloud);
    }

    // 改进的错误点云生成（使用jet colormap）
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr TestRunner::createErrorPointCloud(
            const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
            const pcl::PointCloud<PointT>::Ptr &target_cloud,
            double max_error_threshold) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr error_cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
        error_cloud->reserve(aligned_cloud->points.size());
        pcl::KdTreeFLANN <PointT> kdtree;
        kdtree.setInputCloud(target_cloud);
        // 计算所有误差以确定实际的最大误差
        std::vector<double> errors;
        errors.reserve(aligned_cloud->points.size());
        for (const auto &point : aligned_cloud->points) {
            std::vector<int> indices(1);
            std::vector<float> sq_distances(1);
            if (kdtree.nearestKSearch(point, 1, indices, sq_distances) > 0) {
                double error = std::sqrt(sq_distances[0]);
                errors.push_back(error);
            }
        }
        // 使用实际最大误差或设定阈值中的较小者
        double actual_max_error = *std::max_element(errors.begin(), errors.end());
        double color_max_threshold = std::min(max_error_threshold, actual_max_error);
        // 生成带jet colormap的错误点云
        for (size_t i = 0; i < aligned_cloud->points.size() && i < errors.size(); ++i) {
            pcl::PointXYZRGB colored_point = getJetColorForError(errors[i], color_max_threshold);
            colored_point.x = aligned_cloud->points[i].x;
            colored_point.y = aligned_cloud->points[i].y;
            colored_point.z = aligned_cloud->points[i].z;
            error_cloud->points.push_back(colored_point);
        }
        error_cloud->width = error_cloud->points.size();
        error_cloud->height = 1;
        error_cloud->is_dense = false;
        return error_cloud;
    }


    // Update statistics with a new result
    void TestRunner::updateStatistics(const std::string &method_name, const TestResult &result) {
        auto &stats = statistics_[method_name];
        stats.total_runs++;

        if (result.converged) {
            stats.converged_runs++;
        }

        // Update sums for mean calculation
        stats.mean_trans_error += result.trans_error_m;
        stats.mean_rot_error += result.rot_error_deg;
        stats.mean_time_ms += result.time_ms;
        stats.mean_iterations += result.iterations;
        stats.mean_rmse += result.final_rmse;
        stats.mean_fitness += result.final_fitness;
        stats.mean_p2p_rmse += result.p2p_rmse;
        stats.mean_p2p_fitness += result.p2p_fitness;
        stats.mean_chamfer += result.chamfer_distance;
        stats.corr_num += result.corr_num;

        // Update min/max
        stats.min_trans_error = std::min(stats.min_trans_error, result.trans_error_m);
        stats.max_trans_error = std::max(stats.max_trans_error, result.trans_error_m);
        stats.min_rot_error = std::min(stats.min_rot_error, result.rot_error_deg);
        stats.max_rot_error = std::max(stats.max_rot_error, result.rot_error_deg);
    }

// Finalize statistics (calculate means and standard deviations)
    void TestRunner::finalizeStatistics() {
        for (auto&[method_name, stats] : statistics_) {
            if (stats.total_runs == 0) continue;

            // Calculate means
            stats.mean_trans_error /= stats.total_runs;
            stats.mean_rot_error /= stats.total_runs;
            stats.mean_time_ms /= stats.total_runs;
            stats.mean_iterations /= stats.total_runs;
            stats.mean_rmse /= stats.total_runs;
            stats.mean_fitness /= stats.total_runs;
            stats.mean_p2p_rmse /= stats.total_runs;
            stats.mean_p2p_fitness /= stats.total_runs;
            stats.mean_chamfer /= stats.total_runs;

            // Calculate success rate
            stats.success_rate = static_cast<double>(stats.converged_runs) / stats.total_runs;

            // Calculate standard deviations
            const auto &results = detailed_results_[method_name];
            double sum_sq_trans = 0.0, sum_sq_rot = 0.0, sum_sq_time = 0.0;

            for (const auto &result : results) {
                sum_sq_trans += std::pow(result.trans_error_m - stats.mean_trans_error, 2);
                sum_sq_rot += std::pow(result.rot_error_deg - stats.mean_rot_error, 2);
                sum_sq_time += std::pow(result.time_ms - stats.mean_time_ms, 2);
            }

            stats.std_trans_error = std::sqrt(sum_sq_trans / stats.total_runs);
            stats.std_rot_error = std::sqrt(sum_sq_rot / stats.total_runs);
            stats.std_time_ms = std::sqrt(sum_sq_time / stats.total_runs);
        }
    }

// 修正 saveStatistics 函数
    void TestRunner::saveStatistics() {
        std::string filename = config_.output_folder + "statistics_summary.txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open statistics file: " << filename << std::endl;
            return;
        }
        file << "ICP Test Statistics Summary\n";
        file << "===========================\n\n";
        file << "Configuration:\n";
        file << "  Source: " << config_.source_pcd << "\n";
        file << "  Target: " << config_.target_pcd << "\n";
        file << "  Cloud size: " << source_cloud_->size() << " " << target_cloud_->size() << "\n";
        file << "  Runs per method: " << config_.num_runs << "\n";
//        file << "  ICP Engine: ";
//        switch (config_.icp_engine) {
//            case ICPEngine::CUSTOM_EULER:
//                file << "Custom Euler\n";
//                break;
//            case ICPEngine::CUSTOM_SO3:
//                file << "Custom SO(3)\n";
//                break;
//            case ICPEngine::OPEN3D:
//                file << "Open3D\n";
//                break;
//        }
        file << "\n";
        file << std::fixed << std::setprecision(6);
        // 修正后的表格头，增加迭代次数和ICP残差
        file << std::setw(15) << "Method"
             << std::setw(12) << "Success%"
             << std::setw(12) << "Trans(m)"
             << std::setw(12) << "Rot(deg)"
             << std::setw(12) << "ICP_RMSE"   // 新增
             << std::setw(12) << "Avg_Iters"  // 新增
             << std::setw(12) << "P2PDis"
             << std::setw(12) << "ChamferDis"
             << std::setw(12) << "P2P_Fit%"
             << std::setw(12) << "P2P_Corr"
             << std::setw(12) << "Time(ms)\n";

        file << std::string(135, '-') << "\n";  // 调整分隔线长度
        // 确保表格数据与下方详细数据一致
        for (const auto&[method_name, stats] : statistics_) {
            file << std::setw(15) << method_name
                 << std::setw(12) << std::fixed << std::setprecision(1) << (stats.success_rate * 100)
                 << std::setw(12) << std::fixed << std::setprecision(4) << stats.mean_trans_error
                 << std::setw(12) << std::fixed << std::setprecision(4) << stats.mean_rot_error
                 << std::setw(12) << std::fixed << std::setprecision(4) << stats.mean_rmse        // 新增
                 << std::setw(12) << std::fixed << std::setprecision(1) << stats.mean_iterations  // 新增
                 << std::setw(12) << std::fixed << std::setprecision(4) << stats.mean_p2p_rmse
                 << std::setw(12) << std::fixed << std::setprecision(4) << stats.mean_chamfer
                 << std::setw(12) << std::fixed << std::setprecision(2) << (stats.mean_p2p_fitness * 100)
                 << std::setw(12) << std::fixed << std::setprecision(1) << (stats.corr_num)
                 << std::setw(12) << std::fixed << std::setprecision(2) << stats.mean_time_ms
                 << "\n";
        }
        file << "\n\nDetailed Statistics:\n";
        file << "===================\n\n";
        // 确保详细统计与表格数据一致
        for (const auto&[method_name, stats] : statistics_) {
            file << "Method: " << method_name << "\n";
            file << "  Converged: " << stats.converged_runs << "/" << stats.total_runs
                 << " (Success Rate: " << std::fixed << std::setprecision(1) << (stats.success_rate * 100)
                 << "%)\n";
            file << "  Iterations: " << std::fixed << std::setprecision(1) << stats.mean_iterations << "\n";
            file << "  Translation Error (m): " << std::fixed << std::setprecision(6) << stats.mean_trans_error
                 << " ± " << stats.std_trans_error
                 << " [" << stats.min_trans_error << ", " << stats.max_trans_error << "]\n";
            file << "  Rotation Error (deg): " << std::fixed << std::setprecision(6) << stats.mean_rot_error
                 << " ± " << stats.std_rot_error
                 << " [" << stats.min_rot_error << ", " << stats.max_rot_error << "]\n";
            file << "  Time (ms): " << std::fixed << std::setprecision(2) << stats.mean_time_ms
                 << " ± " << stats.std_time_ms << "\n";
            file << "  ICP RMSE: " << std::fixed << std::setprecision(6) << stats.mean_rmse << "\n";
            file << "  ICP Fitness: " << std::fixed << std::setprecision(4) << stats.mean_fitness << "\n";
            file << "  ICP Correspondence: " << std::fixed << std::setprecision(4) << stats.corr_num << "\n";
            file << "  Point-to-Point RMSE: " << std::fixed << std::setprecision(6) << stats.mean_p2p_rmse << "\n";
            file << "  Point-to-Point Fitness: " << std::fixed << std::setprecision(4) << stats.mean_p2p_fitness
                 << "\n";
            file << "  Chamfer Distance: " << std::fixed << std::setprecision(6) << stats.mean_chamfer << "\n\n";
        }
        file.close();
        std::cout << "Statistics saved to: " << filename << std::endl;


        // Save complete log file
        std::string log_filename = config_.output_folder + "complete_log.txt";
        std::ofstream log_file(log_filename);
        if (log_file.is_open()) {
            log_file << std::fixed << std::setprecision(6);
            log_file << "Complete ICP Test Log\n";
            log_file << "====================\n\n";
            log_file << "Configuration:\n";
            log_file << "  Source: " << config_.source_pcd << "\n";
            log_file << "  Target: " << config_.target_pcd << "\n";
            log_file << "  Runs: " << config_.num_runs << "\n";
            log_file << "  Initial noise: x=" << config_.initial_noise.x
                     << ", y=" << config_.initial_noise.y
                     << ", z=" << config_.initial_noise.z
                     << ", roll=" << pcl::rad2deg(config_.initial_noise.roll)
                     << ", pitch=" << pcl::rad2deg(config_.initial_noise.pitch)
                     << ", yaw=" << pcl::rad2deg(config_.initial_noise.yaw) << " deg\n\n";

            log_file << "ICP Parameters:\n";
            log_file << "  DEGENERACY_THRES_COND: " << config_.icp_params.DEGENERACY_THRES_COND << "\n";
            log_file << "  DEGENERACY_THRES_EIG: " << config_.icp_params.DEGENERACY_THRES_EIG << "\n";
            log_file << "  STD_REG_GAMMA: " << config_.icp_params.STD_REG_GAMMA << "\n";
            log_file << "  ADAPTIVE_REG_ALPHA: " << config_.icp_params.ADAPTIVE_REG_ALPHA << "\n";
            log_file << "  KAPPA_TARGET: " << config_.icp_params.KAPPA_TARGET << "\n";
            log_file << "  PCG_TOLERANCE: " << config_.icp_params.PCG_TOLERANCE << "\n";
            log_file << "  PCG_MAX_ITER: " << config_.icp_params.PCG_MAX_ITER << "\n\n";

            // Copy statistics summary
            log_file << "Results Summary:\n";
            log_file << "================\n";
            for (const auto&[method_name, stats] : statistics_) {
                log_file << "\nMethod: " << method_name << "\n";
                log_file << "  Success rate: " << (stats.success_rate * 100) << "%\n";
                log_file << "  Trans error: " << stats.mean_trans_error << " ± " << stats.std_trans_error << " m\n";
                log_file << "  Rot error: " << stats.mean_rot_error << " ± " << stats.std_rot_error << " deg\n";
                log_file << "  P2P RMSE: " << stats.mean_p2p_rmse << " m\n";
                log_file << "  Chamfer: " << stats.mean_chamfer << " m\n";
                log_file << "  Time: " << stats.mean_time_ms << " ± " << stats.std_time_ms << " ms\n";
            }

            log_file.close();
            std::cout << "Complete log saved to: " << log_filename << std::endl;
        }
    }

    // 在 saveDetailedResults 函数中添加更详细的日志
    void TestRunner::saveDetailedResults() {
        // 保存每种方法的详细transform和分析数据
        std::string transform_log = config_.output_folder + "transform_details.csv";
        std::ofstream tf_file(transform_log);
        if (tf_file.is_open()) {
            tf_file
                    << "Method,Run,Converged,Iterations,Time_ms,Trans_Error_m,Rot_Error_deg,Final_RMSE,Final_Fitness,Corr_Number,";
            tf_file << "Transform_00,Transform_01,Transform_02,Transform_03,";
            tf_file << "Transform_10,Transform_11,Transform_12,Transform_13,";
            tf_file << "Transform_20,Transform_21,Transform_22,Transform_23,";
            tf_file << "Transform_30,Transform_31,Transform_32,Transform_33,";
            tf_file << "SVD_Sigma_0,SVD_Sigma_1,SVD_Sigma_2,SVD_Sigma_3,SVD_Sigma_4,SVD_Sigma_5,";
            tf_file << "EVD_Lambda_0,EVD_Lambda_1,EVD_Lambda_2,EVD_Lambda_3,EVD_Lambda_4,EVD_Lambda_5,";
            tf_file << "Schur_Rot_Lambda_0,Schur_Rot_Lambda_1,Schur_Rot_Lambda_2,";
            tf_file << "Schur_Trans_Lambda_0,Schur_Trans_Lambda_1,Schur_Trans_Lambda_2,";
            tf_file << "Cond_Full_SVD,Cond_Sub_Rot,Cond_Sub_Trans,Cond_Schur_Rot,Cond_Schur_Trans,";
            tf_file
                    << "Degenerate_Mask_0,Degenerate_Mask_1,Degenerate_Mask_2,Degenerate_Mask_3,Degenerate_Mask_4,Degenerate_Mask_5";

            // SuperLoc特有字段
            tf_file << "SuperLoc_Has_Data,SuperLoc_Uncertainty_X,SuperLoc_Uncertainty_Y,SuperLoc_Uncertainty_Z,";
            tf_file << "SuperLoc_Uncertainty_Roll,SuperLoc_Uncertainty_Pitch,SuperLoc_Uncertainty_Yaw,";
            tf_file << "SuperLoc_Cond_Full,SuperLoc_Cond_Rot,SuperLoc_Cond_Trans,SuperLoc_Is_Degenerate\n";


            for (const auto&[method_name, results] : detailed_results_) {
                int run = 0;
                for (const auto &result : results) {
                    tf_file << method_name << "," << run++ << "," << (result.converged ? 1 : 0) << ","
                            << result.iterations << "," << result.time_ms << ","
                            << result.trans_error_m << "," << result.rot_error_deg << ","
                            << result.final_rmse << "," << result.final_fitness << "," << result.corr_num << ",";
                    // Transform matrix (4x4)
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            tf_file << result.final_transform(i, j);
                            if (i < 3 || j < 3) tf_file << ",";
                        }
                    }
                    // 从最后一次迭代获取分析数据
                    if (!result.eigenvalues.empty() && result.eigenvalues.size() >= 6) {
                        // SVD奇异值 (假设存在singular_values数据)
                        for (int i = 0; i < 6; ++i) {
                            tf_file << (i < result.eigenvalues.size() ? result.eigenvalues[i] : 0.0) << ",";
                        }
                        // EVD特征值
                        for (int i = 0; i < 6; ++i) {
                            tf_file << (i < result.eigenvalues.size() ? result.eigenvalues[i] : 0.0) << ",";
                        }
                    } else {
                        // 填充空值
                        for (int i = 0; i < 12; ++i) tf_file << "0.0,";
                    }
                    // Schur特征值 (需要从condition_numbers获取)
                    for (int i = 0; i < 6; ++i) {
                        tf_file << "0.0,"; // 这里需要从iteration log中获取具体数据
                    }
                    // 条件数
                    for (int i = 0; i < result.condition_numbers.size() && i < 5; ++i) {
                        tf_file << result.condition_numbers[i] << ",";
                    }
                    // 填充不足的条件数
                    for (int i = result.condition_numbers.size(); i < 5; ++i) {
                        tf_file << "0.0,";
                    }
                    // 退化掩码
                    for (int i = 0; i < 6; ++i) {
                        tf_file << (i < result.degenerate_mask.size() ? (result.degenerate_mask[i] ? 1 : 0) : 0);
                        if (i < 5) tf_file << ",";
                    }

                    // SuperLoc特有数据
                    if (result.superloc_data.has_data) {
                        tf_file << "1,"
                                << result.superloc_data.uncertainty_x << ","
                                << result.superloc_data.uncertainty_y << ","
                                << result.superloc_data.uncertainty_z << ","
                                << result.superloc_data.uncertainty_roll << ","
                                << result.superloc_data.uncertainty_pitch << ","
                                << result.superloc_data.uncertainty_yaw << ","
                                << result.superloc_data.cond_full << ","
                                << result.superloc_data.cond_rot << ","
                                << result.superloc_data.cond_trans << ","
                                << (result.superloc_data.is_degenerate ? 1 : 0);
                    } else {
                        tf_file << "0,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0";
                    }

                    tf_file << "\n";
                }
            }
            tf_file.close();
            std::cout << "Transform details saved to: " << transform_log << std::endl;
        }

        // 保存每次迭代的详细条件数数据
        if (config_.num_runs == 1) {
            std::string cond_log = config_.output_folder + "condition_numbers_detailed.csv";
            std::ofstream cond_file(cond_log);
            if (cond_file.is_open()) {
                cond_file << "Method,Iteration,Effective_Points,RMSE,Fitness,";
                cond_file << "Cond_Schur_Rot,Cond_Schur_Trans,Cond_Diag_Rot,Cond_Diag_Trans,";
                cond_file << "Cond_Full_EVD_Sub_Rot,Cond_Full_EVD_Sub_Trans,Cond_Full_SVD,";
                cond_file << "Lambda_Schur_Rot_0,Lambda_Schur_Rot_1,Lambda_Schur_Rot_2,";
                cond_file << "Lambda_Schur_Trans_0,Lambda_Schur_Trans_1,Lambda_Schur_Trans_2,";
                cond_file << "Eigenvalues_Full_0,Eigenvalues_Full_1,Eigenvalues_Full_2,";
                cond_file << "Eigenvalues_Full_3,Eigenvalues_Full_4,Eigenvalues_Full_5,";
                cond_file << "Singular_Values_0,Singular_Values_1,Singular_Values_2,";
                cond_file << "Singular_Values_3,Singular_Values_4,Singular_Values_5,";
                cond_file << "Is_Degenerate,Degenerate_Mask_0,Degenerate_Mask_1,Degenerate_Mask_2,";
                cond_file << "Degenerate_Mask_3,Degenerate_Mask_4,Degenerate_Mask_5\n";

                // Write data for each method
                for (const auto&[method_name, results] : detailed_results_) {
                    if (results.empty()) continue;
                    const auto &result = results[0]; // First run only

                    // Write data for each iteration
                    for (const auto &iter_data : result.iteration_data) {
                        cond_file << method_name << ","
                                  << iter_data.iter_count << ","
                                  << iter_data.effective_points << ","
                                  << iter_data.rmse << ","
                                  << iter_data.fitness << ",";

                        // Condition numbers
                        cond_file << iter_data.cond_schur_rot << ","
                                  << iter_data.cond_schur_trans << ","
                                  << iter_data.cond_diag_rot << ","
                                  << iter_data.cond_diag_trans << ","
                                  << iter_data.cond_full_evd_sub_rot << ","
                                  << iter_data.cond_full_evd_sub_trans << ","
                                  << iter_data.cond_full_svd << ",";

                        // Lambda Schur Rot (3 values)
                        for (int i = 0; i < 3; ++i) {
                            if (i < iter_data.lambda_schur_rot.size()) {
                                cond_file << iter_data.lambda_schur_rot(i);
                            } else {
                                cond_file << "NaN";
                            }
                            cond_file << ",";
                        }

                        // Lambda Schur Trans (3 values)
                        for (int i = 0; i < 3; ++i) {
                            if (i < iter_data.lambda_schur_trans.size()) {
                                cond_file << iter_data.lambda_schur_trans(i);
                            } else {
                                cond_file << "NaN";
                            }
                            cond_file << ",";
                        }

                        // Eigenvalues Full (6 values)
                        for (int i = 0; i < 6; ++i) {
                            if (i < iter_data.eigenvalues_full.size()) {
                                cond_file << iter_data.eigenvalues_full(i);
                            } else {
                                cond_file << "NaN";
                            }
                            cond_file << ",";
                        }

                        // Singular Values (6 values)
                        for (int i = 0; i < 6; ++i) {
                            if (i < iter_data.singular_values_full.size()) {
                                cond_file << iter_data.singular_values_full(i);
                            } else {
                                cond_file << "NaN";
                            }
                            cond_file << ",";
                        }

                        // Is Degenerate
                        cond_file << (iter_data.is_degenerate ? 1 : 0) << ",";

                        // Degenerate Mask (6 values)
                        for (int i = 0; i < 6; ++i) {
                            if (i < iter_data.degenerate_mask.size()) {
                                cond_file << (iter_data.degenerate_mask[i] ? 1 : 0);
                            } else {
                                cond_file << "0";
                            }
                            if (i < 5) cond_file << ",";
                        }

                        cond_file << "\n";
                    }
                }

                cond_file.close();
                std::cout << "Condition numbers details saved to: " << cond_log << std::endl;
            }
        }

        // Save all results to CSV for further analysis
        std::string csv_filename = config_.output_folder + "all_results.csv";
        std::ofstream csv(csv_filename);

        if (!csv.is_open()) {
            std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
            return;
        }

        // CSV header
        csv << "Method,Run,Converged,Iterations,Time_ms,Trans_Error_m,Rot_Error_deg,"
            << "ICP_RMSE,ICP_Fitness,P2P_RMSE,P2P_Fitness,Chamfer_Distance\n";

        // Write all results
        for (const auto&[method_name, results] : detailed_results_) {
            int run = 0;
            for (const auto &result : results) {
                csv << method_name << ","
                    << run++ << ","
                    << (result.converged ? 1 : 0) << ","
                    << result.iterations << ","
                    << result.time_ms << ","
                    << result.trans_error_m << ","
                    << result.rot_error_deg << ","
                    << result.final_rmse << ","
                    << result.final_fitness << ","
                    << result.p2p_rmse << ","
                    << result.p2p_fitness << ","
                    << result.chamfer_distance << "\n";
            }
        }

        csv.close();
        std::cout << "Detailed results saved to: " << csv_filename << std::endl;


        if (config_.num_runs == 1) {
            // Save degeneracy analysis for single run - FIRST ITERATION
            std::string filename = config_.output_folder + "degeneracy_analysis_first_iter.txt";
            std::ofstream dn_file(filename);

            if (!dn_file.is_open()) {
                std::cerr << "Failed to open degeneracy analysis file: " << filename << std::endl;
                return;
            }

            dn_file << "Degeneracy Analysis Results (First Iteration)\n";
            dn_file << "============================================\n\n";

            for (const auto&[method_name, results] : detailed_results_) {
                if (results.empty()) continue;
                const auto &result = results[0];

                // 检查是否有迭代数据
                if (result.iteration_data.empty()) {
                    dn_file << "Method: " << method_name << " - No iteration data available\n\n";
                    continue;
                }

                // 获取第一次迭代的数据
                const auto &first_iter = result.iteration_data[0];

                dn_file << "Method: " << method_name << "\n";

                // 条件数
                dn_file << "  Condition Numbers:\n";
                dn_file << "    Schur Rot: " << std::fixed << std::setprecision(2) << first_iter.cond_schur_rot << "\n";
                dn_file << "    Schur Trans: " << std::fixed << std::setprecision(2) << first_iter.cond_schur_trans
                        << "\n";
                dn_file << "    Diag Rot: " << std::fixed << std::setprecision(2) << first_iter.cond_diag_rot << "\n";
                dn_file << "    Diag Trans: " << std::fixed << std::setprecision(2) << first_iter.cond_diag_trans
                        << "\n";
                dn_file << "    SVD Diag Rot: " << std::fixed << std::setprecision(2)
                        << first_iter.cond_full_evd_sub_rot << "\n";
                dn_file << "    SVD Diag Trans: " << std::fixed << std::setprecision(2)
                        << first_iter.cond_full_evd_sub_trans << "\n";
                dn_file << "    Full SVD: " << std::fixed << std::setprecision(2) << first_iter.cond_full_svd << "\n";

                // 特征值
                dn_file << "  Eigenvalues (Full): ";
                for (int i = 0; i < 6; ++i) {
                    if (i < first_iter.eigenvalues_full.size()) {
                        dn_file << std::fixed << std::setprecision(3) << first_iter.eigenvalues_full(i) << " ";
                    }
                }

                // 退化掩码
                dn_file << "\n  Degenerate Mask (wxwywz xyz): ";
                for (bool mask : first_iter.degenerate_mask) {
                    dn_file << (mask ? "1" : "0") << " ";
                }
                dn_file << "\n  Is Degenerate: " << (first_iter.is_degenerate ? "Yes" : "No");
                dn_file << "\n\n";

                dn_file << std::fixed << std::setprecision(6);
                // 对于PCG方法，保存预处理矩阵
                if (method_name.find("PCG") != std::string::npos || method_name == "Ours") {
                    dn_file << "  Preconditioner Matrix P:\n";
                    for (int i = 0; i < 6; ++i) {
                        dn_file << "    ";
                        for (int j = 0; j < 6; ++j) {
                            dn_file << std::setw(12) << first_iter.P_preconditioner(i, j) << " ";
                        }
                        dn_file << "\n";
                    }
                    dn_file << "\n";
                }

                // 对于自适应正则化方法，保存W矩阵
                if (method_name.find("AReg") != std::string::npos) {
                    dn_file << "  Adaptive Regularization Matrix W:\n";
                    for (int i = 0; i < 6; ++i) {
                        dn_file << "    ";
                        for (int j = 0; j < 6; ++j) {
                            dn_file << std::setw(12) << first_iter.W_adaptive(i, j) << " ";
                        }
                        dn_file << "\n";
                    }
                    dn_file << "\n";
                }

                // SuperLoc特有数据 - 第一次迭代
                if (method_name.find("SuperLoc") != std::string::npos) {
                    // SuperLoc使用Ceres优化，可能没有传统的H矩阵分析
                    // 但我们仍然可以保存其特有的不确定性分析
                    if (result.superloc_data.has_data) {
                        dn_file << " SuperLoc Feature Observability Analysis:\n";
                        dn_file << "    Translation Uncertainty (XYZ): ";
                        dn_file << " " << std::fixed << std::setprecision(4)
                                << result.superloc_data.uncertainty_x << " ";
                        dn_file << " " << result.superloc_data.uncertainty_y << " ";
                        dn_file << " " << result.superloc_data.uncertainty_z << "\n";
                        dn_file << "    Rotation Uncertainty (RPY): ";
                        dn_file << " " << result.superloc_data.uncertainty_roll << " ";
                        dn_file << " " << result.superloc_data.uncertainty_pitch << " ";
                        dn_file << " " << result.superloc_data.uncertainty_yaw << "\n";

                        dn_file << "    Feature Histogram: [";
                        for (int i = 0; i < 9; ++i) {
                            dn_file << result.superloc_data.feature_histogram[i];
                            if (i < 8) dn_file << ", ";
                        }
                        dn_file << "]\n\n";
                    }
                }


                // 对于Schur方法，保存对齐分析
                if ((method_name == "Ours" || method_name.find("SCHUR") != std::string::npos) &&
                    first_iter.is_degenerate) {

                    Eigen::Vector3d ref_axes[3];
                    ref_axes[0] << 1.0, 0.0, 0.0;
                    ref_axes[1] << 0.0, 1.0, 0.0;
                    ref_axes[2] << 0.0, 0.0, 1.0;

                    dn_file << "  Alignment Analysis:\n";
                    dn_file << "    Rotation Axes:\n";
                    for (int i = 0; i < 3; ++i) {
                        int idx = first_iter.rot_indices[i];
                        double l = (idx >= 0 && idx < 3) ? first_iter.lambda_schur_rot(idx) : NAN;
                        Eigen::Vector3d v = first_iter.aligned_V_rot.col(i);
                        double dot = std::abs(v.dot(ref_axes[i]));
                        double ang = std::acos(std::min(1.0, std::max(0.0, dot))) * 180.0 / M_PI;
                        double sabs = std::max(1e-9, v.cwiseAbs().sum());
                        double pR = 100 * std::abs(v(0)) / sabs;
                        double pP = 100 * std::abs(v(1)) / sabs;
                        double pY = 100 * std::abs(v(2)) / sabs;

                        dn_file << "      [" << i << "]~" << (i == 0 ? "R" : (i == 1 ? "P" : "Y"))
                                << " (orig_idx=" << idx << "): λ=" << l
                                << ", Angle=" << ang << "°, "
                                << pR << "%R+" << pP << "%P+" << pY << "%Y\n";
                    }

                    dn_file << "    Translation Axes:\n";
                    for (int i = 0; i < 3; ++i) {
                        int idx = first_iter.trans_indices[i];
                        double l = (idx >= 0 && idx < 3) ? first_iter.lambda_schur_trans(idx) : NAN;
                        Eigen::Vector3d v = first_iter.aligned_V_trans.col(i);
                        double dot = std::abs(v.dot(ref_axes[i]));
                        double ang = std::acos(std::min(1.0, std::max(0.0, dot))) * 180.0 / M_PI;
                        double sabs = std::max(1e-9, v.cwiseAbs().sum());
                        double pX = 100 * std::abs(v(0)) / sabs;
                        double pY = 100 * std::abs(v(1)) / sabs;
                        double pZ = 100 * std::abs(v(2)) / sabs;

                        dn_file << "      [" << i << "]~" << (i == 0 ? "X" : (i == 1 ? "Y" : "Z"))
                                << " (orig_idx=" << idx << "): λ=" << l
                                << ", Angle=" << ang << "°, "
                                << pX << "%X+" << pY << "%Y+" << pZ << "%Z\n";
                    }


                    dn_file << " \n";

                }
            }

            dn_file << "\n\n";
            dn_file.close();
            std::cout << "First iteration degeneracy analysis saved to: " << filename << std::endl;
        }

        std::string degeneracy_file = config_.output_folder + "degeneracy_analysis_last_iter.txt";
        std::ofstream deg_file(degeneracy_file);

        if (deg_file.is_open()) {
            deg_file << std::fixed << std::setprecision(6);
            deg_file << "Degeneracy Analysis Results\n";
            deg_file << "==========================\n\n";

            for (const auto&[method_name, results] : detailed_results_) {
                if (results.empty()) continue;
                const auto &result = results[0];

                deg_file << "Method: " << method_name << "\n";

                // 最终变换矩阵
                deg_file << "Final Transform Matrix:\n";
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        deg_file << std::setw(12) << result.final_transform(i, j) << " ";
                    }
                    deg_file << "\n";
                }
                deg_file << "\n";

                // 跳过Open3D方法的退化分析
                if (method_name.find("O3D") != std::string::npos) {
                    deg_file << "  [Open3D method - no degeneracy analysis]\n\n";
                    continue;
                }

                if (!result.iteration_data.empty()) {
                    const auto &last_iter = result.iteration_data.back();

                    deg_file << "  Condition Numbers:\n";
                    deg_file << "    Schur Rot: " << last_iter.cond_schur_rot << "\n";
                    deg_file << "    Schur Trans: " << last_iter.cond_schur_trans << "\n";
                    deg_file << "    Diag Rot: " << last_iter.cond_diag_rot << "\n";
                    deg_file << "    Diag Trans: " << last_iter.cond_diag_trans << "\n";
                    deg_file << "    SVD Diag Rot: " << last_iter.cond_full_evd_sub_rot << "\n";
                    deg_file << "    SVD Diag Trans: " << last_iter.cond_full_evd_sub_trans << "\n";
                    deg_file << "    Full SVD: " << last_iter.cond_full_svd << "\n\n";

                    deg_file << "  EVD Eigenvalues (Full):\n";
                    for (int i = 0; i < 6; ++i) {
                        deg_file << "    λ" << i << ": " << last_iter.eigenvalues_full(i) << "\n";
                    }
                    deg_file << "\n";

                    deg_file << "  SVD Singular Values:\n";
                    for (int i = 0; i < 6; ++i) {
                        deg_file << "    σ" << i << ": " << last_iter.singular_values_full(i) << "\n";
                    }
                    deg_file << "\n";

                    deg_file << "  Diagonal Block Eigenvalues:\n";
                    deg_file << "    Rotation: [" << last_iter.lambda_diag_rot.transpose() << "]\n";
                    deg_file << "    Translation: [" << last_iter.lambda_diag_trans.transpose() << "]\n\n";

                    deg_file << "  Schur Complement Eigenvalues:\n";
                    deg_file << "    Rotation: [" << last_iter.lambda_schur_rot.transpose() << "]\n";
                    deg_file << "    Translation: [" << last_iter.lambda_schur_trans.transpose() << "]\n\n";

                    deg_file << "  Degenerate Mask (ωxωyωz xyz): ";
                    for (bool mask : last_iter.degenerate_mask) {
                        deg_file << (mask ? "1" : "0") << " ";
                    }
                    deg_file << "\n\n";

                    // 对于PCG方法，保存预处理矩阵
                    if (method_name.find("PCG") != std::string::npos || method_name == "Ours") {
                        deg_file << "  Preconditioner Matrix P:\n";
                        for (int i = 0; i < 6; ++i) {
                            deg_file << "    ";
                            for (int j = 0; j < 6; ++j) {
                                deg_file << std::setw(12) << last_iter.P_preconditioner(i, j) << " ";
                            }
                            deg_file << "\n";
                        }
                        deg_file << "\n";
                    }

                    // 对于自适应正则化方法，保存W矩阵
                    if (method_name.find("AReg") != std::string::npos) {
                        deg_file << "  Adaptive Regularization Matrix W:\n";
                        for (int i = 0; i < 6; ++i) {
                            deg_file << "    ";
                            for (int j = 0; j < 6; ++j) {
                                deg_file << std::setw(12) << last_iter.W_adaptive(i, j) << " ";
                            }
                            deg_file << "\n";
                        }
                        deg_file << "\n";
                    }

                    // SuperLoc特有的退化分析
                    if (method_name.find("SuperLoc") != std::string::npos && result.superloc_data.has_data) {
                        deg_file << "  SuperLoc Analysis Results:\n";

                        deg_file << "    Feature Observability Analysis:\n";
                        deg_file << "      Translation Uncertainty: ";
                        deg_file << " X: " << result.superloc_data.uncertainty_x << " ";
                        deg_file << " Y: " << result.superloc_data.uncertainty_y << " ";
                        deg_file << " Z: " << result.superloc_data.uncertainty_z << "\n";
                        deg_file << "      Rotation Uncertainty: ";
                        deg_file << "  Roll: " << result.superloc_data.uncertainty_roll << " ";
                        deg_file << "  Pitch: " << result.superloc_data.uncertainty_pitch << " ";
                        deg_file << "   Yaw: " << result.superloc_data.uncertainty_yaw << "\n\n";

                        //                        deg_file << "    Degeneracy Detection:\n";
                        //                        deg_file << "      Condition Numbers:\n";
                        //                        deg_file << "        Full: " << result.superloc_data.cond_full << "\n";
                        //                        deg_file << "        Rotation: " << result.superloc_data.cond_rot << "\n";
                        //                        deg_file << "        Translation: " << result.superloc_data.cond_trans << "\n";
                        deg_file << "      Is Degenerate: "
                                 << (result.superloc_data.is_degenerate ? "Yes" : "No") << "\n\n";

                        deg_file << "    Feature Histogram: [";
                        for (int i = 0; i < 9; ++i) {
                            deg_file << result.superloc_data.feature_histogram[i];
                            if (i < 8) deg_file << ", ";
                        }
                        deg_file << "]\n\n";

                        deg_file << "    Covariance Matrix (6x6):\n";
                        for (int i = 0; i < 6; ++i) {
                            deg_file << "      ";
                            for (int j = 0; j < 6; ++j) {
                                deg_file << std::setw(12) << result.superloc_data.covariance(i, j) << " ";
                            }
                            deg_file << "\n";
                        }
                        deg_file << "\n";
                    }

                    // 对于Schur方法，保存对齐分析
                    if ((method_name == "Ours" || method_name.find("SCHUR") != std::string::npos) &&
                        last_iter.is_degenerate) {

                        Eigen::Vector3d ref_axes[3];
                        ref_axes[0] << 1.0, 0.0, 0.0;
                        ref_axes[1] << 0.0, 1.0, 0.0;
                        ref_axes[2] << 0.0, 0.0, 1.0;

                        deg_file << "  Alignment Analysis:\n";
                        deg_file << "    Rotation Axes:\n";
                        for (int i = 0; i < 3; ++i) {
                            int idx = last_iter.rot_indices[i];
                            double l = (idx >= 0 && idx < 3) ? last_iter.lambda_schur_rot(idx) : NAN;
                            Eigen::Vector3d v = last_iter.aligned_V_rot.col(i);
                            double dot = std::abs(v.dot(ref_axes[i]));
                            double ang = std::acos(std::min(1.0, std::max(0.0, dot))) * 180.0 / M_PI;
                            double sabs = std::max(1e-9, v.cwiseAbs().sum());
                            double pR = 100 * std::abs(v(0)) / sabs;
                            double pP = 100 * std::abs(v(1)) / sabs;
                            double pY = 100 * std::abs(v(2)) / sabs;

                            deg_file << "      [" << i << "]~" << (i == 0 ? "R" : (i == 1 ? "P" : "Y"))
                                     << " (orig_idx=" << idx << "): λ=" << l
                                     << ", Angle=" << ang << "°, "
                                     << pR << "%R+" << pP << "%P+" << pY << "%Y\n";
                        }

                        deg_file << "    Translation Axes:\n";
                        for (int i = 0; i < 3; ++i) {
                            int idx = last_iter.trans_indices[i];
                            double l = (idx >= 0 && idx < 3) ? last_iter.lambda_schur_trans(idx) : NAN;
                            Eigen::Vector3d v = last_iter.aligned_V_trans.col(i);
                            double dot = std::abs(v.dot(ref_axes[i]));
                            double ang = std::acos(std::min(1.0, std::max(0.0, dot))) * 180.0 / M_PI;
                            double sabs = std::max(1e-9, v.cwiseAbs().sum());
                            double pX = 100 * std::abs(v(0)) / sabs;
                            double pY = 100 * std::abs(v(1)) / sabs;
                            double pZ = 100 * std::abs(v(2)) / sabs;

                            deg_file << "      [" << i << "]~" << (i == 0 ? "X" : (i == 1 ? "Y" : "Z"))
                                     << " (orig_idx=" << idx << "): λ=" << l
                                     << ", Angle=" << ang << "°, "
                                     << pX << "%X+" << pY << "%Y+" << pZ << "%Z\n";
                        }
                    }
                }

                deg_file << "\n" << std::string(60, '-') << "\n\n";
            }

            deg_file.close();
            std::cout << "Degeneracy analysis saved to: " << degeneracy_file << std::endl;
        }


        // 保存迭代历史（包含每次迭代的误差）
        std::string iter_his_filename = config_.output_folder + "iteration_history.csv";
        std::ofstream iter_his_file(iter_his_filename);

        if (iter_his_file.is_open()) {
            iter_his_file << "Method,Iteration,RMSE,Fitness,TransError,RotError,CorrNum\n";
            iter_his_file << std::fixed << std::setprecision(8);

            for (const auto&[method_name, results] : detailed_results_) {
                if (results.empty()) continue;
                const auto &result = results[0];

                for (const auto &iter_data : result.iteration_data) {
                    iter_his_file << method_name << ","
                                  << iter_data.iter_count << ","
                                  << iter_data.rmse << ","
                                  << iter_data.fitness << ","
                                  << iter_data.trans_error_vs_gt << ","
                                  << iter_data.rot_error_vs_gt << ","
                                  << iter_data.corr_num << "\n";
                }
            }
            iter_his_file.close();
            std::cout << "Iteration history saved to: " << iter_his_filename << std::endl;
        }

        // 1. 保存每次迭代的详细数据到CSV（包含update_dx）
        std::string iter_details_file = config_.output_folder + "iteration_details_with_dx.csv";
        std::ofstream iter_csv(iter_details_file);
        iter_csv << std::fixed << std::setprecision(8);

        if (iter_csv.is_open()) {
            // CSV头
            iter_csv << "Method,Run,Iteration,RMSE,Fitness,Time_ms,";
            iter_csv << "Trans_Error_m,Rot_Error_deg,P2P_RMSE,Chamfer_Distance,";
            // update_dx: [wx, wy, wz, x, y, z]
            iter_csv << "dx_wx,dx_wy,dx_wz,dx_x,dx_y,dx_z,";
            // 新增：梯度 [grad_wx, grad_wy, grad_wz, grad_x, grad_y, grad_z]
            iter_csv << "grad_wx,grad_wy,grad_wz,grad_x,grad_y,grad_z,";
            // 新增：目标函数值
            iter_csv << "objective_value,";
            // 变换矩阵 4x4
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    iter_csv << "T_" << i << j << ",";
                }
            }
            // 条件数
            iter_csv << "Cond_Schur_Rot,Cond_Schur_Trans,Cond_Sub_Rot,Cond_Sub_Trans,Cond_Full_SVD,";
            // 退化掩码
            for (int i = 0; i < 6; ++i) {
                iter_csv << "Degenerate_" << i << ",";
            }
            iter_csv << "Is_Degenerate\n";

            Pose6D initial_pose = config_.initial_noise;

            // 写入数据
            for (const auto&[method_name, results] : detailed_results_) {
                for (size_t run = 0; run < results.size(); ++run) {
                    const auto &result = results[run];

                    for (size_t iter = 0; iter < result.iteration_data.size(); ++iter) {
                        const auto &iter_data = result.iteration_data[iter];


                        // Calculate errors vs ground truth (identity)
                        PoseError error = calculatePoseError(config_.gt_matrix, iter_data.transform_matrix, true);
                        double trans_error = error.rotation_error;
                        double rot_error = error.translation_error;

                        // 计算点到点误差（仅在最后几次迭代或每隔几次迭代计算）
                        double p2p_rmse = 0.0, chamfer = 0.0;
                        if (iter == result.iteration_data.size() - 1 || iter % 1 == 0) {
                            pcl::PointCloud<PointT>::Ptr aligned_cloud(new pcl::PointCloud <PointT>);
                            pcl::transformPointCloud(*source_cloud_, *aligned_cloud, iter_data.transform_matrix);
                            double p2p_fitness;
                            int corr_num = 0;
                            calculatePointToPointError(aligned_cloud, target_cloud_, p2p_rmse, p2p_fitness, chamfer,
                                                       corr_num, config_.error_threshold);
                        }

                        // 写入一行数据
                        iter_csv << method_name << "," << run << "," << iter << ",";
                        iter_csv << iter_data.rmse << "," << iter_data.fitness << "," << iter_data.iter_time_ms << ",";
                        iter_csv << trans_error << "," << rot_error << "," << p2p_rmse << "," << chamfer << ",";

                        // update_dx (6维向量)
                        for (int i = 0; i < 6; ++i) {
                            iter_csv << iter_data.update_dx(i) << ",";
                        }
                        // 新增：梯度 (6维向量)
                        for (int i = 0; i < 6; ++i) {
                            iter_csv << iter_data.gradient(i) << ",";
                        }

                        // 新增：目标函数值
                        iter_csv << iter_data.objective_value << ",";

                        // 变换矩阵
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                iter_csv << iter_data.transform_matrix(i, j) << ",";
                            }
                        }

                        // 条件数
                        iter_csv << iter_data.cond_schur_rot << "," << iter_data.cond_schur_trans << ",";
                        iter_csv << iter_data.cond_diag_rot << "," << iter_data.cond_diag_trans << ",";
                        iter_csv << iter_data.cond_full_svd << ",";

                        // 退化掩码
                        for (int i = 0; i < 6; ++i) {
                            iter_csv << (iter_data.degenerate_mask[i] ? 1 : 0) << ",";
                        }
                        iter_csv << (iter_data.is_degenerate ? 1 : 0) << "\n";
                    }
                }
            }
            iter_csv.close();
            std::cout << "Iteration details with dx saved to: " << iter_details_file << std::endl;
        }


        // 为SuperLoc方法保存专门的不确定性和退化分析日志
        for (const auto&[method_name, results] : detailed_results_) {
            if (method_name.find("SUPERLOC") != std::string::npos && !results.empty()) {
                std::string superloc_log = config_.output_folder + method_name + "_superloc_analysis.txt";
                std::ofstream superloc_file(superloc_log);

                if (superloc_file.is_open()) {
                    superloc_file << "SuperLoc Degeneracy Analysis for " << method_name << "\n";
                    superloc_file << "================================================\n\n";

                    for (size_t run = 0; run < results.size(); ++run) {
                        const auto &result = results[run];
                        if (result.superloc_data.has_data) {
                            superloc_file << "Run " << run << ":\n";
                            superloc_file << "--------\n";
                            superloc_file << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
                            superloc_file << "Iterations: " << result.iterations << "\n";
                            superloc_file << "Final RMSE: " << result.final_rmse << "\n";
                            superloc_file << "Final Fitness: " << result.final_fitness << "\n\n";

                            superloc_file << "Feature Observability Uncertainties:\n";
                            superloc_file << "  Translation - X: " << result.superloc_data.uncertainty_x
                                          << ", Y: " << result.superloc_data.uncertainty_y
                                          << ", Z: " << result.superloc_data.uncertainty_z << "\n";
                            superloc_file << "  Rotation - Roll: " << result.superloc_data.uncertainty_roll
                                          << ", Pitch: " << result.superloc_data.uncertainty_pitch
                                          << ", Yaw: " << result.superloc_data.uncertainty_yaw << "\n\n";

                            superloc_file << "Condition Numbers:\n";
                            superloc_file << "  Full (6x6): " << result.superloc_data.cond_full << "\n";
                            superloc_file << "  Rotation (3x3): " << result.superloc_data.cond_rot << "\n";
                            superloc_file << "  Translation (3x3): " << result.superloc_data.cond_trans << "\n\n";

                            superloc_file << "Degeneracy Detection: "
                                          << (result.superloc_data.is_degenerate ? "DEGENERATE" : "NON-DEGENERATE")
                                          << "\n\n";

                            superloc_file << "Feature Observability Histogram:\n";
                            superloc_file << "  rx_cross: " << result.superloc_data.feature_histogram[0] << "\n";
                            superloc_file << "  neg_rx_cross: " << result.superloc_data.feature_histogram[1] << "\n";
                            superloc_file << "  ry_cross: " << result.superloc_data.feature_histogram[2] << "\n";
                            superloc_file << "  neg_ry_cross: " << result.superloc_data.feature_histogram[3] << "\n";
                            superloc_file << "  rz_cross: " << result.superloc_data.feature_histogram[4] << "\n";
                            superloc_file << "  neg_rz_cross: " << result.superloc_data.feature_histogram[5] << "\n";
                            superloc_file << "  tx_dot: " << result.superloc_data.feature_histogram[6] << "\n";
                            superloc_file << "  ty_dot: " << result.superloc_data.feature_histogram[7] << "\n";
                            superloc_file << "  tz_dot: " << result.superloc_data.feature_histogram[8] << "\n\n";

                            superloc_file << "Covariance Matrix (6x6):\n";
                            for (int i = 0; i < 6; ++i) {
                                for (int j = 0; j < 6; ++j) {
                                    superloc_file << std::scientific << std::setprecision(6)
                                                  << result.superloc_data.covariance(i, j) << " ";
                                }
                                superloc_file << "\n";
                            }
                            superloc_file << "\n";
                        }
                    }

                    superloc_file.close();
                    std::cout << "SuperLoc analysis saved to: " << superloc_log << std::endl;
                }
            }
        }

        // 保存SuperLoc特有的迭代数据
        for (const auto&[method_name, results] : detailed_results_) {
            if (method_name.find("SUPERLOC") != std::string::npos && !results.empty()) {
                std::string iter_log = config_.output_folder + method_name + "_superloc_iterations.csv";
                std::ofstream iter_file(iter_log);

                if (iter_file.is_open()) {
                    iter_file << "Run,Iteration,RMSE,Fitness,Corr_Num,Trans_Error,Rot_Error,Time_ms\n";

                    for (size_t run = 0; run < results.size(); ++run) {
                        const auto &result = results[run];
                        for (size_t iter = 0; iter < result.iteration_data.size(); ++iter) {
                            const auto &iter_data = result.iteration_data[iter];
                            iter_file << run << "," << iter << ","
                                      << iter_data.rmse << ","
                                      << iter_data.fitness << ","
                                      << iter_data.corr_num << ","
                                      << iter_data.trans_error_vs_gt << ","
                                      << iter_data.rot_error_vs_gt << ","
                                      << iter_data.iter_time_ms << "\n";
                        }
                    }

                    iter_file.close();
                    std::cout << "SuperLoc iteration details saved to: " << iter_log << std::endl;
                }
            }
        }

    }

/**
 * [功能描述]：基于SO(3)李群参数化的Point-to-Plane ICP算法实现，支持权重导数计算和退化处理。
 *             使用OpenMP并行加速，采用流形优化方法避免旋转表示的奇异性问题。
 * @param measure_cloud：待配准的源点云（测量点云）
 * @param target_cloud：目标点云，用于构建局部平面
 * @param initial_state：SE(3)表示的初始位姿状态（旋转矩阵+平移向量）
 * @param SEARCH_RADIUS：近邻搜索半径，用于寻找对应点
 * @param detection_method：退化检测方法枚举（如条件数检测、特征值检测等）
 * @param handling_method：退化处理方法枚举（如正则化、截断SVD等）
 * @param MAX_ITERATIONS：最大迭代次数
 * @param context：ICP上下文对象，存储中间数据和结果
 * @param result：测试结果对象，记录性能指标
 * @param output_state：输出的优化后SE(3)状态
 * @return bool：收敛成功返回true，失败返回false
 */
bool TestRunner::Point2PlaneICP_SO3_OpenMP(
        pcl::PointCloud<PointT>::Ptr measure_cloud,
        pcl::PointCloud<PointT>::Ptr target_cloud,
        const MathUtils::SE3State &initial_state,
        double SEARCH_RADIUS,
        DetectionMethod detection_method,
        HandlingMethod handling_method,
        int MAX_ITERATIONS,
        ICPContext &context,
        TestResult &result,
        MathUtils::SE3State &output_state) {

        // === 1. 算法初始化和预处理 ===
        
        // 启动总计时器，用于统计整个ICP过程的执行时间
        TicToc total_timer;
        // 清空上下文中的迭代日志数据，为新的ICP过程做准备
        context.iteration_log_data_.clear();
        // 初始化收敛标志为false
        context.final_convergence_flag_ = false;
        context.final_iterations_ = 0;

        // 单次迭代计时器，用于统计每次迭代的耗时
        TicToc tic_toc;
        // 设置初始位姿状态：使用输入的初始状态作为优化起点
        output_state = initial_state;

        // === 输入数据有效性检查 ===
        // 检查源点云是否有效
        if (!measure_cloud || measure_cloud->empty()) {
            std::cerr << "[ICP Error] Input measure cloud is null or empty." << std::endl;
            return false;
        }
        // 检查KD树是否已正确设置（用于快速近邻搜索）
        if (!context.kdtreeSurfFromMap || !context.kdtreeSurfFromMap->getInputCloud()) {
            std::cerr << "[ICP Error] KdTree is not set up in context." << std::endl;
            return false;
        }
        // 检查目标点云是否有效（用于平面拟合）
        if (!target_cloud || target_cloud->empty()) {
            std::cerr << "[ICP Error] Target cloud (for plane fitting) is null or empty." << std::endl;
            return false;
        }

        // === 调试信息输出（可选） ===
        // 可以启用以下代码来调试算法参数和状态
        //        std::cout << "\n=== Starting ICP SO3 ===" << std::endl;
        //                  << "Detection=" << static_cast<int>(detection_method)
        //                  << ", Handling=" << static_cast<int>(handling_method) << std::endl;
        //        std::cout << "[SO3 ICP] Parameters: KAPPA_TARGET=" << config_.icp_params.KAPPA_TARGET
        //                  << ", DEGENERACY_THRES_COND=" << config_.icp_params.DEGENERACY_THRES_COND << std::endl;

        // === 内部数据结构预分配 ===
        // 根据源点云大小预分配内存，避免动态扩容带来的性能开销
        size_t cloud_size = measure_cloud->size();
        if (context.laserCloudOriSurfVec.size() != cloud_size) {
            try {
                // 预分配向量存储空间
                context.laserCloudOriSurfVec.resize(cloud_size);    // 存储原始点云数据
                context.coeffSelSurfVec.resize(cloud_size);         // 存储平面系数（法向量和残差）
                context.laserCloudOriSurfFlag.resize(cloud_size);   // 存储点的有效性标志
            } catch (const std::bad_alloc &e) {
                std::cerr << "[ICP Error] Failed to allocate memory: " << e.what() << std::endl;
                return false;
            }
        }

        // === 2. ICP算法状态变量初始化 ===
        
        // 收敛性监控变量：用于追踪算法的收敛状态
        double prev_rmse = std::numeric_limits<double>::max();  // 上一次迭代的RMSE值
        double prev_fitness = 0.0;                             // 上一次迭代的匹配率
        double curr_rmse = 0.0;                                // 当前迭代的RMSE值
        double current_fitness = 0.0;                          // 当前迭代的匹配率

        // === 3. 线性系统求解矩阵预分配 ===
        // 预分配Jacobian矩阵和相关计算矩阵，避免每次迭代重复分配内存
        Eigen::Matrix<double, Eigen::Dynamic, 6> matA;         // Jacobian矩阵J (N×6)
        Eigen::Matrix<double, 6, Eigen::Dynamic> matAt;        // J的转置 (6×N)
        Eigen::Matrix<double, 6, 6> matAtA;                    // Hessian近似H=J^T*J (6×6)
        Eigen::VectorXd matB;                                  // 残差向量r (N×1)
        Eigen::VectorXd matAtB;                                // 梯度向量g=-J^T*r (6×1)
        Eigen::VectorXd matX;                                  // 更新向量dx (6×1)
        Eigen::Matrix<double, 6, 6> matAtA_last = Eigen::Matrix<double, 6, 6>::Identity();  // 上次迭代的Hessian

        // === 4. 最终状态记录变量 ===
        // 用于记录算法结束时的关键信息
        bool final_isDegenerate = false;                       // 最终是否检测到退化
        std::vector<bool> final_degenerate_mask(6, false);     // 退化维度掩码（ωx,ωy,ωz,x,y,z）
        double final_cond_full = NAN;                          // 最终的完整矩阵条件数
        double final_fitness = 0.0;                            // 最终的匹配率

        // === 5. 权重导数控制标志 ===
        // 控制是否在雅可比计算中包含权重函数的导数项
        // 设置为true时：J = s*J_r + r*(ds/dr)*J_r，更精确但计算量稍大
        // 设置为false时：J = s*J_r，与传统LOAM实现一致
        const bool USE_WEIGHT_DERIVATIVE = false; // 设置为true以包含权重导数

        // === 6. 优化主循环：迭代求解最优位姿变换 ===
        for (int iterCount = 0; iterCount < MAX_ITERATIONS; iterCount++) {
            tic_toc.tic();  // 开始单次迭代计时
            
            // 初始化当前迭代的日志数据结构
            IterationLogData current_iter_data;
            current_iter_data.iter_count = iterCount;

            // 清空上一次迭代的有效点云数据
            context.laserCloudEffective->clear();  // 清空有效的源点云
            context.coeffSel->clear();             // 清空对应的平面系数
            double total_distance_sq = 0.0;        // 累计平方距离，用于计算RMSE

            // 获取当前迭代的位姿变换矩阵（4×4齐次变换矩阵）
            Eigen::Matrix4d current_transform = output_state.matrix();

            // === 6.1 对应点搜索与残差计算阶段 ===
            int correspondence_count = 0;      // 有效对应点数量计数器
            int correspondence_pt_count = 0;   // 参与搜索的点数量计数器

            // 预分配存储数组，用于记录每个点的残差和权重信息
            // 这些信息将在雅可比计算阶段使用，按原始点云索引存储
            std::vector<double> point_raw_residuals(cloud_size, 0.0);      // 原始点到平面距离
            std::vector<double> point_weights(cloud_size, 0.0);            // Huber权重函数值
            std::vector<double> point_weight_derivatives(cloud_size, 0.0);  // 权重函数导数值

            // 使用OpenMP并行处理所有源点云的对应点搜索和平面拟合
            // reduction操作确保多线程安全地累加计数器变量
#pragma omp parallel for num_threads(8) reduction(+:correspondence_count,correspondence_pt_count,total_distance_sq)
            for (std::size_t i = 0; i < cloud_size; i++) {
                // 获取源点云中的当前点（在传感器坐标系下）
                PointT pointOri = measure_cloud->points[i];
                PointT pointSel;  // 变换后的点（在目标坐标系下）
                
                // 将源点从传感器坐标系变换到当前估计的目标坐标系
                // pointBodyToGlobal执行：pointSel = current_transform * pointOri
                pointBodyToGlobal(pointOri, pointSel, current_transform);

                // === 近邻搜索：寻找用于平面拟合的近邻点 ===
                std::vector<int> pointSearchInd(5);        // 存储5个最近邻点的索引
                std::vector<float> pointSearchSqDis(5);    // 存储对应的平方距离
                // 使用KD树快速搜索5个最近邻点
                int neighbors_found = context.kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                                                pointSearchSqDis);

                // 检查搜索结果的有效性
                const double MAX_SEARCH_RADIUS_SQ = SEARCH_RADIUS * SEARCH_RADIUS;
                if (neighbors_found == 5 && pointSearchSqDis[4] < MAX_SEARCH_RADIUS_SQ) {
                    // === 平面拟合：使用5个近邻点拟合局部平面 ===
                    // 设置平面方程 Ax + By + Cz + D = 0 的求解矩阵
                    Eigen::Matrix<double, 5, 3> matA0;     // 系数矩阵：每行是一个点的(x,y,z)
                    Eigen::Matrix<double, 5, 1> matB0 = Eigen::Matrix<double, 5, 1>::Constant(-1.0);  // 右端向量
                    matA0.setZero();
                    bool neighbors_valid = true;           // 近邻点有效性标志
                    correspondence_pt_count++;              // 增加参与搜索的点数计数

                    // 构建平面拟合的线性系统：收集5个近邻点的坐标
                    for (int j = 0; j < 5; ++j) {
                        // 检查近邻点索引的有效性，防止越界访问
                        if (pointSearchInd[j] < 0 || pointSearchInd[j] >= target_cloud->size()) {
                            neighbors_valid = false;
                            break;
                        }
                        // 将近邻点的3D坐标作为矩阵的一行
                        // 求解平面方程 pa*x + pb*y + pc*z + pd = 0
                        matA0.row(j) = target_cloud->points[pointSearchInd[j]].getVector3fMap().cast<double>();
                    }

                    // 如果近邻点无效，跳过当前点
                    if (!neighbors_valid) {
                        context.laserCloudOriSurfFlag[i] = 0;  // 标记为无效点
                        continue;
                    }

                    // === 最小二乘平面拟合 ===
                    // 求解线性系统 matA0 * [pa, pb, pc]^T = matB0
                    // 使用QR分解获得最小二乘解
                    Eigen::Vector3d matX0 = matA0.colPivHouseholderQr().solve(matB0);
                    double pa = matX0(0), pb = matX0(1), pc = matX0(2);  // 平面法向量的非归一化分量
                    double ps = matX0.norm();  // 法向量的模长
                    const double MIN_NORMAL_NORM = 1e-6;  // 最小法向量阈值，避免退化情况

                    // 检查平面法向量的有效性
                    if (ps < MIN_NORMAL_NORM) {
                        context.laserCloudOriSurfFlag[i] = 0;  // 法向量太小，标记为无效
                        continue;
                    }

                    // 归一化平面参数：将法向量归一化为单位向量
                    pa /= ps;  // 归一化后的法向量x分量
                    pb /= ps;  // 归一化后的法向量y分量  
                    pc /= ps;  // 归一化后的法向量z分量
                    double pd = 1.0 / ps;  // 平面距离参数（到原点的距离）

                    // === 平面质量检查：验证拟合平面的可靠性 ===
                    double max_dist_to_plane_sq = 0.0;
                    // 计算5个近邻点到拟合平面的最大距离
                    for (int j = 0; j < 5; ++j) {
                        // 计算点到平面的带符号距离：pa*x + pb*y + pc*z + pd
                        double dist_sq = pa * target_cloud->points[pointSearchInd[j]].x +
                                         pb * target_cloud->points[pointSearchInd[j]].y +
                                         pc * target_cloud->points[pointSearchInd[j]].z + pd;
                        dist_sq *= dist_sq;  // 转换为平方距离
                        max_dist_to_plane_sq = std::max(max_dist_to_plane_sq, dist_sq);
                    }

                    // 平面厚度阈值：如果近邻点分布过于离散，说明局部几何不适合平面拟合
                    const double MAX_PLANE_THICKNESS_SQ = 0.2 * 0.2;  // 最大平面厚度的平方（0.04 m²）
                    if (neighbors_valid && max_dist_to_plane_sq < MAX_PLANE_THICKNESS_SQ) {
                        // === 计算当前点到拟合平面的距离和权重 ===
                        // 点到平面的带符号距离（正值表示在法向量正方向）
                        double point_to_plane_dist = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                        double abs_dist = std::abs(point_to_plane_dist);  // 绝对距离
                        
                        // Huber损失函数的权重：s = max(0, 1 - 0.9 * |r|)
                        // 当距离较小时权重接近1，距离增大时权重线性递减，提高鲁棒性
                        double s = std::max(0.0, 1.0 - 0.9 * abs_dist);

                        // === 计算权重函数的导数（用于更精确的雅可比计算） ===
                        double ds_dr = 0.0;  // 权重对残差的导数 ds/dr
                        if (USE_WEIGHT_DERIVATIVE && s > 0.0 && s < 1.0) {
                            // 在线性递减区间：ds/dr = -0.9 * sign(r)
                            double sign_r = (point_to_plane_dist > 0) ? 1.0 : -1.0;
                            ds_dr = -0.9 * sign_r;
                        }

                        // 权重阈值检查：只保留权重足够大的对应点
                        if (s > 0.1) {  // 权重阈值，过滤权重过小的outliers
                            // === 构建平面约束系数 ===
                            // 存储加权的平面法向量和残差，用于后续雅可比计算
                            PointT coeff;
                            coeff.x = s * pa;                     // 加权法向量x分量
                            coeff.y = s * pb;                     // 加权法向量y分量  
                            coeff.z = s * pc;                     // 加权法向量z分量
                            coeff.intensity = s * point_to_plane_dist;  // 加权残差

                            // 保存原始点信息，将绝对距离存储在intensity字段
                            pointOri.intensity = abs_dist;
                            context.laserCloudOriSurfVec[i] = pointOri;   // 存储原始点
                            context.coeffSelSurfVec[i] = coeff;           // 存储平面约束系数
                            context.laserCloudOriSurfFlag[i] = 1;         // 标记为有效对应点

                            // === 存储权重计算所需的原始数据 ===
                            // 按原始点云索引存储，用于雅可比矩阵构建时的权重导数计算
                            point_raw_residuals[i] = point_to_plane_dist;  // 原始残差
                            point_weights[i] = s;                          // 权重值
                            point_weight_derivatives[i] = ds_dr;           // 权重导数

                            // 更新统计计数器（OpenMP reduction安全）
                            correspondence_count++;  // 有效对应点计数
                            total_distance_sq += point_to_plane_dist * point_to_plane_dist;  // 累计平方误差
                        } else {
                            context.laserCloudOriSurfFlag[i] = 0;  // 权重太小，标记为无效
                        }
                    } else {
                        context.laserCloudOriSurfFlag[i] = 0;  // 平面质量不佳，标记为无效
                    }
                } else {
                    context.laserCloudOriSurfFlag[i] = 0;  // 近邻搜索失败，标记为无效
                }
            }

            // === 6.2 有效对应点收集阶段 ===
            context.laserCloudEffective->clear();  // 清空有效点云容器
            context.coeffSel->clear();             // 清空对应系数容器
            // 根据实际有效对应点数量预分配内存，提高效率
            context.laserCloudEffective->reserve(correspondence_count);
            context.coeffSel->reserve(correspondence_count);

            // === 重新组织数据结构：从稀疏索引映射到密集数组 ===
            // 将按原始点云索引存储的数据重新组织为连续的数组
            // 这样可以确保后续雅可比计算时数据的一致性和访问效率
            std::vector<double> raw_residuals;        // 重组后的原始残差数组
            std::vector<double> weights;              // 重组后的权重数组
            std::vector<double> weight_derivatives;   // 重组后的权重导数数组
            raw_residuals.reserve(correspondence_count);
            weights.reserve(correspondence_count);
            weight_derivatives.reserve(correspondence_count);

            // 遍历所有点，收集标记为有效的对应点及其相关数据
            for (size_t i = 0; i < cloud_size; ++i) {
                if (context.laserCloudOriSurfFlag[i]) {  // 检查有效标志
                    // 将有效点和对应的平面约束系数添加到连续容器中
                    context.laserCloudEffective->push_back(context.laserCloudOriSurfVec[i]);
                    context.coeffSel->push_back(context.coeffSelSurfVec[i]);

                    // 同步收集权重相关数据，保持索引对应关系
                    raw_residuals.push_back(point_raw_residuals[i]);
                    weights.push_back(point_weights[i]);
                    weight_derivatives.push_back(point_weight_derivatives[i]);

                    // 重置标志，为下次迭代做准备
                    context.laserCloudOriSurfFlag[i] = 0;
                }
            }

            // === 6.3 有效对应点数量检查和质量评估 ===
            int laserCloudSelNum = context.laserCloudEffective->size();  // 获取实际有效对应点数量
            current_iter_data.corr_num = laserCloudSelNum;               // 记录到日志数据
            current_iter_data.effective_points = laserCloudSelNum;

            // 检查有效对应点是否足够进行可靠的位姿估计
            // 至少需要10个有效对应点才能构成over-determined系统（6DOF需要至少6个约束）
            if (laserCloudSelNum < 10) {
                std::cerr << "[ICP Warn Iter " << iterCount << "] Not enough effective points: "
                          << laserCloudSelNum << ". Aborting." << std::endl;
                context.final_iterations_ = iterCount + 1;   // 记录最终迭代次数
                context.final_convergence_flag_ = false;     // 标记为未收敛
                context.total_icp_time_ms_ = total_timer.toc();  // 记录总耗时
                return false;  // 提前终止算法
            }

            // === 计算当前迭代的配准质量指标 ===
            // fitness: 匹配率，表示有多少源点找到了有效对应点
            current_fitness = (measure_cloud->size() > 0) ? (double) correspondence_pt_count / measure_cloud->size()
                                                          : 0.0;
            // RMSE: 均方根误差，衡量对应点对的平均几何距离
            curr_rmse = (laserCloudSelNum > 0) ? std::sqrt(total_distance_sq / (double) laserCloudSelNum) : 0.0;
            current_iter_data.rmse = curr_rmse;      // 记录RMSE到日志
            current_iter_data.fitness = current_fitness;  // 记录匹配率到日志

            // === 6.4 构建雅可比矩阵J和残差向量r (SO(3)参数化 + 权重导数) ===
            matA.resize(laserCloudSelNum, 6);  // 雅可比矩阵J: N×6 (N个约束，6个待估参数)
            matB.resize(laserCloudSelNum);     // 残差向量r: N×1
            
            // 为每个有效对应点构建雅可比行和残差项
            for (int i = 0; i < laserCloudSelNum; i++) {
                // === 数据一致性说明 ===
                // coeffSel中存储的数据格式与欧拉角版本保持一致：
                // coeff.x = s * pa    (加权法向量x分量)
                // coeff.y = s * pb    (加权法向量y分量) 
                // coeff.z = s * pc    (加权法向量z分量)
                // coeff.intensity = s * point_to_plane_dist  (加权残差)

                // === 提取几何和权重信息 ===
                // 获取源点在传感器坐标系（body frame）下的坐标
                Eigen::Vector3d point_body(
                        context.laserCloudEffective->points[i].x,
                        context.laserCloudEffective->points[i].y,
                        context.laserCloudEffective->points[i].z
                );
                
                // 从coeffSel提取加权法向量（已包含权重s）
                Eigen::Vector3d weighted_normal(
                        context.coeffSel->points[i].x,
                        context.coeffSel->points[i].y,
                        context.coeffSel->points[i].z
                );
                
                // 获取当前点对应的权重和原始残差
                double s = weights[i];              // Huber权重函数值
                double r = raw_residuals[i];        // 原始点到平面距离
                
                // 恢复原始（未加权）法向量
                Eigen::Vector3d normal_unweighted = weighted_normal / s;

                // === SO(3)李群上的雅可比计算 ===
                // 计算点到平面约束在SE(3)流形上的雅可比矩阵
                // J_r = ∂r/∂ξ，其中ξ∈se(3)是李代数参数，r是点到平面距离
                Eigen::Matrix<double, 1, 6> J_r = MathUtils::computePointToPlaneJacobian(
                        point_body, normal_unweighted, output_state.R
                );

                // === 权重导数增强的雅可比计算 ===
                // 完整的雅可比考虑权重函数的变化：
                // 目标函数: f = s(r) * r²，其导数为: ∂f/∂ξ = (s + r*ds/dr) * r * ∂r/∂ξ
                // 因此雅可比为: J = (s + r*ds/dr) * J_r
                Eigen::Matrix<double, 1, 6> J;
                double ds_dr = weight_derivatives[i];  // 权重函数导数 ds/dr
                
                if (USE_WEIGHT_DERIVATIVE && ds_dr != 0.0) {
                    // 包含权重导数的完整雅可比（更精确但计算稍复杂）
                    J = (s + r * ds_dr) * J_r;
                } else {
                    // LOAM标准实现：只考虑当前权重，忽略权重导数
                    // 这与传统的欧拉角实现保持一致，计算更简单
                    J = s * J_r;
                }
                
                // === 填充线性系统矩阵 ===
                matA.row(i) = J;  // 第i行雅可比
                // 残差项使用负的加权残差（与欧拉角版本一致）
                matB(i) = -context.coeffSel->points[i].intensity;  // -s*r
            }

            // --- 5. Compute Hessian H = J^T * J and Gradient g = -J^T * r ---
            if (matAt.cols() != laserCloudSelNum) {
                matAt.resize(6, laserCloudSelNum);
            }
            matAt = matA.transpose();
            matAtA = matAt * matA;
            matAtB = matAt * matB;

            // 新增：保存梯度和目标函数值
            current_iter_data.gradient = -matAtB;  // 梯度是 -J^T * r（因为我们最小化）
            current_iter_data.objective_value = 0.5 * matB.squaredNorm();  // 0.5 * ||r||^2

            // --- 6. Degeneracy Analysis using unified function ---
            dcreg_->setConfig(config_);
            auto degeneracy_result = dcreg_->analyzeDegeneracy(matAtA, detection_method, handling_method);

            current_iter_data.eigenvalues_full = degeneracy_result.eigenvalues_full;
            current_iter_data.singular_values_full = degeneracy_result.singular_values;
            current_iter_data.cond_schur_rot = degeneracy_result.cond_schur_rot;
            current_iter_data.cond_schur_trans = degeneracy_result.cond_schur_trans;
            current_iter_data.cond_full_svd = degeneracy_result.cond_full;
            current_iter_data.cond_full_evd_sub_rot = degeneracy_result.cond_full_sub_rot;
            current_iter_data.cond_diag_trans = degeneracy_result.cond_diag_trans;
            current_iter_data.cond_diag_rot = degeneracy_result.cond_diag_rot;
            current_iter_data.cond_full_evd_sub_trans = degeneracy_result.cond_full_sub_trans;
            current_iter_data.lambda_schur_rot = degeneracy_result.lambda_schur_rot;
            current_iter_data.lambda_schur_trans = degeneracy_result.lambda_schur_trans;
            current_iter_data.is_degenerate = degeneracy_result.isDegenerate;
            current_iter_data.degenerate_mask = degeneracy_result.degenerate_mask;

            // --- 7. Solve Linear System H*dx = g using unified solver ---
            matX = dcreg_->solveDegenerateSystem(matAtA, matAtB, handling_method, degeneracy_result);

            if (!matX.allFinite()) {
                std::cerr << "[ICP Error Iter " << iterCount << "] Solver returned non-finite values!"
                          << std::endl;
                matX.setZero();
                context.final_iterations_ = iterCount;
                context.final_convergence_flag_ = false;
                context.total_icp_time_ms_ = total_timer.toc();
                return false;
            }

            // --- 8. Update State on Manifold ---
            output_state = output_state.boxplus(matX);
            current_iter_data.transform_matrix = output_state.matrix();
            current_iter_data.update_dx = matX;

            // --- 9. Check Convergence ---
            double deltaR_norm = matX.head<3>().norm();
            double deltaT_norm = matX.tail<3>().norm();

            double relative_rmse = std::abs(curr_rmse - prev_rmse);
            double relative_fitness = std::abs(current_fitness - prev_fitness);
            prev_rmse = curr_rmse;
            prev_fitness = current_fitness;
            matAtA_last = matAtA;

            final_isDegenerate = degeneracy_result.isDegenerate;
            final_degenerate_mask = degeneracy_result.degenerate_mask;
            final_cond_full = degeneracy_result.cond_full;
            final_fitness = current_fitness;

            // Log iteration data
            current_iter_data.iter_time_ms = tic_toc.toc();

            // Calculate errors vs ground truth (identity)
            PoseError error = calculatePoseError(config_.gt_matrix, current_iter_data.transform_matrix, true);
            current_iter_data.rot_error_vs_gt = error.rotation_error;
            current_iter_data.trans_error_vs_gt = error.translation_error;


            // 保存对角块特征值
            current_iter_data.lambda_diag_rot = degeneracy_result.lambda_sub_rot;
            current_iter_data.lambda_diag_trans = degeneracy_result.lambda_sub_trans;
            // 保存预处理矩阵
            if (handling_method == HandlingMethod::PRECONDITIONED_CG) {
                current_iter_data.P_preconditioner = degeneracy_result.P_preconditioner;
            } else if (handling_method == HandlingMethod::ADAPTIVE_REGULARIZATION) {
                current_iter_data.W_adaptive = degeneracy_result.W_adaptive;
            }
            // 保存对齐信息
            current_iter_data.aligned_V_rot = degeneracy_result.aligned_V_rot;
            current_iter_data.aligned_V_trans = degeneracy_result.aligned_V_trans;
            current_iter_data.rot_indices = degeneracy_result.rot_indices;
            current_iter_data.trans_indices = degeneracy_result.trans_indices;
            current_iter_data.cond_full = degeneracy_result.cond_full;
            context.iteration_log_data_.push_back(current_iter_data);

            if (deltaR_norm < config_.CONVERGENCE_THRESH_ROT && deltaT_norm < config_.CONVERGENCE_THRESH_TRANS) {
                context.final_convergence_flag_ = true;
                context.final_iterations_ = iterCount + 1;
                break;
            }
            context.final_iterations_ = iterCount + 1;
        }

        // --- 10. Post-Loop Processing ---
        context.total_icp_time_ms_ = total_timer.toc();

        // Convert final state to Pose6D for compatibility
        Eigen::Matrix4d final_matrix = output_state.matrix();
        context.final_pose_ = MatrixToPose6D(final_matrix);

        // --- 11. Covariance Calculation ---
        if (context.final_convergence_flag_) {
            Eigen::Matrix<double, 6, 6> H_final_for_cov = matAtA_last;
            Eigen::FullPivLU <Eigen::Matrix<double, 6, 6>> lu_cov(H_final_for_cov);
            if (lu_cov.isInvertible()) {
                context.icp_cov = lu_cov.inverse();
                // Regularize to ensure positive definite
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_cov(context.icp_cov);
                if (es_cov.info() != Eigen::Success || es_cov.eigenvalues().minCoeff() <= 1e-12) {
                    const double min_eig_cov = 1e-9;
                    Eigen::VectorXd eigenvalues_cov = es_cov.eigenvalues();
                    for (int i = 0; i < 6; ++i) {
                        eigenvalues_cov(i) = std::max(eigenvalues_cov(i), min_eig_cov);
                    }
                    context.icp_cov = es_cov.eigenvectors() * eigenvalues_cov.asDiagonal() *
                                      es_cov.eigenvectors().transpose();
                }
            } else {
                context.icp_cov.setIdentity();
                context.icp_cov *= 1e6;
            }
        } else {
            context.icp_cov.setIdentity();
            context.icp_cov *= 1e6;
        }

        // --- 12. Print Debug Info ---
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "--- ICP SO(3) Final State (Iter " << context.final_iterations_ << ") ---\n";
            std::cout << "Converged: " << (context.final_convergence_flag_ ? "Yes" : "No")
                      << " | RMSE: " << curr_rmse
                      << " | Fitness: " << final_fitness << "\n";
            std::cout << "Weight Derivative: " << (USE_WEIGHT_DERIVATIVE ? "Enabled" : "Disabled") << "\n";
            std::cout << "Degenerate: " << (final_isDegenerate ? "Yes" : "No") << "\n";
            std::cout << "Full Cond: " << final_cond_full << "\n";
            std::cout << "degenerate_mask ωxωyωz xyz: ";
            for (int i = 0; i < final_degenerate_mask.size(); ++i) {
                std::cout << final_degenerate_mask[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "----------------------------------------" << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }

        return context.final_convergence_flag_;
    }


    // --- Standalone ICP Function ---
    bool TestRunner::Point2PlaneICP(
            // Input Data & Parameters
            pcl::PointCloud<PointT>::Ptr measure_cloud,
            pcl::PointCloud<PointT>::Ptr target_cloud, // Still needed for plane fitting
            const Pose6D &initial_pose,
            double SEARCH_RADIUS,
            DetectionMethod detection_method,
            HandlingMethod handling_method,
            int MAX_ITERATIONS,
            ICPContext &context,        // Holds internal state, logs, and results
            TestResult &result,
            Pose6D &output_pose         // Final optimized pose is written here
    ) {
        // --- Overall Timer ---
        TicToc total_timer;
        context.iteration_log_data_.clear(); // Clear previous log data
        context.final_convergence_flag_ = false; // Reset convergence flag
        context.final_iterations_ = 0;

        // --- Initialization ---
        TicToc tic_toc; // Iteration timer
        output_pose = initial_pose; // Start with the initial guess

        // Input validation
        if (!measure_cloud || measure_cloud->empty()) {
            std::cerr << "[ICP Error] Input measure cloud is null or empty." << std::endl;
            return false;
        }
        if (!context.kdtreeSurfFromMap || !context.kdtreeSurfFromMap->getInputCloud()) {
            std::cerr << "[ICP Error] KdTree is not set up in context." << std::endl;
            return false;
        }
        if (!target_cloud || target_cloud->empty()) { // Need target for plane fitting
            std::cerr << "[ICP Error] Target cloud (for plane fitting) is null or empty." << std::endl;
            return false;
        }

        // Resize internal vectors in the context
        size_t cloud_size = measure_cloud->size();
        if (context.laserCloudOriSurfVec.size() != cloud_size) {
            try {
                context.laserCloudOriSurfVec.resize(cloud_size);
                context.coeffSelSurfVec.resize(cloud_size);
                context.laserCloudOriSurfFlag.resize(cloud_size);
            } catch (const std::bad_alloc &e) {
                std::cerr << "[ICP Error] Failed to allocate memory for point vectors: " << e.what() << std::endl;
                return false;
            }
        }

        // ICP internal state variables
        double prev_rmse = std::numeric_limits<double>::max();
        double prev_fitness = 0.0;
        double curr_rmse = 0.0;
        double current_fitness = 0.0; // Use local var for current fitness
        double realtive_rmse = std::numeric_limits<double>::max();
        double realtive_fitness = 0.0; // Use local var for current fitness

        Eigen::Vector3d ref_axes[3];
        ref_axes[0] << 1.0, 0.0, 0.0;
        ref_axes[1] << 0.0, 1.0, 0.0;
        ref_axes[2] << 0.0, 0.0, 1.0;

        // Pre-allocate matrices
        Eigen::Matrix<double, Eigen::Dynamic, 6> matA;
        Eigen::Matrix<double, 6, Eigen::Dynamic> matAt;
        Eigen::Matrix<double, 6, 6> matAtA;
        Eigen::VectorXd matB;
        Eigen::VectorXd matAtB;
        Eigen::VectorXd matX;
        Eigen::Matrix<double, 6, 6> matAtA_last = Eigen::Matrix<double, 6, 6>::Identity();

        // 变量用于存储最终状态，供打印和保存
        bool final_isDegenerate = false;
        std::vector<bool> final_degenerate_mask(6, false);
        double final_cond_full = NAN;
        double final_cond_full_sub_rot = NAN, final_cond_full_sub_trans = NAN;
        double final_cond_diag_rot = NAN, final_cond_diag_trans = NAN;
        double final_cond_schur_rot = NAN, final_cond_schur_trans = NAN;
        Eigen::Vector3d final_lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d final_lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Matrix<double, 6, 1> final_eigenvalues_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 1> final_singular_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 6> final_W_adaptive_current = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> final_P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix3d final_aligned_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d final_aligned_V_trans = Eigen::Matrix3d::Identity();
        std::vector<int> final_rot_indices = {0, 1, 2}, final_trans_indices = {0, 1, 2};
        double final_fitness = 0.0;

        // --- Optimization Main Loop ---
        for (int iterCount = 0; iterCount < MAX_ITERATIONS; iterCount++) {
            tic_toc.tic(); // Start iteration timer
            IterationLogData current_iter_data;
            current_iter_data.iter_count = iterCount;

            // Use context's clouds
            context.laserCloudEffective->clear();
            context.coeffSel->clear();
            double total_distance_sq = 0.0;

            Eigen::Matrix4d current_transform = Pose6D2Matrix(output_pose);

            // --- 1. Find Correspondences & Calculate Residuals/Normals ---
            int correspondence_count = 0, correspondence_pt_count = 0;
#pragma omp parallel for num_threads(8)
            for (size_t i = 0; i < cloud_size; i++) {
                PointT pointOri = measure_cloud->points[i];
                PointT pointSel;
                pointBodyToGlobal(pointOri, pointSel, current_transform);
                std::vector<int> pointSearchInd(5);
                std::vector<float> pointSearchSqDis(5);
                int neighbors_found = context.kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                                                pointSearchSqDis);
                const double MAX_SEARCH_RADIUS_SQ = SEARCH_RADIUS * SEARCH_RADIUS;
                if (neighbors_found == 5 && pointSearchSqDis[4] < MAX_SEARCH_RADIUS_SQ) {
                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = Eigen::Matrix<double, 5, 1>::Constant(-1.0);
                    matA0.setZero();
                    bool neighbors_valid = true;
                    correspondence_pt_count++;
                    for (int j = 0; j < 5; ++j) {
                        if (pointSearchInd[j] < 0 || pointSearchInd[j] >= target_cloud->size()) {
                            neighbors_valid = false;
                            break;
                        }
                        matA0.row(j) = target_cloud->points[pointSearchInd[j]].getVector3fMap().cast<double>();
                    }
                    if (!neighbors_valid) {
                        context.laserCloudOriSurfFlag[i] = 0;
                        continue;
                    }
                    Eigen::Vector3d matX0 = matA0.colPivHouseholderQr().solve(matB0);


                    double pa = matX0(0), pb = matX0(1), pc = matX0(2);
                    double ps = matX0.norm();
                    const double MIN_NORMAL_NORM = 1e-6;
                    if (ps < MIN_NORMAL_NORM) {
                        context.laserCloudOriSurfFlag[i] = 0;
                        continue;
                    }
                    pa /= ps;
                    pb /= ps;
                    pc /= ps;
                    double pd = 1.0 / ps;
                    double max_dist_to_plane_sq = 0.0; // 使用平方距离检查平面厚度仍然可以
                    for (int j = 0; j < 5; ++j) {
                        double dist_sq = pa * target_cloud->points[pointSearchInd[j]].x +
                                         pb * target_cloud->points[pointSearchInd[j]].y +
                                         pc * target_cloud->points[pointSearchInd[j]].z + pd;
                        dist_sq *= dist_sq; // 计算距离的平方
                        max_dist_to_plane_sq = std::max(max_dist_to_plane_sq, dist_sq);
                    }
                    const double MAX_PLANE_THICKNESS_SQ = 0.2 * 0.2; // 平面厚度检查阈值
                    if (neighbors_valid && max_dist_to_plane_sq < MAX_PLANE_THICKNESS_SQ) {
                        // 计算原始的带符号距离
                        double point_to_plane_dist = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                        // 计算绝对距离
                        double abs_dist = std::abs(point_to_plane_dist);
                        // 恢复原始的权重 s 计算方式
                        double s = std::max(0.0, 1.0 - 0.9 * abs_dist);
                        // 恢复原始的基于 s 的有效性判断
                        if (s > 0.1) {
                            PointT coeff;
                            // 使用计算出的 s 加权
                            coeff.x = s * pa;
                            coeff.y = s * pb;
                            coeff.z = s * pc;
                            // 使用计算出的 s 和带符号距离加权残差
                            coeff.intensity = s * point_to_plane_dist;
                            // 存储原始点的绝对距离（可选，用于调试或分析）
                            pointOri.intensity = abs_dist;
                            // 存储到 context 的 vectors 中
                            context.laserCloudOriSurfVec[i] = pointOri;
                            context.coeffSelSurfVec[i] = coeff;
                            context.laserCloudOriSurfFlag[i] = 1; // 标记为有效
                            correspondence_count++;
                            // 累加平方距离用于计算 RMSE
                            total_distance_sq += point_to_plane_dist * point_to_plane_dist;
                        } else {
                            context.laserCloudOriSurfFlag[i] = 0; // 标记为无效 (s 太小)
                        }
                    } else {
                        context.laserCloudOriSurfFlag[i] = 0; // 标记为无效 (平面厚度太大)
                    }
                } else {
                    context.laserCloudOriSurfFlag[i] = 0; // 标记为无效 (邻近点查找失败)
                }
            } // End correspondence loop

            // --- 2. Collect Effective Points ---
            context.laserCloudEffective->clear();
            context.coeffSel->clear();
            context.laserCloudEffective->reserve(correspondence_count);
            context.coeffSel->reserve(correspondence_count);
            for (size_t i = 0; i < cloud_size; ++i) {
                if (context.laserCloudOriSurfFlag[i]) {
                    context.laserCloudEffective->push_back(context.laserCloudOriSurfVec[i]);
                    context.coeffSel->push_back(context.coeffSelSurfVec[i]);
                }
            }
            std::fill(context.laserCloudOriSurfFlag.begin(), context.laserCloudOriSurfFlag.end(), 0);

            // --- 3. Check Effective Point Count & Calculate RMSE ---
            int laserCloudSelNum = context.laserCloudEffective->size();
            current_iter_data.effective_points = laserCloudSelNum;

            if (laserCloudSelNum < 10) {
                std::cerr << "[ICP Warn Iter " << iterCount << "] Not enough effective points: "
                          << laserCloudSelNum << ". Aborting." << std::endl;
                context.final_iterations_ = iterCount;
                context.final_convergence_flag_ = false;
                context.total_icp_time_ms_ = total_timer.toc();
                context.final_pose_ = output_pose; // Store pose before abort
                // Log the aborted iteration attempt
                current_iter_data.rmse = std::numeric_limits<double>::quiet_NaN();
                current_iter_data.fitness = (measure_cloud->size() > 0) ? (double) laserCloudSelNum /
                                                                          measure_cloud->size() : 0.0;
                current_iter_data.iter_time_ms = tic_toc.toc();
                context.iteration_log_data_.push_back(current_iter_data);
                return false; // Exit function
            }

            // current_fitness = (measure_cloud->size() > 0) ? (double) laserCloudSelNum / measure_cloud->size() : 0.0;
            current_fitness = (measure_cloud->size() > 0) ? (double) correspondence_pt_count / measure_cloud->size()
                                                          : 0.0;
            curr_rmse = (laserCloudSelNum > 0) ? std::sqrt(total_distance_sq / (double) laserCloudSelNum) : 0.0;
            current_iter_data.rmse = curr_rmse;
            current_iter_data.fitness = current_fitness;
            current_iter_data.corr_num = laserCloudSelNum;

            // --- 4. Build Jacobian J (matA) and Residual -r (matB) ---
            matA.resize(laserCloudSelNum, 6);
            matB.resize(laserCloudSelNum);
            double srx = sin(output_pose.pitch);
            double crx = cos(output_pose.pitch);
            double sry = sin(output_pose.yaw);
            double cry = cos(output_pose.yaw);
            double srz = sin(output_pose.roll);
            double crz = cos(output_pose.roll);

            PointT pointOri, coeff;
            for (int i = 0; i < laserCloudSelNum; i++) {
                // Coordinate transformation/swapping as in original code
                pointOri.x = context.laserCloudEffective->points[i].y;
                pointOri.y = context.laserCloudEffective->points[i].z;
                pointOri.z = context.laserCloudEffective->points[i].x;
                // Weighted normal components and residual from coeffSel cloud
                coeff.x = context.coeffSel->points[i].y;
                coeff.y = context.coeffSel->points[i].z;
                coeff.z = context.coeffSel->points[i].x;
                coeff.intensity = context.coeffSel->points[i].intensity; // Weighted residual s*dist

                // Calculate Jacobian entries (arx, ary, arz) - same calculation as before
                double crx_sry = crx * sry;
                double crz_sry = crz * sry;
                double srx_sry = srx * sry;
                double srx_srz = srx * srz;
                double arx =
                        (crx_sry * srz * pointOri.x + crx * crz_sry * pointOri.y - srx_sry * pointOri.z) * coeff.z +
                        (-srx_srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.x +
                        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) *
                        coeff.y;
                double ary = ((cry * srx_srz - crz_sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y +
                              crx * cry * pointOri.z) * coeff.z +
                             ((-cry * crz - srx_sry * srz) * pointOri.x + (cry * srz - crz * srx_sry) * pointOri.y -
                              crx_sry * pointOri.z) * coeff.y;
                double arz =
                        ((crz * srx_sry - cry * srz) * pointOri.x + (-cry * crz - srx_sry * srz) * pointOri.y) *
                        coeff.z + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.x +
                        ((sry * srz + cry * crz * srx) * pointOri.x + (crz_sry - cry * srx_srz) * pointOri.y) *
                        coeff.y;

                // Fill Jacobian Row (matA)
                matA(i, 0) = arz;
                matA(i, 1) = arx;
                matA(i, 2) = ary;
                matA(i, 3) = coeff.z;
                matA(i, 4) = coeff.x;
                matA(i, 5) = coeff.y;
                // Fill Residual Vector (matB) = -error
                matB(i) = -coeff.intensity;
            }

            // --- 5. Compute Hessian H = J^T * J and Gradient g = -J^T * r ---
            if (matAt.cols() != laserCloudSelNum) { matAt.resize(6, laserCloudSelNum); }
            matAt = matA.transpose();
            matAtA = matAt * matA;
            matAtB = matAt * matB;
            // 新增：保存梯度和目标函数值
            current_iter_data.gradient = -matAtB;  // 梯度是 -J^T * r
            current_iter_data.objective_value = 0.5 * matB.squaredNorm();  // 0.5 * ||r||^2

            // --- 6. Degeneracy Analysis & Preparation for Solver ---
            bool isDegenerate_current = false;
            std::vector<bool> degenerate_mask(6, false); // Mask for degenerate dimensions (R, P, Y, x, y, z)
            Eigen::Matrix<double, 6, 6> W_adaptive_current = Eigen::Matrix<double, 6, 6>::Zero(); // For Adaptive Regularization
            Eigen::Matrix<double, 6, 6> P_preconditioner_current = Eigen::Matrix<double, 6, 6>::Identity(); // For PCG

            // --- Analysis results (local to this iteration) ---
            double current_cond_schur_rot = NAN, current_cond_schur_trans = NAN;
            double current_cond_diag_rot = NAN, current_cond_diag_trans = NAN;
            double current_cond_full = NAN, current_cond_full_sub_rot = NAN, current_cond_full_sub_trans = NAN;
            Eigen::Vector3d current_lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
            Eigen::Vector3d current_lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
            Eigen::Matrix3d current_aligned_V_rot = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d current_aligned_V_trans = Eigen::Matrix3d::Identity();
            std::vector<int> current_rot_indices = {0, 1, 2}, current_trans_indices = {0, 1, 2};
            Eigen::Matrix<double, 6, 1> current_eigenvalues_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
            Eigen::Matrix<double, 6, 6> current_eigenvectors_full = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double, 6, 1> current_singular_values = Eigen::Matrix<double, 6, 1>::Constant(NAN);


            // --- 6a. Perform Necessary Analyses based on selected methods ---
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_full;
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd_full;
           
            es_full.compute(matAtA);
            if (es_full.info() == Eigen::Success) {
                current_eigenvalues_full = es_full.eigenvalues(); // Sorted small to large
                current_eigenvectors_full = es_full.eigenvectors();
                if (current_eigenvalues_full.size() == 6) {
                    double min_eig_trans = std::max(std::abs(current_eigenvalues_full(0)), 1e-12);
                    double max_eig_trans = std::abs(current_eigenvalues_full(2));
                    current_cond_full_sub_trans = max_eig_trans / min_eig_trans;
                    double min_eig_rot = std::max(std::abs(current_eigenvalues_full(3)), 1e-12);
                    double max_eig_rot = std::abs(current_eigenvalues_full(5));
                    current_cond_full_sub_rot = max_eig_rot / min_eig_rot;
                } else {
                    current_cond_full_sub_rot = current_cond_full_sub_trans = std::numeric_limits<double>::infinity();
                }
            } else {
                std::cerr << "[ICP Warn Iter " << iterCount << "] Full EVD failed." << std::endl;
                current_cond_full_sub_rot = current_cond_full_sub_trans = std::numeric_limits<double>::infinity();
                current_eigenvalues_full.fill(NAN); // Mark as invalid
            }
           
            svd_full.compute(matAtA, Eigen::ComputeThinU | Eigen::ComputeThinV);
            current_singular_values = svd_full.singularValues(); // Sorted large to small
            if (current_singular_values.size() == 6 && current_singular_values(5) > 1e-12) {
                current_cond_full = current_singular_values(0) / current_singular_values(5);
            } else {
                current_cond_full = std::numeric_limits<double>::infinity();
            }
            // }

            // Note about eigenvalue/singular value relationship
            // For symmetric positive semi-definite H = J^T*J:
            // - eigenvalues(H) are real and non-negative
            // - singular_values(H) = abs(eigenvalues(H))
            // - condition_number = max_eigenvalue / min_eigenvalue = max_singular / min_singular

            /* ... Diagonal Block Condition Number computation ... */
            Eigen::Matrix3d H_RR = matAtA.block<3, 3>(0, 0);
            Eigen::Matrix3d H_tt = matAtA.block<3, 3>(3, 3);
            Eigen::Matrix3d H_Rt = matAtA.block<3, 3>(0, 3);
            Eigen::Matrix3d H_tR = matAtA.block<3, 3>(3, 0);
            Eigen::FullPivLU <Eigen::Matrix3d> lu_tt(H_tt);
            Eigen::FullPivLU <Eigen::Matrix3d> lu_rr(H_RR);


            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_diag_rot(H_RR);
            if (es_diag_rot.info() == Eigen::Success) {
                current_cond_diag_rot = es_diag_rot.eigenvalues().maxCoeff() /
                                        std::max(es_diag_rot.eigenvalues().minCoeff(), 1e-12);
            } else {
                current_cond_diag_rot = std::numeric_limits<double>::infinity();
            }
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_diag_trans(H_tt);
            if (es_diag_trans.info() == Eigen::Success) {
                current_cond_diag_trans = es_diag_trans.eigenvalues().maxCoeff() /
                                          std::max(es_diag_trans.eigenvalues().minCoeff(), 1e-12);
            } else {
                current_cond_diag_trans = std::numeric_limits<double>::infinity();
            }
            // CACULATE SCHUR CONDITION NUMBER


            if (lu_tt.isInvertible() && lu_rr.isInvertible()) {
                Eigen::Matrix3d H_tt_inv = lu_tt.inverse();
                Eigen::Matrix3d H_RR_inv = lu_rr.inverse();
                Eigen::Matrix3d Schur_R = H_RR - H_Rt * H_tt_inv * H_tR;
                Eigen::Matrix3d Schur_T = H_tt - H_tR * H_RR_inv * H_Rt;
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_rot(Schur_R);
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_trans(Schur_T);

                if (es_rot.info() == Eigen::Success && es_trans.info() == Eigen::Success) {
                    current_lambda_schur_rot = es_rot.eigenvalues();
                    Eigen::Matrix3d V_rot_raw = es_rot.eigenvectors();
                    current_lambda_schur_trans = es_trans.eigenvalues();
                    Eigen::Matrix3d V_trans_raw = es_trans.eigenvectors();
                    current_cond_schur_rot = current_lambda_schur_rot.maxCoeff() /
                                             std::max(current_lambda_schur_rot.minCoeff(), 1e-12);
                    current_cond_schur_trans = current_lambda_schur_trans.maxCoeff() /
                                               std::max(current_lambda_schur_trans.minCoeff(), 1e-12);
                } else {
                    std::cerr << "[ICP Warn Iter " << iterCount << "] Schur EVD failed." << std::endl;
                    current_cond_schur_rot = current_cond_schur_trans = std::numeric_limits<double>::infinity();
                }
            } else {
                std::cerr << "[ICP Warn Iter " << iterCount
                          << "] H_tt or H_RR blocks are singular. Cannot compute Schur complement.\n"
                          << matAtA.block<3, 3>(0, 0) << std::endl;
                current_cond_schur_rot = current_cond_schur_trans = std::numeric_limits<double>::infinity();
            }
            // }

            // --- 6b. Determine Degeneracy Flag & Mask based on chosen detection_method ---
            isDegenerate_current = false;
            degenerate_mask.assign(6, false); // Reset
            switch (detection_method) {
                case DetectionMethod::SCHUR_CONDITION_NUMBER: {
                    std::cout << "wait for code release" << std::endl;
                    break;
                }
                case DetectionMethod::FULL_EVD_MIN_EIGENVALUE: {
                    if (es_full.info() == Eigen::Success && current_eigenvalues_full.allFinite() &&
                        current_eigenvalues_full.size() == 6) {
                        for (int i = 0; i < 6; ++i)
                            if (current_eigenvalues_full(i) < config_.icp_params.DEGENERACY_THRES_EIG) {
                                isDegenerate_current = true;
                                degenerate_mask[i] = true;
                            }
                    }
                    break;
                }
                case DetectionMethod::EVD_SUB_CONDITION: {
                    isDegenerate_current = (current_cond_full_sub_rot > config_.icp_params.DEGENERACY_THRES_COND ||
                                            current_cond_full_sub_trans > config_.icp_params.DEGENERACY_THRES_COND);
                    if (isDegenerate_current && es_full.info() == Eigen::Success &&
                        current_eigenvalues_full.allFinite() && current_eigenvalues_full.size() == 6) {
                        if (current_cond_full_sub_trans >
                            config_.icp_params.DEGENERACY_THRES_COND) { degenerate_mask[0] = true; }
                        if (current_cond_full_sub_rot >
                            config_.icp_params.DEGENERACY_THRES_COND) { degenerate_mask[3] = true; }
                    } else if (isDegenerate_current) { degenerate_mask.assign(6, true); }
                    break;
                }
                case DetectionMethod::FULL_SVD_CONDITION: {
                    isDegenerate_current = (current_cond_full > config_.icp_params.DEGENERACY_THRES_COND);
                    if (isDegenerate_current && current_singular_values.allFinite() &&
                        current_singular_values.size() == 6) {
                        double max_sv = current_singular_values(0);
                        for (int i = 0; i < 6; ++i)
                            if (current_singular_values(i) < 1e-12 || max_sv / current_singular_values(i) >
                                                                      config_.icp_params.DEGENERACY_THRES_COND) { degenerate_mask[i] = true; }
                    } else if (isDegenerate_current) { degenerate_mask.assign(6, true); }
                    break;
                }
                default:
                    isDegenerate_current = false;
                    break;
            }


            // Store analysis results in log data
            current_iter_data.cond_schur_rot = current_cond_schur_rot;
            current_iter_data.cond_schur_trans = current_cond_schur_trans;
            current_iter_data.cond_diag_rot = current_cond_diag_rot;
            current_iter_data.cond_diag_trans = current_cond_diag_trans;
            current_iter_data.cond_full_evd_sub_rot = current_cond_full_sub_rot;
            current_iter_data.cond_full_evd_sub_trans = current_cond_full_sub_trans;
            current_iter_data.cond_full_svd = current_cond_full;
            current_iter_data.lambda_schur_rot = current_lambda_schur_rot;
            current_iter_data.lambda_schur_trans = current_lambda_schur_trans;
            current_iter_data.eigenvalues_full = current_eigenvalues_full;
            current_iter_data.singular_values_full = current_singular_values;
            current_iter_data.is_degenerate = isDegenerate_current;
            current_iter_data.degenerate_mask = degenerate_mask;


            // --- 6c. Construct Regularization/Preconditioner Matrix if needed ---
              
            // 在步骤7中，替换求解器代码：
            // matX = solveDegenerateSystem(matAtA, matAtB, handling_method, degeneracy_result);

            // --- 7. Solve Linear System H*dx = g ---
            try {
                switch (handling_method) {
                    case HandlingMethod::STANDARD_REGULARIZATION: {
                        Eigen::Matrix<double, 6, 6> H_reg = matAtA;
                        if (isDegenerate_current) H_reg.diagonal().array() += config_.icp_params.STD_REG_GAMMA;
                        matX = H_reg.colPivHouseholderQr().solve(matAtB);
                        break;
                    }
    
                    case HandlingMethod::PRECONDITIONED_CG: {
                        // wait for code release
                        // matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        break;
                    }
                    case HandlingMethod::SOLUTION_REMAPPING: {
                        Eigen::Matrix<double, 6, 1> Sigma_inv_diag_svd;
                        Sigma_inv_diag_svd.setZero();
                        if (svd_full.computeU() && svd_full.computeV() && current_singular_values.allFinite() &&
                            current_singular_values.size() == 6) {
                            for (int i = 0; i < 6; ++i)
                                if (current_singular_values(i) > 1e-9)
                                    Sigma_inv_diag_svd(i) = 1.0 / current_singular_values(i);
                            matX = svd_full.matrixV() * Sigma_inv_diag_svd.asDiagonal() *
                                   svd_full.matrixU().transpose() * matAtB;
                        } else { matX = matAtA.colPivHouseholderQr().solve(matAtB); }
                        if (isDegenerate_current && es_full.info() == Eigen::Success &&
                            current_eigenvectors_full.cols() == 6) {
                            Eigen::Matrix<double, 6, 6> P_projector = Eigen::Matrix<double, 6, 6>::Zero();
                            int good_dims = 0;
                            for (int i = 0; i < 6; ++i)
                                if (!degenerate_mask[i]) {
                                    P_projector += current_eigenvectors_full.col(i) * current_eigenvectors_full.col(
                                            i).transpose();
                                    good_dims++;
                                }
                            if (good_dims > 0) matX = P_projector * matX; else { matX.setZero(); }
                        }
                        break;
                    }
                    case HandlingMethod::TRUNCATED_SVD: {
                        if (svd_full.computeU() && svd_full.computeV() && current_singular_values.allFinite() &&
                            current_singular_values.size() == 6) {
                            Eigen::Matrix<double, 6, 1> Sigma_prime_inv_diag;
                            int retained_dims = 0;
                            for (int i = 0; i < 6; ++i) {
                                if (!degenerate_mask[i] && current_singular_values(i) > 1e-9) {
                                    Sigma_prime_inv_diag(i) = 1.0 / current_singular_values(i);
                                    retained_dims++;
                                } else { Sigma_prime_inv_diag(i) = 0.0; }
                            }
                            if (retained_dims == 0) matX.setZero();
                            else
                                matX = svd_full.matrixV() * Sigma_prime_inv_diag.asDiagonal() *
                                       svd_full.matrixU().transpose() * matAtB;
                        } else { matX = matAtA.colPivHouseholderQr().solve(matAtB); }
                        break;
                    }
                    case HandlingMethod::NONE_HAND:
                    default: {
                        matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        break;
                    }
                }
                if (!matX.allFinite()) {
                    std::cerr << "[ICP Error Iter " << iterCount << "] Solver returned non-finite values! Aborting."
                              << std::endl;
                    matX.setZero(); // Reset update
                    context.final_iterations_ = iterCount;
                    context.final_convergence_flag_ = false;
                    context.total_icp_time_ms_ = total_timer.toc();
                    context.final_pose_ = output_pose;
                    current_iter_data.update_dx.fill(NAN);
                    current_iter_data.iter_time_ms = tic_toc.toc();
                    context.iteration_log_data_.push_back(current_iter_data);
                    return false; // Exit function
                }
            } catch (const std::exception &e) {
                std::cerr << "[ICP Error Iter " << iterCount << "] Exception during solver: " << e.what()
                          << ". Aborting." << std::endl;
                context.final_iterations_ = iterCount;
                context.final_convergence_flag_ = false;
                context.total_icp_time_ms_ = total_timer.toc();
                context.final_pose_ = output_pose;
                current_iter_data.update_dx.fill(NAN);
                current_iter_data.iter_time_ms = tic_toc.toc();
                context.iteration_log_data_.push_back(current_iter_data);
                return false; // Exit function
            }
            current_iter_data.update_dx = matX; // Log the computed update

            // --- 8. Update Pose ---
            output_pose.roll += matX(0);
            output_pose.pitch += matX(1);
            output_pose.yaw += matX(2);
            output_pose.x += matX(3);
            output_pose.y += matX(4);
            output_pose.z += matX(5);

            // --- 9. Check Convergence & Store Last Iteration Info ---
            double deltaR_rad_sq = sqrt(pow(pcl::rad2deg(matX(0)), 2) + pow(pcl::rad2deg(matX(1)), 2) +
                                        pow(pcl::rad2deg(matX(2)), 2));
            double deltaT_m_sq = sqrt(pow(matX(3) * 100, 2) + pow(matX(4) * 100, 2) + pow(matX(5) * 100, 2));

            realtive_rmse = curr_rmse - prev_rmse;
            realtive_fitness = current_fitness - prev_fitness;
            prev_rmse = curr_rmse;
            prev_fitness = current_fitness;
            matAtA_last = matAtA; // Store Hessian for final covariance calculation

            // 保存最后一次迭代的状态，用于最终输出
            final_isDegenerate = isDegenerate_current;
            final_degenerate_mask = degenerate_mask;
            final_cond_full = current_cond_full;
            final_cond_full_sub_rot = current_cond_full_sub_rot;
            final_cond_full_sub_trans = current_cond_full_sub_trans;
            final_cond_diag_rot = current_cond_diag_rot;
            final_cond_diag_trans = current_cond_diag_trans;
            final_cond_schur_rot = current_cond_schur_rot;
            final_cond_schur_trans = current_cond_schur_trans;
            final_lambda_schur_rot = current_lambda_schur_rot;
            final_lambda_schur_trans = current_lambda_schur_trans;
            final_eigenvalues_full = current_eigenvalues_full;
            final_singular_full = current_singular_values;
            final_W_adaptive_current = W_adaptive_current;
            final_P_preconditioner = P_preconditioner_current;
            final_aligned_V_rot = current_aligned_V_rot;  // 不再进行类型转换
            final_aligned_V_trans = current_aligned_V_trans;  // 不再进行类型转换
            final_rot_indices = current_rot_indices;
            final_trans_indices = current_trans_indices;
            final_fitness = current_fitness;

            // Log iteration data
            current_iter_data.iter_time_ms = tic_toc.toc(); // Stop iteration timer
            current_iter_data.transform_matrix = Pose6D2Matrix(output_pose);
            context.iteration_log_data_.push_back(current_iter_data); // Add data for this iteration

            // Convergence Check
            const double CONVERGENCE_THRESH_ROT_SQ_ORIG = 1e-4; // 原始阈值 (deg²)
            const double CONVERGENCE_THRESH_TRANS_SQ_ORIG = 1e-4; // 原始阈值 (cm²)

            if (std::abs(realtive_rmse) < CONVERGENCE_THRESH_ROT_SQ_ORIG &&
                std::abs(realtive_fitness) < CONVERGENCE_THRESH_TRANS_SQ_ORIG) {
                context.final_convergence_flag_ = true; // Converged
                context.final_iterations_ = iterCount + 1;
                break; // Exit loop
            }
            context.final_iterations_ = iterCount + 1; // Update iteration count
        } // --- End of optimization loop ---

        // --- 10. Post-Loop Processing & Final State ---
        context.total_icp_time_ms_ = total_timer.toc(); // Stop total timer
        context.final_pose_ = output_pose; // Store the final pose in the context as well

        // --- 11. Covariance Calculation ---
        if (context.final_convergence_flag_) {
            Eigen::Matrix<double, 6, 6> H_final_for_cov = matAtA_last; // Use Hessian from before final update
            Eigen::FullPivLU <Eigen::Matrix<double, 6, 6>> lu_cov(H_final_for_cov);
            if (lu_cov.isInvertible()) {
                Eigen::Matrix<double, 6, 6> initial_cov_euler = lu_cov.inverse();
                // Optional: Regularize initial_cov_euler to ensure positive definite
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_cov(initial_cov_euler);
                if (es_cov.info() != Eigen::Success || es_cov.eigenvalues().minCoeff() <= 1e-12) {
                    const double min_eig_cov = 1e-9;
                    Eigen::VectorXd eigenvalues_cov = es_cov.eigenvalues();
                    for (int i = 0; i < 6; ++i) eigenvalues_cov(i) = std::max(eigenvalues_cov(i), min_eig_cov);
                    initial_cov_euler =
                            es_cov.eigenvectors() * eigenvalues_cov.asDiagonal() *
                            es_cov.eigenvectors().transpose();
                }

                Eigen::Matrix<double, 6, 6> cov_euler_d = initial_cov_euler;
                Eigen::Matrix3d J_rot_lie = MathUtils::computeEulerToLieJacobian(output_pose.roll, output_pose.pitch,
                                                                                 output_pose.yaw);
                Eigen::Matrix<double, 6, 6> J_cov_transform = Eigen::Matrix<double, 6, 6>::Identity();
                J_cov_transform.block<3, 3>(0, 0) = J_rot_lie;

                Eigen::Matrix<double, 6, 6> icp_cov_temp_d =
                        J_cov_transform * cov_euler_d * J_cov_transform.transpose();

                Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> solver_final_cov(icp_cov_temp_d);
                Eigen::Matrix<double, 6, 1> eigenvalues_final = solver_final_cov.eigenvalues();
                const double min_eigenvalue_cov_final = 1e-9;
                for (int i = 0; i < 6; ++i)
                    eigenvalues_final(i) = std::max(eigenvalues_final(i), min_eigenvalue_cov_final);
                context.icp_cov = (solver_final_cov.eigenvectors() * eigenvalues_final.asDiagonal() *
                                   solver_final_cov.eigenvectors().transpose());
            } else {
                std::cerr
                        << "[ICP Warning] Final Hessian for covariance (matAtA_last) is singular. Setting covariance to high Identity."
                        << std::endl;
                context.icp_cov.setIdentity();
                context.icp_cov *= 1e6;
            }
        } else {
            context.icp_cov.setIdentity();
            context.icp_cov *= 1e6;
        }

        // --- 12. Print Debug Info to Terminal and save to file ---
        {
            std::cout << std::fixed
                      << std::setprecision(6);
            std::cout << "--- ICP Final State (Iter " << context.final_iterations_ << ") ---\n";
            std::cout << "Converged: " << (context.final_convergence_flag_ ? "Yes" : "No") << " | RMSE: "
                      << curr_rmse
                      << " | Fitness: "
                      << final_fitness << "\n";
            std::cout << "Detect Method: ";
            switch (detection_method) {
                case DetectionMethod::NONE_DETE:
                    std::cout << "NONE";
                    break;
                case DetectionMethod::SCHUR_CONDITION_NUMBER:
                    std::cout << "SchurCond";
                    break;
                case DetectionMethod::FULL_EVD_MIN_EIGENVALUE:
                    std::cout << "FullMinEig";
                    break;
                case DetectionMethod::EVD_SUB_CONDITION:
                    std::cout << "EVDSubCond";
                    break;
                case DetectionMethod::FULL_SVD_CONDITION:
                    std::cout << "FullSVDCond";
                    break;
                default:
                    std::cout << "Unknown";
                    break;
            }
            std::cout << " | Handle Method: ";
            switch (handling_method) {
                case HandlingMethod::NONE_HAND:
                    std::cout << "NONE";
                    break;
                case HandlingMethod::STANDARD_REGULARIZATION:
                    std::cout << "STD_REG(g=" << config_.icp_params.STD_REG_GAMMA << ")";
                    break;
                case HandlingMethod::ADAPTIVE_REGULARIZATION:
                    std::cout << "ADAPTIVE_REG(a=" << config_.icp_params.ADAPTIVE_REG_ALPHA << ")";
                    break;
                case HandlingMethod::PRECONDITIONED_CG:
                    std::cout << "PCG(k=" << config_.icp_params.KAPPA_TARGET << ")";
                    break;
                case HandlingMethod::SOLUTION_REMAPPING:
                    std::cout << "LOAM_Remap(t=" << config_.icp_params.LOAM_EIGEN_THRESH << ")";
                    break;
                case HandlingMethod::TRUNCATED_SVD:
                    std::cout << "TSVD(t=" << config_.icp_params.TSVD_SINGULAR_THRESH << ")";
                    break;
            }
            std::cout << " | Degenerate: " << (final_isDegenerate ? "Yes" : "No") << "\n";

            // Condensed Condition Numbers
            std::cout << "CondNums: Full=" << final_cond_full
                      << " | FullSub(R/T)=" << final_cond_full_sub_rot << "/" << final_cond_full_sub_trans
                      << " | Diag(R/T)=" << final_cond_diag_rot << "/" << final_cond_diag_trans
                      << " | Schur(R/T)=" << final_cond_schur_rot << "/" << final_cond_schur_trans << "\n";
            std::cout << "SchurLambdas: R=[" << final_lambda_schur_rot.transpose() << "] T=["
                      << final_lambda_schur_trans.transpose() << "]\n";
            std::cout << "FullEvals (EVD): [" << final_eigenvalues_full.transpose() << "]\n";
            if (detection_method == DetectionMethod::FULL_SVD_CONDITION) {
                std::cout << "FullSigmas(SVD): [" << final_singular_full.transpose() << "]\n";
            }

            // Print final W or P diagonals if relevant method was used and degenerate
            if (final_isDegenerate) {
                if (handling_method == HandlingMethod::ADAPTIVE_REGULARIZATION) {
                    std::cout << "Final W Diag: [" << final_W_adaptive_current.diagonal().transpose() << "]\n";
                } else if (handling_method == HandlingMethod::PRECONDITIONED_CG) {
                    std::cout << "Final P Diag: [" << final_P_preconditioner.diagonal().transpose() << "]\n";
                }
            }
            // print degenerate_mask
            std::cout << "degenerate_mask rpyxyz: ";
            for (int i = 0; i < final_degenerate_mask.size(); ++i) {
                std::cout << final_degenerate_mask[i] << " ";
            }
            std::cout << std::endl;

            // Print alignment details only if adaptive/PCG method used AND degenerate
            if (final_isDegenerate &&
                (handling_method == HandlingMethod::ADAPTIVE_REGULARIZATION ||
                 handling_method == HandlingMethod::PRECONDITIONED_CG)) {
                // wait for code release
            }
            std::cout << "----------------------------------------" << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }

        return context.final_convergence_flag_; // Return convergence status
    }


    /**
     * [功能描述]：基于XICP（Extended ICP）的点到平面ICP配准算法
     * XICP是一种具有退化感知能力的ICP方法，通过特征分析检测配准过程中的退化方向，
     * 并采用约束优化或投影方法缓解退化问题，确保在结构退化环境下的鲁棒定位。
     * 该实现使用SO(3)×R³左乘扰动参数化，并支持TBB并行加速。
     * 
     * @param measure_cloud：测量点云（源点云），需要配准的点云数据
     * @param target_cloud：目标点云（参考点云），配准的目标
     * @param initial_state：初始SE(3)位姿状态，包含旋转和平移
     * @param SEARCH_RADIUS：最近邻搜索半径，用于建立点对对应关系
     * @param detection_method：退化检测方法枚举，指定XICP的检测策略
     * @param handling_method：退化处理方法枚举，指定缓解策略（约束/投影）
     * @param MAX_ITERATIONS：最大迭代次数
     * @param context：[输入/输出] ICP上下文，包含K-d树、法向量等预计算数据
     * @param result：[输出] 测试结果，存储配准性能指标
     * @param output_state：[输出] 最终优化后的SE(3)位姿状态
     * @return bool：配准成功返回true，失败返回false
     */
    bool TestRunner::Point2PlaneICP_SO3_tbb_XICP(
            pcl::PointCloud<PointT>::Ptr measure_cloud,
            pcl::PointCloud<PointT>::Ptr target_cloud,
            const MathUtils::SE3State &initial_state,
            double SEARCH_RADIUS,
            DetectionMethod detection_method,
            HandlingMethod handling_method,
            int MAX_ITERATIONS,
            ICPContext &context,
            TestResult &result,
            MathUtils::SE3State &output_state) {

        // ========== 步骤1：设置XICP参数 ==========
        // 创建XICP退化检测参数对象
        XICP::DegeneracyDetectionParameters<double> xicpParams;

        // 从配置文件中加载XICP算法参数
        // enoughInformationThreshold: 充足信息阈值，用于判断某方向是否有足够约束
        xicpParams.enoughInformationThreshold = config_.icp_params.XICP_ENOUGH_INFO_THRESHOLD;
        // insufficientInformationThreshold: 不足信息阈值，低于此值认为该方向退化
        xicpParams.insufficientInformationThreshold = config_.icp_params.XICP_INSUFFICIENT_INFO_THRESHOLD;
        // highInformationThreshold: 高信息阈值，表示该方向约束充分
        xicpParams.highInformationThreshold = config_.icp_params.XICP_HIGH_INFO_THRESHOLD;
        // solutionRemappingThreshold: 解重映射阈值，用于Solution Remapping方法
        xicpParams.solutionRemappingThreshold = config_.icp_params.XICP_SOLUTION_REMAPPING_THRESHOLD;
        // 点到法向量最小对齐角度的余弦值（角度转弧度）
        xicpParams.point2NormalMinimalAlignmentCosineThreshold =
                std::cos(config_.icp_params.XICP_MINIMAL_ALIGNMENT_ANGLE * M_PI / 180.0);
        // 点到法向量强对齐角度的余弦值
        xicpParams.point2NormalStrongAlignmentCosineThreshold =
                std::cos(config_.icp_params.XICP_STRONG_ALIGNMENT_ANGLE * M_PI / 180.0);
        // 不等式约束边界乘数，用于Inequality Constraints方法
        xicpParams.inequalityBoundMultiplier = config_.icp_params.XICP_INEQUALITY_BOUND_MULTIPLIER;
        // 启用内部打印，用于调试和监控
        xicpParams.isPrintingEnabled = true;


        // 如果启用打印，输出所有XICP参数
        if (xicpParams.isPrintingEnabled) {
            std::cout << "[XICP] Parameters:" << std::endl;
            std::cout << "  充足约束阈值: " << xicpParams.enoughInformationThreshold << std::endl;
            std::cout << "  不足约束阈值: " << xicpParams.insufficientInformationThreshold
                      << std::endl;
            std::cout << "  高约束阈值: " << xicpParams.highInformationThreshold << std::endl;
            std::cout << "  解重映射阈值: " << xicpParams.solutionRemappingThreshold << std::endl;
            std::cout << "  最小对齐角度: " << config_.icp_params.XICP_MINIMAL_ALIGNMENT_ANGLE << " deg"
                      << std::endl;
            std::cout << "  强对齐角度: " << config_.icp_params.XICP_STRONG_ALIGNMENT_ANGLE << " deg"
                      << std::endl;
            std::cout << "  不等式约束边界乘数: " << config_.icp_params.XICP_INEQUALITY_BOUND_MULTIPLIER
                      << " deg"
                      << std::endl;
        }

        // ========== 步骤2：根据检测方法设置退化感知策略 ==========
        std::cout << "\n[XICP] Starting XICP with:" << std::endl;
        std::cout << "  退化检测方法: ";
        switch (detection_method) {
            case DetectionMethod::XICP_OPTIMIZED_EQUALITY:
                // 优化等式约束方法：在优化中添加等式约束，效率较高
                xicpParams.degeneracyAwarenessMethod = XICP::DegeneracyAwarenessMethod::kOptimizedEqualityConstraints;
                break;
            case DetectionMethod::XICP_EQUALITY:
                // 标准等式约束方法
                xicpParams.degeneracyAwarenessMethod = XICP::DegeneracyAwarenessMethod::kEqualityConstraints;
                break;
            case DetectionMethod::XICP_INEQUALITY:
                // 不等式约束方法：使用不等式约束限制退化方向的移动
                xicpParams.degeneracyAwarenessMethod = XICP::DegeneracyAwarenessMethod::kInequalityConstraints;
                break;
            case DetectionMethod::XICP_SOLUTION_REMAPPING:
                // 解重映射方法：将解投影到可靠子空间
                xicpParams.degeneracyAwarenessMethod = XICP::DegeneracyAwarenessMethod::kSolutionRemapping;
                break;
            default:
                std::cerr << "[XICP] 无效的退化检测方法" << std::endl;
        }
        
        // ========== 步骤3：创建并初始化XICP核心对象 ==========
        XICP::XICPCore<double> xicpCore;
        xicpCore.setParameters(xicpParams);

        // ========== 步骤4：初始化ICP迭代变量 ==========
        // 将输出状态初始化为初始位姿
        output_state = initial_state;
        // 前一次迭代的误差，初始化为最大值
        double prev_error = std::numeric_limits<double>::max();
        // 点到平面误差平方和累加器
        double point_to_plane_error_sum = 0.0;
        // 有效对应点数量计数器
        int valid_correspondence_count = 0;
        // 收敛标志
        bool converged = false;

        // ========== 步骤5：验证上下文数据完整性 ==========
        // 确保context已经通过setTargetCloud设置了目标点云和法向量
        if (!context.targetCloud || !context.targetNormals) {
            std::cerr << "[XICP] 错误: 目标点云未设置在context中. 请先调用setTargetCloud方法." << std::endl;
            return false;
        }

        // ========== 步骤6：预分配内存以提高效率 ==========
        // 预分配对应关系容器的最大容量，避免迭代中频繁的内存重分配
        const size_t max_correspondences = measure_cloud->size();
        std::vector <Eigen::Vector3d> valid_src;        // 有效源点集合
        std::vector <Eigen::Vector3d> valid_tgt;        // 有效目标点集合
        std::vector <Eigen::Vector3d> valid_normals;    // 对应的法向量集合
        valid_src.reserve(max_correspondences);
        valid_tgt.reserve(max_correspondences);
        valid_normals.reserve(max_correspondences);

        // ========== 步骤7：初始化context中的数据结构 ==========
        // 利用context中预分配的数据结构，避免每次迭代重新分配
        if (context.laserCloudOriSurfVec.size() != measure_cloud->size()) {
            context.laserCloudOriSurfVec.resize(measure_cloud->size());     // 存储变换后的源点
            context.coeffSelSurfVec.resize(measure_cloud->size());          // 存储对应的法向量系数
            context.laserCloudOriSurfFlag.resize(measure_cloud->size());    // 存储有效性标志
        }

        // ========== 步骤8：ICP主迭代循环 ==========
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // 启动迭代计时器
            TicToc iter_timer;
            // 创建迭代日志数据对象，用于记录本次迭代的详细信息
            IterationLogData iter_data;
            iter_data.iter_count = iter;

            // ========== 步骤8.1：用当前位姿变换源点云 ==========
            // 将测量点云按照当前估计的output_state进行刚体变换
            pcl::transformPointCloud(*measure_cloud, *context.laserCloudEffective, output_state.matrix());

            // ========== 步骤8.2：清空上一次迭代的对应关系 ==========
            valid_src.clear();           // 清空有效源点
            valid_tgt.clear();           // 清空有效目标点
            valid_normals.clear();       // 清空对应法向量

            // ========== 步骤8.3：并行搜索最近邻点对 ==========
            // 重置所有点的有效性标志为0（无效）
            std::fill(context.laserCloudOriSurfFlag.begin(), context.laserCloudOriSurfFlag.end(), 0);

            // 使用OpenMP并行化最近邻搜索，8线程并行处理
#pragma omp parallel for num_threads(8)
            for (size_t i = 0; i < context.laserCloudEffective->size(); ++i) {
                // 为每个线程分配独立的索引和距离容器
                std::vector<int> k_indices(1);          // 存储最近邻点索引（k=1）
                std::vector<float> k_distances(1);      // 存储最近邻点距离的平方

                // 在目标点云的K-d树中搜索变换后源点的最近邻
                if (context.kdtreeSurfFromMap->nearestKSearch(context.laserCloudEffective->points[i],
                                                              1, k_indices, k_distances) > 0) {
                    // 检查距离是否在搜索半径内（注意k_distances存储的是平方距离）
                    if (k_distances[0] < SEARCH_RADIUS * SEARCH_RADIUS) {
                        // 标记为有效对应关系
                        context.laserCloudOriSurfFlag[i] = 1;

                        // 存储变换后的源点
                        context.laserCloudOriSurfVec[i] = context.laserCloudEffective->points[i];
                        // 存储目标点对应的法向量（从预计算的法向量点云中获取）
                        context.coeffSelSurfVec[i].x = context.targetNormals->points[k_indices[0]].normal_x;
                        context.coeffSelSurfVec[i].y = context.targetNormals->points[k_indices[0]].normal_y;
                        context.coeffSelSurfVec[i].z = context.targetNormals->points[k_indices[0]].normal_z;
                        // 利用intensity字段存储目标点在点云中的索引
                        context.coeffSelSurfVec[i].intensity = k_indices[0];
                    }

                }
            }

            // ========== 步骤8.4：收集有效对应关系并计算点到平面误差 ==========
            for (size_t i = 0; i < context.laserCloudOriSurfFlag.size(); ++i) {
                // 仅处理标记为有效的对应关系
                if (context.laserCloudOriSurfFlag[i]) {
                    // 获取变换后的源点
                    const PointT &src_pt = context.laserCloudOriSurfVec[i];
                    // 从intensity字段提取目标点索引
                    int tgt_idx = static_cast<int>(context.coeffSelSurfVec[i].intensity);
                    // 获取目标点
                    const PointT &tgt_pt = context.targetCloud->points[tgt_idx];

                    // 提取并归一化法向量
                    const PointT &normal_pt = context.coeffSelSurfVec[i];
                    Eigen::Vector3d normal(normal_pt.x, normal_pt.y, normal_pt.z);
                    normal.normalize();  // 确保法向量为单位向量

                    // 计算点到平面距离（有符号距离）
                    Eigen::Vector3d src_vec(src_pt.x, src_pt.y, src_pt.z);
                    Eigen::Vector3d tgt_vec(tgt_pt.x, tgt_pt.y, tgt_pt.z);
                    Eigen::Vector3d diff = src_vec - tgt_vec;
                    // 点到平面距离 = (p_src - p_tgt) · n
                    double point_to_plane_dist = diff.dot(normal);
                    // 累加平方误差用于计算RMSE
                    point_to_plane_error_sum += point_to_plane_dist * point_to_plane_dist;

                    // 将有效对应关系添加到容器中
                    // （注释的代码可用于基于距离的异常值过滤）
                    //  if (std::abs(point_to_plane_dist) < 0.5 * 0.5) {
                    valid_src.emplace_back(src_pt.x, src_pt.y, src_pt.z);
                    valid_tgt.emplace_back(tgt_pt.x, tgt_pt.y, tgt_pt.z);
                    valid_normals.push_back(normal);
                    valid_correspondence_count++;
                    //                    }
                }
            }

            // ========== 步骤8.5：计算本次迭代的误差指标 ==========
            // iter_data.corr_num = valid_src.size();
            iter_data.effective_points = valid_src.size();  // 有效对应点数量
            // 计算RMSE（均方根误差）
            iter_data.rmse = (valid_correspondence_count > 0) ?
                             std::sqrt(point_to_plane_error_sum / valid_correspondence_count) :
                             std::numeric_limits<double>::max();
            // 计算配准适应度（有效对应点比例）
            iter_data.fitness = valid_correspondence_count * 1.0 / max_correspondences;

            // ========== 步骤8.6：检查对应点数量是否足够 ==========
            // 如果有效对应点太少，配准无法继续
            if (valid_src.size() < 10) {
                std::cout << "[XICP] 有效对应点数量太少: " << valid_src.size() << std::endl;
                return false;
            }

            // ========== 步骤8.7：构建齐次坐标点矩阵 ==========
            // XICP算法需要4xN的齐次坐标矩阵（最后一行为1表示点，0表示向量）
            Eigen::MatrixXd sourcePoints(4, valid_src.size());
            Eigen::MatrixXd targetPoints(4, valid_tgt.size());
            Eigen::MatrixXd targetNormals(4, valid_normals.size());
            for (size_t i = 0; i < valid_src.size(); ++i) {
                sourcePoints.col(i) << valid_src[i], 1.0;      // 源点，齐次坐标最后为1
                targetPoints.col(i) << valid_tgt[i], 1.0;      // 目标点，齐次坐标最后为1
                targetNormals.col(i) << valid_normals[i], 0.0; // 法向量，齐次坐标最后为0
            }

            // ========== 步骤8.8：构建点到平面ICP的Hessian矩阵和约束向量 ==========
            // 初始化6x6 Hessian矩阵（近似二阶导数矩阵）和6x1约束向量
            Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> constraints = Eigen::Matrix<double, 6, 1>::Zero();

            {
                // 构建点到平面ICP的线性系统：J^T * W * J * dx = -J^T * W * r
                // 其中J是雅可比矩阵，W是权重矩阵，r是残差向量
                size_t numPoints = valid_src.size();

                // 计算交叉积矩阵：p_src × n，用于构建旋转部分的雅可比
                // 点到平面残差对旋转的导数为：∂r/∂ω = (p_src × n)
                Eigen::MatrixXd crosses(3, numPoints);
                for (size_t i = 0; i < numPoints; ++i) {
                    crosses.col(i) = valid_src[i].cross(valid_normals[i]);
                }

                // 构建特征矩阵 F (6xN)，即点到平面残差的雅可比矩阵
                // F的结构：前3行为旋转部分(p×n)，后3行为平移部分(n)
                Eigen::MatrixXd F(6, numPoints);
                Eigen::MatrixXd wF(6, numPoints);  // 加权版本 W^(1/2) * F

                // 初始化权重为1（可扩展为M-estimator等鲁棒权重）
                Eigen::VectorXd weights = Eigen::VectorXd::Ones(numPoints);
                for (size_t i = 0; i < numPoints; ++i) {
                    // 旋转部分（前3行）：∂r/∂ω = p_src × n
                    F.block(0, i, 3, 1) = crosses.col(i);
                    wF.block(0, i, 3, 1) = weights(i) * crosses.col(i);

                    // 平移部分（后3行）：∂r/∂t = n
                    F.block(3, i, 3, 1) = valid_normals[i];
                    wF.block(3, i, 3, 1) = weights(i) * valid_normals[i];
                }
                
                // 计算Hessian矩阵（高斯-牛顿近似）：H = J^T * W * J = wF * F^T
                // 这是一个6x6对称正定矩阵（在非退化情况下）
                hessian = wF * F.transpose();

                // 计算残差向量：r_i = (p_src_i - p_tgt_i) · n_i（点到平面的有符号距离）
                Eigen::VectorXd dotProd = Eigen::VectorXd::Zero(numPoints);
                for (size_t i = 0; i < numPoints; ++i) {
                    Eigen::Vector3d delta = valid_src[i] - valid_tgt[i];
                    dotProd(i) = delta.dot(valid_normals[i]);
                }
                
                // 计算约束向量（负梯度）：b = -J^T * W * r = -(wF * dotProd)
                constraints = -(wF * dotProd);

                // 计算当前目标函数值：E = 0.5 * r^T * W * r
                iter_data.objective_value = 0.5 * dotProd.transpose() * weights.asDiagonal() * dotProd;
                // 梯度为 J^T * W * r = -constraints
                // iter_data.gradient = -constraints;
            }


            // 第一次迭代时输出Hessian矩阵的条件数和约束向量范数，用于诊断
            if (iter == 0) {
                std::cout << "[XICP] Hessian矩阵的条件数: " <<
                          hessian.norm() / (hessian.inverse().norm() * hessian.norm()) << std::endl;
                std::cout << "[XICP] 约束向量范数: " << constraints.norm() << std::endl;
            }

            // ========== 步骤8.9前置：更新XICP参数中的当前变换矩阵 ==========
            // 设置当前位姿到优化坐标系的变换（用于某些XICP方法）
            xicpParams.transformationToOptimizationFrame = output_state.matrix();
            xicpCore.setParameters(xicpParams);  // 更新XICP核心对象的参数

            // ========== （可选）步骤8.9.1：计算World坐标系下的条件数和特征值 ==========
            // 注意：此部分代码已禁用(if(0))，仅用于调试和分析
            if (0) {
                // 使用奇异值分解(SVD)计算完整Hessian矩阵的条件数
                Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd_full(hessian,
                                                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
                iter_data.singular_values_full = svd_full.singularValues();  // 奇异值（降序排列）
                double max_sv = iter_data.singular_values_full.maxCoeff();   // 最大奇异值
                double min_sv = iter_data.singular_values_full.minCoeff();   // 最小奇异值
                // 条件数 = 最大奇异值 / 最小奇异值（避免除零）
                iter_data.cond_full_svd = (min_sv > 1e-10) ? max_sv / min_sv : 1e10;

                // 使用特征值分解(EVD)计算完整Hessian矩阵的条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_full(hessian);
                iter_data.eigenvalues_full = es_full.eigenvalues();  // 特征值（升序排列）
                // 条件数 = 最大特征值 / 最小特征值
                iter_data.cond_full = (iter_data.eigenvalues_full.minCoeff() > 1e-10) ?
                                      iter_data.eigenvalues_full.maxCoeff() / iter_data.eigenvalues_full.minCoeff()
                                                                                      : 1e10;


                // 提取旋转块和平移块（Hessian矩阵的对角块结构）
                Eigen::Matrix3d H_rr = hessian.block<3, 3>(0, 0);  // 旋转-旋转耦合块
                Eigen::Matrix3d H_tt = hessian.block<3, 3>(3, 3);  // 平移-平移耦合块

                // 计算旋转块的特征值和条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_rot(H_rr);
                iter_data.lambda_diag_rot = es_rot.eigenvalues();
                iter_data.rot_eigenvalues = iter_data.lambda_diag_rot;
                // 旋转块条件数
                iter_data.cond_diag_rot = iter_data.cond_full_evd_sub_rot = (iter_data.lambda_diag_rot.minCoeff() >
                                                                             1e-10) ?
                                                                            iter_data.lambda_diag_rot.maxCoeff() /
                                                                            iter_data.lambda_diag_rot.minCoeff() : 1e10;

                // 计算平移块的特征值和条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_trans(H_tt);
                iter_data.lambda_diag_trans = es_trans.eigenvalues();
                iter_data.trans_eigenvalues = iter_data.lambda_diag_trans;
                // 平移块条件数
                iter_data.cond_diag_trans = iter_data.cond_full_evd_sub_trans = (iter_data.lambda_diag_trans.minCoeff() >
                                                                                 1e-10) ?
                                                                                iter_data.lambda_diag_trans.maxCoeff() /
                                                                                iter_data.lambda_diag_trans.minCoeff()
                                                                                        : 1e10;
            }


            // ========== 步骤8.9：XICP退化检测 ==========
            // 调用XICP核心的退化检测函数，分析当前配准问题的可定位性
            XICP::LocalizabilityAnalysisResults<double> xicpResults;
            bool detectSuccess = xicpCore.detectDegeneracy(
                    sourcePoints, targetPoints, targetNormals, hessian, xicpResults);

            // 记录高贡献点数量（用于统计）
            iter_data.corr_num = xicpCore.getParameters().highlyContributingNumberOfPoints_rot;

            // 检查退化检测是否成功
            if (!detectSuccess) {
                std::cerr << "[XICP] 退化检测失败在第 " << iter << " 次迭代" << std::endl;
                return false;
            }

            // ========== 步骤8.10前置：记录退化信息 ==========
            // 初始化退化标志位和退化掩码
            iter_data.is_degenerate = 0;  // 使用位掩码表示退化方向
            iter_data.degenerate_mask = std::vector<bool>(6, false);  // 6个自由度的退化掩码
            
            // 在第一次迭代或调试模式时输出详细的退化检测结果
            if (iter == 0 || config_.icp_params.XICP_DEBUG) {
                std::cout << "[XICP] 退化检测结果:" << std::endl;
                std::cout << "  旋转退化: ";
                // 检查旋转自由度的退化情况（Roll, Pitch, Yaw）
                for (int i = 0; i < 3; ++i) {
                    if (xicpResults.localizabilityRpy_(i) ==
                        static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                        std::cout << "R" << i << " ";  // 输出退化的旋转轴
                        iter_data.is_degenerate |= (1 << i);  // 设置对应位
                        iter_data.degenerate_mask[i] = true;   // 标记为退化
                    }
                }
                std::cout << std::endl << "  平移退化: ";
                // 检查平移自由度的退化情况（X, Y, Z）
                for (int i = 0; i < 3; ++i) {
                    if (xicpResults.localizabilityXyz_(i) ==
                        static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                        std::cout << "T" << i << " ";  // 输出退化的平移轴
                        iter_data.is_degenerate |= (1 << (i + 3));  // 设置对应位（平移占高3位）
                        iter_data.degenerate_mask[i + 3] = true;     // 标记为退化
                    }
                }
                std::cout << std::endl;

                // 输出退化掩码（顺序：ωx, ωy, ωz, x, y, z）
                std::cout << "退化掩码 ωxωyωz xyz: ";
                for (int i = 0; i < iter_data.degenerate_mask.size(); ++i) {
                    std::cout << iter_data.degenerate_mask[i] << " ";
                }
                std::cout << std::endl;
            }

            // ========== 步骤8.10：求解优化问题（考虑退化约束）==========
            // 初始化位姿增量（6维：前3维为旋转增量，后3维为平移增量）
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            
            if (handling_method == HandlingMethod::XICP_CONSTRAINT) {
                // ========== 方法A：使用Ceres求解器处理带约束的优化问题 ==========
                // 在退化方向上添加等式或不等式约束，限制解的移动
                int diff_method = 0; // 选择求导方式：0=自动微分, 1=解析导数, 2=KKT直接求解
                
                if (diff_method == 0) {
                    // 使用Ceres自动微分（AutoDiff）版本
                    // AutoDiff通过模板元编程自动计算导数，精度高且灵活
                    xicpCore.solveDegenerateSystemWithCeresAutoDiff(valid_src, valid_tgt, valid_normals, xicpResults,
                                                                    delta, false);
                    // 备选：使用数值微分（设置第5个参数为true）
                    //  solveDegenerateSystemWithCeresAutoDiff(hessian, constraints, xicpResults, delta, true);
                    
                } else if (diff_method == 1) {
                    // 使用Ceres解析导数版本
                    // 直接使用预计算的Hessian矩阵和梯度，速度更快但灵活性稍差
                    xicpCore.solveDegenerateSystemWithCeres(hessian, constraints, xicpResults, delta);
                    
                } else if (diff_method == 2) {
                    // 使用KKT（Karush-Kuhn-Tucker）条件直接求解欠约束系统
                    // 将约束和目标函数合并为增广拉格朗日系统求解
                    std::cout << "使用KKT求解器直接求解欠约束系统" << std::endl;
                    xicpCore.solveDegenerateSystemWithCeresKKT(hessian, constraints, xicpResults, delta);
                    
                } else {
                    std::cout << "pls set your mitigation method!!!" << std::endl;
                }
                
            } else if (handling_method == HandlingMethod::XICP_PROJECTION) {
                // ========== 方法B：使用投影方法处理退化问题 ==========
                // 首先求解标准系统，然后将解投影到可靠子空间

                // 步骤B.1：使用截断SVD求解线性系统 H * delta = -g
                Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);

                // 设置奇异值阈值，过滤小奇异值以避免数值不稳定
                double singular_threshold = 1e-6;
                Eigen::Matrix<double, 6, 1> singular_values = svd.singularValues();
                Eigen::Matrix<double, 6, 1> inv_singular_values = Eigen::Matrix<double, 6, 1>::Zero();

                // 计算伪逆的奇异值（仅对大于阈值的奇异值取倒数）
                // std::cout << "[XICP] Singular values: " << singular_values.transpose() << std::endl;
                for (int i = 0; i < 6; ++i) {
                    if (singular_values(i) > singular_threshold) {
                        inv_singular_values(i) = 1.0 / singular_values(i);
                    }
                }
                // 使用SVD伪逆求解：delta = V * Σ^(-1) * U^T * (-g)
                delta = svd.matrixV() * inv_singular_values.asDiagonal() * svd.matrixU().transpose() * constraints;

                // 步骤B.2：根据不同的检测方法应用相应的投影策略
                if (detection_method == DetectionMethod::XICP_SOLUTION_REMAPPING) {
                    // ========== 策略1：Solution Remapping投影 ==========
                    // 使用XICP预计算的投影矩阵，将解映射到可靠子空间
                    // std::cout << "[XICP] Applying Solution Remapping projection matrix" << std::endl;
                    Eigen::Matrix<double, 6, 1> delta_before = delta;
                    delta = xicpResults.solutionRemappingProjectionMatrix_ * delta;
                    // 可选：输出投影效果
                    // std::cout << "[XICP] Projection matrix effect: " << (delta - delta_before).norm() << std::endl;
                    
                } else {
                    // ========== 策略2：基于退化方向的正交投影 ==========
                    // 对于Optimized Equality和Inequality方法，逐个投影退化方向
                    // std::cout << "[XICP] Applying directional projection for degenerate directions" << std::endl;
                    int num_projections = 0;
                    
                    for (int i = 0; i < 3; ++i) {
                        // 检查旋转自由度是否退化
                        if (xicpResults.localizabilityRpy_(i) ==
                            static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                            // 将旋转增量在该退化特征向量方向上的分量置零
                            // 即：delta_rot = delta_rot - (delta_rot · v_i) * v_i
                            Eigen::Vector3d rot_dir = xicpResults.rotationEigenvectors_.col(i);
                            double projection = delta.head<3>().dot(rot_dir);
                            delta.head<3>() -= projection * rot_dir;
                            // std::cout << "[XICP] Projected out rotation direction " << i 
                            //           << ", projection magnitude: " << std::abs(projection) << std::endl;
                            num_projections++;
                        }
                        
                        // 检查平移自由度是否退化
                        if (xicpResults.localizabilityXyz_(i) ==
                            static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                            // 将平移增量在该退化特征向量方向上的分量置零
                            Eigen::Vector3d trans_dir = xicpResults.translationEigenvectors_.col(i);
                            double projection = delta.tail<3>().dot(trans_dir);
                            delta.tail<3>() -= projection * trans_dir;
                            // std::cout << "[XICP] Projected out translation direction " << i
                            //           << ", projection magnitude: " << std::abs(projection) << std::endl;
                            num_projections++;
                        }
                    }
                }
            } else {
                std::cerr << "[XICP] Invalid handling method" << std::endl;
                return false;
            }

            // ========== 步骤8.11：使用左乘扰动更新位姿 ==========
            // XICP使用SO(3)×R³左乘扰动参数化：T_new = exp(delta^) * T_old
            // 其中delta^是6维李代数元素对应的SE(3)群元素
            output_state = output_state.boxplus_left(delta);
            // 保存更新后的变换矩阵到迭代日志
            iter_data.transform_matrix = output_state.matrix();
            // iter_data.update_dx = delta;  // 暂不记录（后面会转换到body系）

            // ========== 步骤8.12：计算位姿增量范数用于收敛判断 ==========
            double delta_norm = delta.norm();              // 总增量范数
            double deltaR_norm = delta.head<3>().norm();   // 旋转增量范数
            double deltaT_norm = delta.tail<3>().norm();   // 平移增量范数


            // ========== 步骤8.13：坐标系转换（World系 → Body系）==========
            // 重要说明：XICP使用左乘扰动（body系），而其他基线方法使用右乘扰动（world系）
            // 为了公平比较和统一记录，需要通过伴随变换将Hessian和梯度转换到body系
            
            // 计算当前位姿的伴随矩阵 Ad_T ∈ R^{6×6}
            Eigen::Matrix<double, 6, 6> Ad_T = output_state.Adjoint();
            
            // World坐标系下的梯度和Hessian（优化前的坐标系）
            Eigen::Matrix<double, 6, 1> gradient_world = -constraints;  // 梯度 = -右端项
            Eigen::Matrix<double, 6, 6> hessian_world = hessian;        // Hessian矩阵
            
            // 通过伴随变换转换到Body坐标系
            // 梯度变换：g_body = Ad_T^T * g_world
            Eigen::Matrix<double, 6, 1> gradient_body = Ad_T.transpose() * gradient_world;
            // Hessian变换：H_body = Ad_T^T * H_world * Ad_T
            Eigen::Matrix<double, 6, 6> hessian_body = Ad_T.transpose() * hessian_world * Ad_T;
            // 增量变换：delta_body = Ad_T^(-1) * delta_world
            Eigen::Matrix<double, 6, 1> delta_body = Ad_T.inverse() * delta;


            // ========== 步骤8.14：记录Body坐标系下的数据 ==========
            // 为了与其他使用右乘扰动的方法进行公平比较，统一记录body系数据
            iter_data.gradient = gradient_body;     // 记录body系梯度
            iter_data.update_dx = delta_body;       // 记录body系位姿增量
            double delta_norm_body = delta_body.norm();            // Body系总增量范数
            double deltaR_norm_body = delta_body.head<3>().norm(); // Body系旋转增量范数
            double deltaT_norm_body = delta_body.tail<3>().norm(); // Body系平移增量范数

            // 目标函数值已在步骤8.8中计算
            //  iter_data.objective_value = 0.5 * dotProd.transpose() * weights.asDiagonal() * dotProd;

            // （可选）调试模式下输出坐标系比较信息
            //  if (config_.icp_params.XICP_DEBUG && iter == 0) {
            //      std::cout << "[XICP] Coordinate system comparison:" << std::endl;
            //      std::cout << "  Gradient norm (world): " << gradient_world.norm() << std::endl;
            //      std::cout << "  Gradient norm (body): " << gradient_body.norm() << std::endl;
            //      std::cout << "  Hessian cond (world): "
            //                << hessian_world.norm() / (hessian_world.inverse().norm() * hessian_world.norm())
            //                << std::endl;
            //      std::cout << "  Hessian cond (body): "
            //                << hessian_body.norm() / (hessian_body.inverse().norm() * hessian_body.norm()) << std::endl;
            //  }

            // ========== 步骤8.15：计算Body系下的Hessian矩阵数值特性 ==========
            // 注意：XICP使用SO(3)×R³左乘扰动，因此Hessian本质上在body系
            // 其他基线方法使用右乘扰动，Hessian在world系
            // 若要统一绘图比较，需要通过伴随变换在两个坐标系间转换
            if (1) {
                // 使用SVD计算body系Hessian的条件数
                Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd_body(hessian_body,
                                                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
                iter_data.singular_values_full = svd_body.singularValues();  // 奇异值（6维）
                double max_sv = iter_data.singular_values_full.maxCoeff();   // 最大奇异值
                double min_sv = iter_data.singular_values_full.minCoeff();   // 最小奇异值
                iter_data.cond_full_svd = (min_sv > 1e-10) ? max_sv / min_sv : 1e10;  // 条件数

                // 使用EVD计算body系Hessian的特征值和条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_body(hessian_body);
                iter_data.eigenvalues_full = es_body.eigenvalues();  // 特征值（6维，升序）
                iter_data.cond_full = (iter_data.eigenvalues_full.minCoeff() > 1e-10) ?
                                      iter_data.eigenvalues_full.maxCoeff() / iter_data.eigenvalues_full.minCoeff()
                                                                                      : 1e10;

                // 提取body系下的旋转块和平移块
                Eigen::Matrix3d H_rr_body = hessian_body.block<3, 3>(0, 0);  // 旋转-旋转块
                Eigen::Matrix3d H_tt_body = hessian_body.block<3, 3>(3, 3);  // 平移-平移块

                // 计算旋转块的特征值和条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_rot_body(H_rr_body);
                iter_data.lambda_diag_rot = es_rot_body.eigenvalues();
                iter_data.rot_eigenvalues = iter_data.lambda_diag_rot;
                iter_data.cond_diag_rot = (iter_data.lambda_diag_rot.minCoeff() > 1e-10) ?
                                          iter_data.lambda_diag_rot.maxCoeff() / iter_data.lambda_diag_rot.minCoeff()
                                                                                         : 1e10;

                // 计算平移块的特征值和条件数
                Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> es_trans_body(H_tt_body);
                iter_data.lambda_diag_trans = es_trans_body.eigenvalues();
                iter_data.trans_eigenvalues = iter_data.lambda_diag_trans;
                iter_data.cond_diag_trans = (iter_data.lambda_diag_trans.minCoeff() > 1e-10) ?
                                            iter_data.lambda_diag_trans.maxCoeff() /
                                            iter_data.lambda_diag_trans.minCoeff() : 1e10;
            }


            // ========== 步骤8.16：检查收敛条件 ==========
            // 当旋转和平移增量都小于阈值时，认为算法收敛
            if (deltaR_norm < config_.CONVERGENCE_THRESH_ROT && deltaT_norm < config_.CONVERGENCE_THRESH_TRANS) {
                std::cout << "[XICP] 收敛在第 " << iter + 1 << " 次迭代"
                          << " 增量范数: " << std::fixed << std::setprecision(8) << delta_norm << std::endl;
                converged = true;
                context.final_convergence_flag_ = true;
                context.final_iterations_ = iter + 1;
                break;  // 跳出迭代循环
            }

            // ========== 步骤8.17：计算相对于真值的位姿误差 ==========
            // 如果提供了真值（ground truth），计算当前估计相对于真值的误差
            {
                PoseError error = calculatePoseError(config_.gt_matrix, iter_data.transform_matrix, true);
                iter_data.rot_error_vs_gt = error.rotation_error;     // 旋转误差（度）
                iter_data.trans_error_vs_gt = error.translation_error; // 平移误差（米）
            }

            // ========== 步骤8.18：记录本次迭代数据并继续下一次迭代 ==========
            iter_data.iter_time_ms = iter_timer.toc();  // 记录本次迭代耗时
            context.iteration_log_data_.push_back(iter_data);  // 添加到迭代日志
        }  // 迭代循环结束

        // ========== 步骤9：处理未收敛情况 ==========
        if (!converged) {
            std::cout << "\n[XICP] 达到最大迭代次数" << std::endl;
            context.final_iterations_ = MAX_ITERATIONS;
        }

        return true;  // 返回成功标志
    }


    bool TestRunner::runOpen3DICP(const std::string &method_name, TestResult &result) {
        std::cout << "Running Open3D ICP for method: " << method_name << std::endl;
        // 转换点云格式
        auto source_o3d = PclToO3d(*source_cloud_);
        auto target_o3d = PclToO3d(*target_cloud_);
        if (!source_o3d || source_o3d->points_.empty() || !target_o3d || target_o3d->points_.empty()) {
            std::cerr << "Error: Failed to convert PCL clouds to Open3D format." << std::endl;
            return false;
        }

        // 估计目标点云法线 (Point-to-Plane需要)
        TicToc timer;
        std::cout << "Estimating normals for target cloud (radius=" << config_.search_radius
                  << ", nn=" << config_.normal_nn << ")..." << std::endl;
        target_o3d->EstimateNormals(
                open3d::geometry::KDTreeSearchParamHybrid(config_.search_radius, config_.normal_nn));
        target_o3d->OrientNormalsTowardsCameraLocation(); // 统一法线方向
        if (!target_o3d->HasNormals()) {
            std::cerr << "Error: Failed to estimate normals for Open3D target cloud." << std::endl;
            return false;
        }

        // 设置初始变换
        open3d::pipelines::registration::ICPConvergenceCriteria criteria;
        criteria.max_iteration_ = config_.max_iterations;

        auto estimation = std::make_shared<open3d::pipelines::registration::TransformationEstimationPointToPlane>();

        std::cout << "Running Open3D Point-to-Plane ICP..." << std::endl;
        auto reg_result = open3d::pipelines::registration::RegistrationICP(
                *source_o3d, *target_o3d, config_.search_radius, config_.initial_matrix, *estimation, criteria);
        double icp_time_ms = timer.toc();

        // 提取结果
        result.time_ms = icp_time_ms;
        result.iterations = config_.max_iterations;
        result.final_transform = reg_result.transformation_;
        result.final_fitness = reg_result.fitness_;
        result.final_rmse = reg_result.inlier_rmse_;
        result.corr_num = reg_result.correspondence_set_.size();

        result.condition_numbers.clear();
        result.eigenvalues.clear();
        result.singular_values.clear();
        result.degenerate_mask.assign(6, false);

        return true;
    }


// Modified visualizeResults using Open3D
    void ICPRunner::TestRunner::visualizeResults(const pcl::PointCloud<PointT>::Ptr &aligned_cloud_pcl,
                                                 const pcl::PointCloud<PointT>::Ptr &target_cloud_pcl,
                                                 const std::string &method_name,
                                                 const ICPRunner::TestResult &result) { // Use namespaced TestResult
        std::cout << "Preparing Open3D visualization for " << method_name << "..." << std::endl;
        // Convert PCL PointClouds to Open3D PointClouds
        auto aligned_cloud_o3d = PclToO3d(*aligned_cloud_pcl);
        auto target_cloud_o3d = PclToO3d(*target_cloud_pcl);
        if (!aligned_cloud_o3d || aligned_cloud_o3d->IsEmpty()) {
            std::cerr
                    << "Error: Aligned PCL cloud could not be converted or is empty for Open3D visualization (Method: "
                    << method_name << ")." << std::endl;
        }
        if (!target_cloud_o3d || target_cloud_o3d->IsEmpty()) {
            std::cerr
                    << "Error: Target PCL cloud could not be converted or is empty for Open3D visualization (Method: "
                    << method_name << ")." << std::endl;
        }
        // Create an Open3D visualizer object
        open3d::visualization::Visualizer visualizer;
        std::string window_title = "Open3D ICP Results - " + method_name;
        if (!visualizer.CreateVisualizerWindow(window_title, 1024, 768)) { // Width, Height
            std::cerr << "Failed to create Open3D visualizer window for " << method_name << std::endl;
            return;
        }
        // Set rendering options - CORRECTED USAGE WITH public members
        visualizer.GetRenderOption().background_color_ = Eigen::Vector3d(1.0, 1.0, 1.0); // Dark grey background
        visualizer.GetRenderOption().point_size_ = 2.0; // Point size
        // Color the point clouds and add them if they are valid
        if (aligned_cloud_o3d && !aligned_cloud_o3d->IsEmpty()) {
            aligned_cloud_o3d->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0)); // Aligned (source) cloud in Red
            if (!visualizer.AddGeometry(aligned_cloud_o3d)) {
                std::cerr << "Failed to add aligned_cloud_o3d to visualizer for " << method_name << std::endl;
            }
        } else {
            std::cout << "Skipping aligned cloud in visualization as it's empty or invalid for " << method_name
                      << "."
                      << std::endl;
        }
        if (target_cloud_o3d && !target_cloud_o3d->IsEmpty()) {
            target_cloud_o3d->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0)); // Target cloud in Green
            if (!visualizer.AddGeometry(target_cloud_o3d)) {
                std::cerr << "Failed to add target_cloud_o3d to visualizer for " << method_name << std::endl;
            }
        } else {
            std::cout << "Skipping target cloud in visualization as it's empty or invalid for " << method_name
                      << "."
                      << std::endl;
        }
        // Add a coordinate system (optional, but helpful)
        auto coord_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, Eigen::Vector3d(0, 0,
                                                                                                      0)); // Size 1.0, origin 0,0,0
        if (!visualizer.AddGeometry(coord_frame)) {
            std::cerr << "Failed to add coordinate frame to visualizer for " << method_name << std::endl;
        }
        // Print text information to the console
//        std::cout << "\n--- Visualization Details for " << method_name << " ---" << std::endl;
//        std::cout << "Method: " << method_name << std::endl;
//        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
//        std::cout << "Iterations: " << result.iterations << std::endl;
//        std::cout << "Time: " << std::fixed << std::setprecision(2) << result.time_ms << " ms" << std::endl;
//        std::cout << "Trans Error: " << std::fixed << std::setprecision(4) << result.trans_error_m << " m" << std::endl;
//        std::cout << "Rot Error: " << std::fixed << std::setprecision(2) << result.rot_error_deg << " deg" << std::endl;
//        std::cout << "RMSE: " << std::fixed << std::setprecision(4) << result.final_rmse << std::endl;
//        std::cout << "Fitness: " << std::fixed << std::setprecision(3) << result.final_fitness << std::endl;
//        std::cout << "--- End of Details ---" << std::endl;
//        std::cout << "\nDisplaying Open3D window for " << method_name << ". Close the window to continue..."
//        << std::endl;
        // Run the visualizer (this is a blocking call)
        visualizer.Run();
        // Destroy the visualizer window
        visualizer.DestroyVisualizerWindow();
        std::cout << "Open3D visualization window closed for " << method_name << "." << std::endl;
    }
} // namespace ICPRunner
