#include "icp_test_runner.h"


namespace fs = std::filesystem;

/**
 * [功能描述]：ICP测试套件的主函数，用于执行多种ICP算法的性能测试和比较。
 * @param argc：命令行参数数量（当前版本未使用）
 * @param argv：命令行参数数组（当前版本未使用）
 * @return 返回程序执行状态：0表示成功，1表示失败
 */
int main(int argc, char **argv) {

    // 配置文件路径设置 - 可选择不同的配置文件用于不同的测试场景
    //    std::string config_file = "../config/icp_pk01.yaml"; // 真实世界数据配置，用于规划器退化情况测试
    // std::string config_file = "../config/icp.yaml";         // 基础ICP配置文件
    // std::string config_file = "../config/icp_iter.yaml";    // 迭代配置文件
    std::string config_file = "../config/icp.yaml";            // 当前使用的配置文件



    // 声明配置对象，用于存储从YAML文件读取的配置参数
    ICPRunner::Config config;
    // 加载配置文件，如果加载失败则退出程序
    if (!ICPRunner::loadConfig(config_file, config)) {
        std::cerr << "Failed to load configuration from: " << config_file << std::endl;
        return 1;  // 配置加载失败，返回错误码
    }

    // 创建输出目录，确保结果文件有存储位置
    fs::create_directories(config.output_folder);

    // 使用加载的配置初始化测试运行器
    ICPRunner::TestRunner runner(config);

    // 打印测试开始信息，显示测试方法数量和运行次数
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting ICP Test Suite" << std::endl;
    std::cout << "Number of methods: " << config.test_methods.size() << std::endl;  // 显示配置的测试方法数量
    std::cout << "Number of runs per method: " << config.num_runs << std::endl;      // 显示每个方法的运行次数
    std::cout << "========================================\n" << std::endl;

    // 记录测试开始时间，用于计算总执行时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行所有配置的ICP测试方法
    if (!runner.runAllTests()) {
        std::cerr << "Test execution failed!" << std::endl;
        return 1;  // 测试执行失败，返回错误码
    }

    // 记录测试结束时间并计算总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // 打印测试完成信息，包括总耗时和结果保存路径
    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;      // 显示总执行时间（秒）
    std::cout << "Results saved to: " << config.output_folder << std::endl;          // 显示结果保存目录
    std::cout << "========================================" << std::endl;

    return 0;  // 程序成功执行完毕
}