#pragma once

#include "json.hpp"
#include <opencv2/opencv.hpp>
#include <paddle_api.h>
#include <sys/time.h>
#include <cstdio>
#include <functional>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cstdlib>
#include <memory>
#include <stdlib.h>
#include "common.hpp"
#include <opencv2/imgcodecs.hpp>
#include <paddle_image_preprocess.h>

using namespace std;
using namespace cv;
using namespace paddle::lite_api;

/**
 * @brief 目标检测结果
 *
 */
struct PredictResult
{
    int type;          // ID
    std::string label; // 标签
    float score;       // 置信度
    int x;             // 坐标
    int y;             // 坐标
    int width;         // 尺寸
    int height;        // 尺寸
};

/**
 * @brief 目标检测类
 *
 */
class Detection
{
public:
    std::vector<PredictResult> results; // AI推理结果
    float score = 0.5;                  // AI检测置信度
    Detection(const std::string pathModel)
    {
        // 检查模型文件是否存在
        labels.clear(); // 清空容器标签
        std::ifstream labelName(pathModel + "/label_list.txt");
        std::ifstream modelName(pathModel + "/mobilenet-ssd-model");
        std::ifstream paramsName(pathModel + "/mobilenet-ssd-params");
        if (!modelName.good() && !paramsName.good())
        {
            std::cout << "[Error]: ModelFile mobilenet-ssd-model not exit, Please Check your model " << std::endl;
            exit(-1);
        }
        modelName.close();  // 关闭文件
        paramsName.close(); // 关闭文件

        // 模型初始化
        std::vector<Place> valid_places({
            // 设置 valid places
            Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
            Place{TARGET(kHost), PRECISION(kFloat)},
            Place{TARGET(kARM), PRECISION(kFloat)},
        });
        paddle::lite_api::CxxConfig config;                         // 初始化paddlelite模型接口
        config.set_model_file(pathModel + "/mobilenet-ssd-model");  // 加载模型结构
        config.set_param_file(pathModel + "/mobilenet-ssd-params"); // 加载模型参数
        config.set_valid_places(valid_places);                      // 设置 valid places

        _predictor = paddle::lite_api::CreatePaddlePredictor(config); // 根据 Config 创建 Predictor

        if (!_predictor) // 判断是否模型初始化成功
        {
            std::cout << "[Error]: CreatePaddlePredictor Failed." << std::endl;
            exit(-1);
        }
        std::cout << "Predictor Init Success !!!" << std::endl;

        // 模型标签加载
        if (labelName.is_open())
        {
            std::string line;
            while (getline(labelName, line)) // 按行读取标签文件
            {
                labels.push_back(line); // 将标签按行存入容器
            }
            labelName.close(); // 关闭文件
        }
        else
        {
            std::cout << "[Error]: Open Lable File failed: " << pathModel + "/label_list.txt" << std::endl;
        }
    };

    /**
     * @brief AI图像前处理
     *
     * @param img
     * @param tensor
     */
    inline void fpgaPreprocess(cv::Mat img, std::unique_ptr<Tensor> &tensor)
    {

        int width = img.cols;
        int height = img.rows;
        if (height > 1080)
        {
            float fx = img.cols / 300;
            float fy = img.rows / 300;
            Mat resize_mat;
            resize(img, resize_mat, Size(300, 300), fx, fy);
            img = resize_mat;
            height = img.rows;
            width = img.cols;
        }
        uint8_t *src = (uint8_t *)malloc(3 * width * height);
        if (img.isContinuous())
        {
            memcpy(src, img.data, 3 * width * height * sizeof(uint8_t));
        }
        else
        {
            uint8_t *img_data = img.data;
            for (int i = 0; i < img.rows; ++i)
            {
                src = src + i * (width * 3);
                img_data = img_data + i * (width * 3);
                memcpy(src, img_data, width * 3 * sizeof(uint8_t));
            }
        }
        TransParam tparam;
        tparam.ih = img.rows;
        tparam.iw = img.cols;
        tparam.oh = 300;
        tparam.ow = 300;
        Preprocess preprocess(
            ImageFormat::BGR,
            "RGB" == "RGB" ? ImageFormat::RGB : ImageFormat::BGR, tparam);
        float means[] = {127.5, 127.5, 127.5};
        float scales[] = {0.007843, 0.007843, 0.007843};
        preprocess.image_to_tensor(src, tensor.get(), LayoutType::kNHWC, means, scales);
        free(src);
    }

    /**
     * @brief AI推理
     *
     * @param inputFrame
     */
    void inference(cv::Mat &inputFrame)
    {
        results.clear();
        auto input = _predictor->GetInput(0);
        input->Resize({1, 3, 300, 300});
        fpgaPreprocess(inputFrame, input);
        _predictor->Run();
        auto output = _predictor->GetOutput(0);
        float *result_data = output->mutable_data<float>();
        int size = output->shape()[0];
        for (int i = 0; i < size; i++)
        {
            float *data = result_data + i * 6;
            float scoreNow = data[1];
            if (scoreNow < score)
            {
                continue;
            }
            PredictResult result;
            result.type = (int)data[0];
            if (result.type < labels.size())
                result.label = labels[result.type];
            result.score = scoreNow;
            result.x = data[2] * inputFrame.cols;
            result.y = data[3] * inputFrame.rows;
            result.width = data[4] * inputFrame.cols - result.x;
            result.height = data[5] * inputFrame.rows - result.y;
            results.push_back(result);
        }
    };

    /**
     * @brief 图像绘制检测框信息
     *
     * @param img
     */
    void drawBox(Mat &img)
    {
        for (int i = 0; i < results.size(); i++)
        {
            PredictResult result = results[i];
            boundaryCorrection(result, img.cols, img.rows);
            auto score = std::to_string(result.score);
            int pointY = result.y - 20;
            if (pointY < 0)
                pointY = 0;
            cv::Rect rectText(result.x, pointY, result.width, 20);
            cv::rectangle(img, rectText, getCvcolor(result.type), -1);
            std::string label_name = result.label + " [" + score.substr(0, score.find(".") + 3) + "]";
            cv::Rect rect(result.x, result.y, result.width, result.height);
            cv::rectangle(img, rect, getCvcolor(result.type), 1);
            cv::putText(img, label_name, Point(result.x, result.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 254), 1);
        }
    }

    /**
     * @brief 获取Opencv颜色
     *
     * @param index 序号
     * @return cv::Scalar
     */
    cv::Scalar getCvcolor(int index)
    {
        switch (index)
        {
        case 1:
            return cv::Scalar(0, 255, 0); // 绿
            break;
        case 2:
            return cv::Scalar(255, 255, 0); // 天空蓝
            break;
        case 3:
            return cv::Scalar(0, 0, 255); // 大红
            break;
        case 4:
            return cv::Scalar(0, 250, 250); // 大黄
            break;
        case 5:
            return cv::Scalar(250, 0, 250); // 粉色
            break;
        case 6:
            return cv::Scalar(0, 102, 255); // 橙黄
            break;
        case 7:
            return cv::Scalar(255, 0, 0); // 深蓝
            break;
        case 8:
            return cv::Scalar(255, 255, 255); // 大白
            break;
        case 9:
            return cv::Scalar(247, 43, 113);
            break;
        case 10:
            return cv::Scalar(40, 241, 245);
            break;
        case 11:
            return cv::Scalar(237, 226, 19);
            break;
        case 12:
            return cv::Scalar(245, 117, 233);
            break;
        case 13:
            return cv::Scalar(55, 13, 19);
            break;
        case 14:
            return cv::Scalar(255, 255, 255);
            break;
        case 15:
            return cv::Scalar(237, 226, 19);
            break;
        case 16:
            return cv::Scalar(0, 255, 0);
            break;
        default:
            return cv::Scalar(255, 0, 0);
            break;
        }
    }

private:
    void boundaryCorrection(PredictResult &r, int width_range,
                            int height_range)
    {
#define MARGIN_PIXELS (2)
        r.width = (r.width > (width_range - r.x - MARGIN_PIXELS))
                      ? (width_range - r.x - MARGIN_PIXELS)
                      : r.width;
        r.height = (r.height > (height_range - r.y - MARGIN_PIXELS))
                       ? (height_range - r.y - MARGIN_PIXELS)
                       : r.height;

        r.x = (r.x < MARGIN_PIXELS) ? MARGIN_PIXELS : r.x;
        r.y = (r.y < MARGIN_PIXELS) ? MARGIN_PIXELS : r.y;
    }
    std::vector<std::string> labels;
    // predictor
    std::shared_ptr<PaddlePredictor> _predictor;
    typedef paddle::lite_api::Tensor Tensor;
    typedef paddle::lite_api::DataLayoutType LayoutType;
    typedef paddle::lite::utils::cv::FlipParam FlipParam;
    typedef paddle::lite::utils::cv::TransParam TransParam;
    typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
    typedef paddle::lite::utils::cv::ImagePreprocess Preprocess;
};
