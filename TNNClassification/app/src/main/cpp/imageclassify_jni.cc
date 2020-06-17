#include "imageclassify_jni.h"
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include <android/bitmap.h>
#include "tnn/core/common.h"
#include <cmath>


std::shared_ptr<TNN_NS::TNN> net_ = nullptr;
std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
TNN_NS::DeviceType device_type_ = TNN_NS::DEVICE_ARM;


JNIEXPORT JNICALL jint
TNN_CLASSIFY(init)(JNIEnv *env, jobject thiz, jstring modelPath, jstring protoPath, jint computeUnitType) {
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    std::string protoPathStr(jstring2string(env, protoPath));
    protoContent = fdLoadFile(protoPathStr);
    modelContent = fdLoadFile(modelPathStr);

    TNN_NS::Status status;
    TNN_NS::ModelConfig config;
    config.model_type = TNN_NS::MODEL_TYPE_TNN;
    config.params = {protoContent, modelContent};
    auto net = std::make_shared<TNN_NS::TNN>();
    status = net->Init(config);
    net_ = net;

    device_type_ = TNN_NS::DEVICE_ARM;
    if (computeUnitType >= 1) {
        device_type_ = TNN_NS::DEVICE_OPENCL;
    }

    TNN_NS::InputShapesMap shapeMap;
    TNN_NS::NetworkConfig network_config;
    network_config.library_path = {""};
    network_config.device_type = device_type_;
    auto instance = net_->CreateInst(network_config, status, shapeMap);
    if (status != TNN_NS::TNN_OK || !instance) {
        // 如何出现GPU加载失败，自动切换CPU
        network_config.device_type = TNN_NS::DEVICE_ARM;
        instance = net_->CreateInst(network_config, status, shapeMap);
    }
    instance_ = instance;

    if (status != TNN_NS::TNN_OK) {
        LOGE("TNN init failed %d", (int) status);
        return -1;
    }
    return 0;
}

JNIEXPORT JNICALL jfloatArray
TNN_CLASSIFY(predict)(JNIEnv *env, jobject thiz, jobject imageSource, jint width,
                              jint height) {
    AndroidBitmapInfo sourceInfocolor;
    void *sourcePixelscolor;

    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        return nullptr;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return nullptr;
    }

    if (AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        return nullptr;
    }

    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims,
                                                   sourcePixelscolor);
    // step 1. set input mat
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    auto status = instance_->SetInputMat(input_mat, input_cvt_param);

    // step 2. Forward
    status = instance_->ForwardAsync(nullptr);

    // step 3. get output mat
    std::shared_ptr<TNN_NS::Mat> output_mat_scores = nullptr;
    status = instance_->GetOutputMat(output_mat_scores);

    if (status != TNN_NS::TNN_OK) {
        return nullptr;
    }

    // 返回预测结果
    auto *scores_data = (float *)output_mat_scores->GetData();
//    LOGE("length: %d", output_mat_scores->GetChannel());
//    for (int i = 1; i < output_mat_scores->GetChannel(); ++i) {
//        LOGE("score: %f", scores_data[i]);
//    }

    jfloatArray result;
    result = env->NewFloatArray(output_mat_scores->GetChannel());
    env->SetFloatArrayRegion(result, 0, output_mat_scores->GetChannel(), scores_data);
    return result;
}


JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thiz) {
    net_ = nullptr;
    instance_ = nullptr;
    return 0;
}
