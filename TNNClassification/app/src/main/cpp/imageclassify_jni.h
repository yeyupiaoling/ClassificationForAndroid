#ifndef ANDROID_IMAGECLASSIFY_JNI_H
#define ANDROID_IMAGECLASSIFY_JNI_H

#include <jni.h>
#include <string>
#include <fstream>

#define TNN_CLASSIFY(sig) Java_com_yeyupiaoling_tnnclassification_tnn_ImageClassifyUtil_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint
TNN_CLASSIFY(init)(JNIEnv *env, jobject thiz, jstring modelPath, jstring protoPath,
                   jint computeUnitType);
JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jfloatArray
TNN_CLASSIFY(predict)(JNIEnv *env, jobject thiz, jobject imageSource, jint width,
                      jint height);
#ifdef __cplusplus
std::string fdLoadFile(std::string path) {
    std::ifstream file(path, std::ios::in);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size = file.tellg();
        char *content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

char *jstring2string(JNIEnv *env, jstring jstr) {
    char *rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte *ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char *) malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}
}
#endif
#endif
