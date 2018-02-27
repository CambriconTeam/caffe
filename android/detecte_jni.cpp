#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <cblas.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "caffe_detection.hpp"

#ifdef __cplusplus
extern "C" {
#endif

using std::string;
using std::vector;
using caffe::Detection;
typedef unsigned char byte;

int getTimeSec() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int)now.tv_sec;
}

string jstring2string(JNIEnv *env, jstring jstr) {
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string str(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  return str;
}

/**
 * NOTE: byte[] buf = str.getBytes("US-ASCII")
 */
string bytes2string(JNIEnv *env, jbyteArray buf) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  string s((char *)ptr, env->GetArrayLength(buf));
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return s;
}

cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
  cv::cvtColor(img, img, CV_YUV2RGBA_NV21);
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return img;
}

byte * matToBytes(cv::Mat image)
{
   int size = image.total() * image.elemSize();
   byte * bytes = new byte[size];  // you will have to delete[] that later
   std::memcpy(bytes,image.data,size * sizeof(byte));
   return bytes;
}


cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
  return (width == 0 && height == 0) ? cv::imread(bytes2string(env, buf), -1)
                                     : imgbuf2mat(env, buf, width, height);
}

//huangyaling

JNIEXPORT void JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeDetection_setNumThreads(JNIEnv *env,
                                                             jobject thiz,
                                                             jint numThreads) {
  LOG(INFO) << "detect setNumThreads";
  int num_threads = numThreads;
  openblas_set_num_threads(num_threads);
}

JNIEXPORT void JNICALL Java_com_cambricon_productdisplay_caffenative_CaffeDetection_enableLog(
    JNIEnv *env, jobject thiz, jboolean enabled) {}

JNIEXPORT jint JNICALL Java_com_cambricon_productdisplay_caffenative_CaffeDetection_loadModel(
    JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
  LOG(INFO) << "detect loadModel";
  Detection::Get(jstring2string(env, modelPath),
                   jstring2string(env, weightsPath));
  return 0;
}

JNIEXPORT void JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeDetection_setMeanWithMeanFile(
    JNIEnv *env, jobject thiz, jstring meanFile) {
  Detection *caffe_detecte = Detection::Get();
  caffe_detecte->SetMean(jstring2string(env, meanFile));
}

JNIEXPORT void JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeDetection_setMeanWithMeanValues(
    JNIEnv *env, jobject thiz, jfloatArray meanValues) {
  Detection *caffe_detecte = Detection::Get();
  int num_channels = env->GetArrayLength(meanValues);
  jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
  vector<float> mean_values(ptr, ptr + num_channels);
  caffe_detecte->SetMean(mean_values);
}

JNIEXPORT void JNICALL Java_com_cambricon_productdisplay_caffenative_CaffeDetection_setScale(
    JNIEnv *env, jobject thiz, jfloat scale) {
  Detection *caffe_detecte = Detection::Get();
  caffe_detecte->SetScale(scale);
}


/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jbyteArray JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeDetection_detectImage(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height) {
  LOG(INFO) << "detect Image";
  Detection *caffe_detecte = Detection::Get();

  //matToBytes(caffe_detecte->show(getImage(env, buf, width, height)));
  cv::Mat image=caffe_detecte->show(getImage(env, buf, width, height));
  int size = image.total() * image.elemSize();
  jbyte * bytes = new jbyte[size];  // you will have to delete[] that later
  std::memcpy(bytes,image.data,size * sizeof(jbyte));

  jbyteArray resultByteArray = env->NewByteArray(size);
  env->SetByteArrayRegion(resultByteArray, 0, size, bytes);
  //env->ReleaseByteArrayElements(env, thiz, 0);
  LOG(INFO)<<"detect image end:"<<bytes;
  return resultByteArray;
}
//huangyaling

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

  FLAGS_redirecttologcat = true;
  FLAGS_android_logcat_tag = "caffe_jni";

  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
