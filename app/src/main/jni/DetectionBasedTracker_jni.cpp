#include <DetectionBasedTracker_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <android/log.h>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace cv;
using namespace cv::face;
using namespace std;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}
void saveImage(Mat image, const char * path){
    IplImage img = image;
    int res = 0;//cvSaveImage("/sdcard/selcuk.pgm", &img);
    bool written = imwrite(path, image);
    LOGD("------------------------------------------- saved: %d written: %d", res, written);
}


jint lbpRecognize(vector<Mat> images, vector<int> labels, Mat mat){

    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::LBPHFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an LBPH model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    //
    // The LBPHFaceRecognizer uses Extended Local Binary Patterns
    // (it's probably configurable with other operators at a later
    // point), and has the following default values
    //
    //      radius = 1
    //      neighbors = 8
    //      grid_x = 8
    //      grid_y = 8
    //
    // So if you want a LBPH FaceRecognizer using a radius of
    // 2 and 16 neighbors, call the factory method with:
    //
    //      cv::createLBPHFaceRecognizer(2, 16);
    //
    // And if you want a threshold (e.g. 123.0) call it with its default values:
    //
    //      cv::createLBPHFaceRecognizer(1,8,8,8,123.0)
    //
    //Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    // First we'll use it to set the threshold of the LBPHFaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    model->setThreshold(0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    predictedLabel = model->predict(testSample);
    cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    cout << "Model Information:" << endl;
    string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
                               model->getRadius(),
                               model->getNeighbors(),
                               model->getGridX(),
                               model->getGridY(),
                               model->getThreshold());
    cout << model_info << endl;
    // We could get the histograms for example:
    vector<Mat> histograms = model->getHistograms();
    // But should I really visualize it? Probably the length is interesting:
    cout << "Size of the histograms: " << histograms[0].total() << endl;
    return 0;
};

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
    {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)", scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width, maxObjSize.height);
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter()
    {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator
{
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;
    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter>& _mainDetector, cv::Ptr<CascadeDetectorAdapter>& _trackingDetector):
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector)
    {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};


JNIEXPORT jlong JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeCreateObject
(JNIEnv * jenv, jclass, jstring jFileName, jint faceSize)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeCreateObject enter");
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeCreateObject");

    try
    {
        cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
            makePtr<CascadeClassifier>(stdFileName));
        cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
            makePtr<CascadeClassifier>(stdFileName));
        result = (jlong)new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0)
        {
            mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
        catch (...)
        {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDestroyObject
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDestroyObject");

    try
    {
        if(thiz != 0)
        {
            ((DetectorAgregator*)thiz)->tracker->stop();
            delete (DetectorAgregator*)thiz;
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStart
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStart");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->run();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStop
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStop");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->stop();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeSetFaceSize
(JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeSetFaceSize -- BEGIN");

    try
    {
        if (faceSize > 0)
        {
            ((DetectorAgregator*)thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeSetFaceSize -- END");
}


JNIEXPORT void JNICALL Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDetect
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDetect");

    try
    {
        vector<Rect> RectFaces;
        ((DetectorAgregator*)thiz)->tracker->process(*((Mat*)imageGray));
        ((DetectorAgregator*)thiz)->tracker->getObjects(RectFaces);
        *((Mat*)faces) = Mat(RectFaces, true);
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDetect END");
}



JNIEXPORT jint JNICALL Java_project_illnino_com_facerec_FaceDetectActivity_faceRecognition
        (JNIEnv * jenv, jclass jclazz, jlong sample, jint size)
{


    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_faceRecognition enter");
    try
    {
        jclass clazz = jenv->FindClass("project/illnino/com/facerec/FaceDetectActivity");
        jmethodID getFaceFolder = jenv->GetStaticMethodID(clazz, "getFaceFolder","(I)Ljava/lang/String;");
        jmethodID getPersonID = jenv->GetStaticMethodID(clazz, "getPersonID","(I)J");

        vector<Mat> images;
        vector<int> labels;
        for(int i = 0; i<size; i++){
            jstring folderName = (jstring)jenv->CallStaticObjectMethod(clazz, getFaceFolder, (jint)i);
            jboolean isCopy;
            const char *savePath = jenv->GetStringUTFChars(folderName, &isCopy);
            jlong id = jenv->CallStaticLongMethod(clazz, getPersonID, (jint)i);
            std::ostringstream stringstream;
            for(int j = 0; j < 10; j++) {
                stringstream << savePath;
                stringstream << "/";
                stringstream << j;
                stringstream << ".jpg";
                Mat greymat, colormat;
                colormat = imread(stringstream.str().c_str());
                if(colormat.data) {
                    LOGD("---------------------------------------------- image Loaded");
                    cvtColor(colormat, greymat, CV_BGR2GRAY);

                    images.push_back(greymat);
                    labels.push_back((int)id);
                }
                stringstream.clear();

            }
        }
        vector<Mat> histograms;
        jint lbpResult = lbpRecognize(images, labels, *((Mat*)sample));

        /*jint result = fisherRecognize(images, labels, *((Mat*)sample));
        if(result != lbpResult){
            result = -2;
        }*/
        return lbpResult;

    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
        return -1;
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
        return - 1;
    }
    LOGD("Java_project_illnino_com_facerec_DetectionBasedTracker_nativeDetect exit");
    return - 1;
}

JNIEXPORT void JNICALL Java_project_illnino_com_facerec_utils_ImageUtils_saveImageAsPGM
        (JNIEnv * jenv, jclass jclazz, jstring path, jlong imageReference)
{
    jboolean isCopy;
    const char *savePath = jenv->GetStringUTFChars(path, &isCopy);
    Mat image = *((Mat*)imageReference);
    saveImage(image, savePath);


}
