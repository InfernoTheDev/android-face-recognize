package project.illnino.com.facerec.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.util.Log;

import project.illnino.com.facerec.DetectionBasedTracker;
import project.illnino.com.facerec.R;
import project.illnino.com.facerec.facedatabase.FacesDataSource;


public class FaceDetectionUtils {
	private static final String TAG = FaceDetectionUtils.class.getName();
    private static File mCascadeFile;
    private static File mCascadeLeftEyeFile;
    private static File mCascadeRightEyeFile;
    public static DetectionBasedTracker mNativeDetector;
    public static DetectionBasedTracker  mNativeLeftEyeDetector;
    public static DetectionBasedTracker  mNativeRightEyeDetector;
    public static boolean cascadeFilesLoaded = false;
	
	public static BaseLoaderCallback  mLoaderCallback; 
	private static Context mContext;
	public static FacesDataSource faceDataSource;
	
    public static void initialize(Context context){
    	mContext = context;
    	faceDataSource = new FacesDataSource(context);
		mLoaderCallback = new BaseLoaderCallback(mContext) {
			@Override
			public void onManagerConnected(int status) {
				switch (status) {
					case LoaderCallbackInterface.SUCCESS:
					{
						// Load native library after(!) OpenCV initialization
						System.loadLibrary("opencv_java3");
						System.loadLibrary("detection_based_tracker");
						loadCascadeFiles();
						cascadeFilesLoaded = true;

					} break;
					default:
					{
						super.onManagerConnected(status);
					} break;
				}
			}
		};

		if(!OpenCVLoader.initDebug()){
			Log.d(TAG, "OpenCV not loaded");
			OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, context.getApplicationContext(), mLoaderCallback);
		} else {
			Log.d(TAG, "OpenCV loaded");
			mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
		}

    }

	private static void loadCascadeFiles(){
		try{
	        InputStream is = mContext.getResources().openRawResource(R.raw.lbpcascade_frontalface);
	        File cascadeDir = mContext.getDir("cascade", Context.MODE_PRIVATE);
	        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
	        FileOutputStream os = new FileOutputStream(mCascadeFile);
	
	        byte[] buffer = new byte[4096];
	        int bytesRead;
	        while ((bytesRead = is.read(buffer)) != -1) {
	            os.write(buffer, 0, bytesRead);
	        }
	        is.close();
	        os.close();
	
	        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0, true);
	
	        cascadeDir.delete();
	
	        ////////////////////////////////////////////////////
	        InputStream isLeftEye = mContext.getResources().openRawResource(R.raw.ojoleft);
	        File cascadeDirLeftEye = mContext.getDir("cascadelefteye", Context.MODE_PRIVATE);
	        mCascadeLeftEyeFile = new File(cascadeDirLeftEye, "left_eye_detect.xml");
	        FileOutputStream osLeftEye = new FileOutputStream(mCascadeLeftEyeFile);
	        
	        byte[] bufferLeftEye = new byte[4096];
	        int bytesReadLeftEye;
	        while ((bytesReadLeftEye = isLeftEye.read(bufferLeftEye)) != -1) {
	        	osLeftEye.write(bufferLeftEye, 0, bytesReadLeftEye);
	        }
	        isLeftEye.close();
	        osLeftEye.close();
	        
	        mNativeLeftEyeDetector = new DetectionBasedTracker(mCascadeLeftEyeFile.getAbsolutePath(), 0, false);
	        
	        cascadeDirLeftEye.delete();
	        /////////////////////////////////////////////////////////////
	        ////////////////////////////////////////////////////
	        InputStream isRightEye = mContext.getResources().openRawResource(R.raw.ojoright);
	        File cascadeDirRightEye = mContext.getDir("cascaderighteye", Context.MODE_PRIVATE);
	        mCascadeRightEyeFile = new File(cascadeDirRightEye, "right_eye_detect.xml");
	        FileOutputStream osRightEye = new FileOutputStream(mCascadeRightEyeFile);
	        
	        byte[] bufferRightEye = new byte[4096];
	        int bytesReadRightEye;
	        while ((bytesReadRightEye = isRightEye.read(bufferRightEye)) != -1) {
	        	osRightEye.write(bufferRightEye, 0, bytesReadRightEye);
	        }
	        isRightEye.close();
	        osRightEye.close();
	        
	        mNativeRightEyeDetector = new DetectionBasedTracker(mCascadeRightEyeFile.getAbsolutePath(), 0, false);
	        
	        cascadeDirRightEye.delete();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
