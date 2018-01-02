package project.illnino.com.facerec;

import java.io.File;
import java.util.HashMap;
import java.util.Vector;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.Utils;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.Color;

import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.util.TypedValue;
import android.view.View.OnClickListener;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.TranslateAnimation;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ImageView.ScaleType;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import project.illnino.com.facerec.facedatabase.Person;
import project.illnino.com.facerec.methods.FaceDetection;
import project.illnino.com.facerec.utils.FaceDetectionUtils;
import project.illnino.com.facerec.utils.ImageUtils;

import static project.illnino.com.facerec.methods.FaceDetection.FACE_COLOR;
import static project.illnino.com.facerec.utils.FaceDetectionUtils.mNativeDetector;

public class FaceDetectActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = FaceDetectActivity.class.getName();
    private Mat mRgba;
    private Mat mGray;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;
    private HashMap<Integer, Mat> capturedMats = new HashMap<Integer, Mat>();
    private Mat capturedMat;

    private boolean capturingImage = false;
    private ImageView img_preview;
    private Button actionButton;
    private ImageButton swapCamButton;

    private Person thisPerson;
    private boolean isTraining = true;
    private static Vector<Person> persons = new Vector<Person>();
    private boolean detectionInProgress = false;
    private int screenWidth;
    private int screenHeight;
    private long lastDetectionTime = 0;
    private int deleteButtonFirstPos;
    private int deleteButtonSecondPos;
    private LinearLayout bgLayout;
    private Context mContext;
    private int mCameraIndex = 0;


    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        isTraining = getIntent().getBooleanExtra("Training", true);
        setContentView(R.layout.activity_face_detect);
        mContext = getApplicationContext();
        FaceDetectionUtils.initialize(FaceDetectActivity.this);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);


        img_preview = (ImageView) findViewById(R.id.img_preview);
        actionButton = (Button) findViewById(R.id.btn_action);
        actionButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                closeSoftInput();
                if (capturingImage) {
                    actionButton.setText("START");
                    capturingImage = false;
                } else {
                    actionButton.setText("STOP");
                    capturingImage = true;
                }
            }
        });

        swapCamButton = (ImageButton) findViewById(R.id.btn_swapcam);
        swapCamButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                swapCamera();
            }
        });

        resetImagesForTraining();
        mOpenCvCameraView.enableView();

    }

    public static String getFaceFolder(int index){
        return persons.get(index).getFacesFolderPath();
    }

    public static long getPersonID(int index){
        return persons.get(index).getId();
    }

    private void recognize(){
        new Thread(new Runnable() {

            @Override
            public void run() {
                int result = faceRecognition(capturedMat.getNativeObjAddr(), persons.size());
                for(int i = 0; i < persons.size(); i++) {
                    int id = (int)persons.get(i).getId();
                    if(result == id) {
                        final int index = i;
                        FaceDetectActivity.this.runOnUiThread(new Runnable() {

                            @Override
                            public void run() {
                                Mat m = Imgcodecs.imread(persons.get(index).getFacesFolderPath()+"/1.jpg");
                                final Bitmap bmp = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.RGB_565);
                                Utils.matToBitmap(m, bmp);
                                img_preview.setImageBitmap(bmp);
                                Toast.makeText(
                                        getApplication(),
                                        persons.get(index).getName(),
                                        Toast.LENGTH_LONG
                                ).show();
                            }
                        });
                    }
                }
            }
        }).start();
    }



    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        mOpenCvCameraView.enableView();
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    private void detectFaceOnFrame(final Mat frame) {
        Thread t = new Thread(new Runnable() {

            @Override
            public void run() {
                detectionInProgress = true;
                if (mAbsoluteFaceSize == 0) {
                    int height = frame.rows();
                    if (Math.round(height * mRelativeFaceSize) > 0) {
                        mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
                    }
                    if (mNativeDetector!= null)
                        mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
                }


                Mat faceMat = FaceDetection.detectFaces(null, frame, capturingImage);
                if (faceMat != null) {

                    long now = System.currentTimeMillis();
                    if (now - lastDetectionTime > 400) {
                        Mat m = new Mat(faceMat.rows(), faceMat.cols(), faceMat.type());
                        faceMat.copyTo(m);
                        if (capturingImage) {
                            onFaceCaptured(m);
                        }else{
                            onFaceCapturedRecognize(m);
                        }
                    }
                    lastDetectionTime = now;
                }
                detectionInProgress = false;
            }
        });
        if (!detectionInProgress) {
            t.start();
        }
    }
    private void onFaceCapturedRecognize(Mat faceMat){
        capturedMat = faceMat;
        final Bitmap bmp = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(faceMat, bmp);
        recognize();
    }

    private void onFaceCaptured(Mat faceMat) {

        capturedMat = faceMat;

        for (int i = 0; i < 10; i++) {
            if (!capturedMats.containsKey(i)) {
                capturedMats.put(i, faceMat);
                final Bitmap bmp = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.RGB_565);
                Utils.matToBitmap(faceMat, bmp);
                final int index = i;
                FaceDetectActivity.this.runOnUiThread(new Runnable() {

                    @Override
                    public void run() {
                        img_preview.setImageBitmap(bmp);
                    }
                });
                break;
            }
        }

        if (capturedMats.size() == 10) {
            FaceDetectActivity.this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    capturingImage = false;
                    actionButton.setText("START");
                    if (isTraining) {
                        FaceDetectActivity.this.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                showInputNameBox();
                            }
                        });
                    }
                }
            });
        }
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();

        mGray = inputFrame.gray();

        drawFaceFrame();

        if (!detectionInProgress) {
            Mat image = new Mat(mGray.rows(), mGray.cols(), mGray.type());
            mGray.copyTo(image);
            detectFaceOnFrame(image);
        }

        return mRgba;
    }

    public void drawFaceFrame() {

        MatOfRect faces = new MatOfRect();

        if (mNativeDetector != null)
            mNativeDetector.detect(mGray, faces);

        Rect[] facesArray = faces.toArray();

        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_COLOR, 3);
        }
    }


    private void insertToDatabase(String name) {
        long id = -1;
        String properPath = null;
        Person p = null;
        File photoFolder;
        while (true) {
            id++;
            String path = "" + Environment.getExternalStorageDirectory();
            path += "/Android/data/" + getPackageName() + "/.faces/" + id;
            photoFolder = new File(path);
            if (!photoFolder.exists()) {
                properPath = path;
                break;
            }
        }
        if (name == null || name.length() == 0) {
            name = "unNamed" + id;
        }
        final long idToSave = id;
        final String nameToSave = name;
        final String pathToSave = properPath;
        final File faceFolder = photoFolder;
        new Thread(new Runnable() {

            @Override
            public void run() {
                FaceDetectionUtils.faceDataSource.open();
                FaceDetectionUtils.faceDataSource.createPerson(idToSave, nameToSave, pathToSave);
                FaceDetectionUtils.faceDataSource.close();
                faceFolder.mkdirs();
                for (int i = 0; i < capturedMats.size(); i++) {
                    if (capturedMats.containsKey(i)) {
                        ImageUtils.saveImageAsPGM(pathToSave + "/" + i + ".jpg", capturedMats.get(i).getNativeObjAddr());
                    }
                }
                FaceDetectActivity.this.runOnUiThread(new Runnable() {

                    @Override
                    public void run() {
                        if (isTraining) {
                            showAlert("Face Recognition Training", "Images are saved!", "OK");
                        }
                    }
                });

            }
        }).start();
    }

    private void swapCamera() {
        mCameraIndex = mCameraIndex == 0 ? 1 : 0;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraIndex);
        mOpenCvCameraView.enableView();
    }

    private void closeSoftInput() {
        InputMethodManager inputMethodManager = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        IBinder windowToken = getWindow().getDecorView().getWindowToken();
        inputMethodManager.hideSoftInputFromWindow(windowToken, 0);
    }

    private void resetImagesForTraining() {
        capturedMats.clear();
        actionButton.setText("START");
        capturingImage = false;
    }

    private void showAlert(String title, String message, String buttonText) {
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(this);

        alertDialogBuilder.setTitle(title);

        alertDialogBuilder
                .setMessage(message)
                .setCancelable(false)
                .setPositiveButton(buttonText, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.cancel();
                        if (isTraining) {
                            resetImagesForTraining();
                        }
                    }
                });

        // create alert dialog
        AlertDialog alertDialog = alertDialogBuilder.create();

        // show it
        alertDialog.show();
    }

    private void showInputNameBox() {

        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(this);

        final EditText input = new EditText(FaceDetectActivity.this);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT);
        input.setLayoutParams(lp);
        alertDialogBuilder.setView(input);

        alertDialogBuilder
                .setCancelable(false)
                .setPositiveButton("save", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        closeSoftInput();
                        if(isTraining){
                            insertToDatabase(String.valueOf(input.getText()));
                        }
                    }
                });

        // create alert dialog
        AlertDialog alertDialog = alertDialogBuilder.create();

        // show it
        alertDialog.show();
    }
    private static native int faceRecognition(long inputImage, int personCount);
}