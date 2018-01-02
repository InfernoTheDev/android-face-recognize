package project.illnino.com.facerec;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.opencv.android.OpenCVLoader;

import project.illnino.com.facerec.utils.FaceDetectionUtils;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        FaceDetectionUtils.initialize(MainActivity.this);

        startActivity(new Intent(this, FaceDetectActivity.class));

    }
}
