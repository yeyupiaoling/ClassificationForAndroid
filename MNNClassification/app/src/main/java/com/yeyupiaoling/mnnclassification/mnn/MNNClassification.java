package com.yeyupiaoling.mnnclassification.mnn;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MNNClassification {
    private static final String TAG = MNNClassification.class.getName();

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private final MNNImageProcess.Config dataConfig;
    private Matrix imgData;
    private final int inputWidth = 224;
    private final int inputHeight = 224;
    private static final int NUM_THREADS = 4;

    /**
     * @param modelPath model path
     */
    public MNNClassification(String modelPath) throws Exception {
        dataConfig = new MNNImageProcess.Config();
        dataConfig.mean = new float[]{128.0f, 128.0f, 128.0f};
        dataConfig.normal = new float[]{0.0078125f, 0.0078125f, 0.0078125f};
        dataConfig.dest = MNNImageProcess.Format.RGB;
        imgData = new Matrix();

        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }
        try {
            mNetInstance = MNNNetInstance.createFromFile(modelPath);
            MNNNetInstance.Config config = new MNNNetInstance.Config();
            config.numThread = NUM_THREADS;
            config.forwardType = MNNForwardType.FORWARD_CPU.type;
            mSession = mNetInstance.createSession(config);
            mInputTensor = mSession.getInput(null);
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    public float[] predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        float[] result = predictImage(bitmap);
        if (bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return result;
    }

    public float[] predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }

    // get max probability label
    public static int getMaxResult(float[] result) {
        float probability = 0;
        int r = 0;
        for (int i = 0; i < result.length; i++) {
            if (probability < result[i]) {
                probability = result[i];
                r = i;
            }
        }
        return r;
    }

    // prediction
    private float[] predict(Bitmap bmp) throws Exception {
        imgData.reset();
        imgData.postScale(inputWidth / (float) bmp.getWidth(), inputHeight / (float) bmp.getHeight());
        imgData.invert(imgData);
        MNNImageProcess.convertBitmap(bmp, mInputTensor, dataConfig, imgData);

        try {
            mSession.run();
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
        MNNNetInstance.Session.Tensor output = mSession.getOutput(null);
        float[] result = output.getFloatData();

//        // 归一化处理
//        float min = 100;
//        float max = -100;
//        for (float res: result) {
//            if (res>max){
//                max = res;
//            }
//            if (res<min){
//                min = res;
//            }
//        }
//        for (int i = 0; i < result.length; i++) {
//            result[i] = (result[i] - min )/ (max - min);
//        }

        Log.d(TAG, Arrays.toString(result));
        int l = getMaxResult(result);
        return new float[]{l, result[l]};
    }

    public void release() {
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }
    }
}
