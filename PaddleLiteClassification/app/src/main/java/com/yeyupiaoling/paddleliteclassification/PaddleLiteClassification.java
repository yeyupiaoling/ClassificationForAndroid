package com.yeyupiaoling.paddleliteclassification;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;

public class PaddleLiteClassification {
    private static final String TAG = PaddleLiteClassification.class.getName();

    private PaddlePredictor paddlePredictor;
    private Tensor inputTensor;
    private long[] inputShape = new long[]{1, 3, 224, 224};
    private static float[] scale = new float[]{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    private static float[] inputMean = new float[]{0.485f, 0.456f, 0.406f};
    private static float[] inputStd = new float[]{0.229f, 0.224f, 0.225f};
    private static final int NUM_THREADS = 4;

    /**
     * @param modelPath model path
     */
    public PaddleLiteClassification(String modelPath) throws Exception {
        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }
        try {
            MobileConfig config = new MobileConfig();
            config.setModelFromFile(modelPath);
            config.setThreads(NUM_THREADS);
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
            paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

            inputTensor = paddlePredictor.getInput(0);
            inputTensor.resize(inputShape);
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


    // prediction
    private float[] predict(Bitmap bmp) throws Exception {
        Bitmap b = getScaleBitmap(bmp);
        float[] inputData = getScaledMatrix(b, (int) inputShape[2], (int) inputShape[3]);
        b.recycle();
        bmp.recycle();
        inputTensor.setData(inputData);

        try {
            paddlePredictor.run();
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
        Tensor outputTensor = paddlePredictor.getOutput(0);
        float[] result = outputTensor.getFloatData();
        Log.d(TAG, Arrays.toString(result));
        int l = getMaxResult(result);
        return new float[]{l, result[l]};
    }

    // 对将要预测的图片进行预处理
    private static float[] getScaledMatrix(Bitmap bitmap, int desWidth, int desHeight) {
        float[] dataBuf = new float[3 * desWidth * desHeight];
        int rIndex;
        int gIndex;
        int bIndex;
        int[] pixels = new int[desWidth * desHeight];
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, desWidth, desHeight, false);
        bm.getPixels(pixels, 0, desWidth, 0, 0, desWidth, desHeight);
        int j = 0;
        int k = 0;
        for (int i = 0; i < pixels.length; i++) {
            int clr = pixels[i];
            j = i / desHeight;
            k = i % desWidth;
            rIndex = j * desWidth + k;
            gIndex = rIndex + desHeight * desWidth;
            bIndex = gIndex + desHeight * desWidth;
            // 转成RGB通道顺序
            dataBuf[rIndex] = (((clr & 0x00ff0000) >> 16) * scale[0] - inputMean[0]) / inputStd[0];
            dataBuf[gIndex] = (((clr & 0x0000ff00) >> 8) * scale[1] - inputMean[1]) / inputStd[1];
            dataBuf[bIndex] = (((clr & 0x000000ff)) * scale[2] - inputMean[2]) / inputStd[2];

        }
        if (bm.isRecycled()) {
            bm.recycle();
        }
        return dataBuf;
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

    private Bitmap getScaleBitmap(Bitmap bitmap) {
        int bmpWidth = bitmap.getWidth();
        int bmpHeight = bitmap.getHeight();
        int size = (int) inputShape[2];
        float scaleWidth = (float) size / bitmap.getWidth();
        float scaleHeight = (float) size / bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bitmap, 0, 0, bmpWidth, bmpHeight, matrix, true);
    }
}
