package com.yeyupiaoling.tnnclassification.tnn;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;
import java.io.FileInputStream;

public class ImageClassifyUtil {
    private static final int WIDTH = 224;
    private  static final int HEIGHT = 224;

    public ImageClassifyUtil() {
        System.loadLibrary("TNN");
        System.loadLibrary("tnn_wrapper");
    }

    // 重载方法，根据图片路径转Bitmap预测
    public float[] predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        Bitmap scaleBitmap = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, false);
        float[] result = predictImage(scaleBitmap);
        if (bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return result;
    }

    // 重载方法，直接使用Bitmap预测
    public float[] predictImage(Bitmap bitmap) {
        Bitmap scaleBitmap = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, false);
        float[] results = predict(scaleBitmap, WIDTH, HEIGHT);
        int l = getMaxResult(results);
        return new float[]{l, results[l] * 0.01f};
    }


    // 获取概率最大的标签
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


    public native int init(String modelPath, String protoPath, int computeUnitType);

    public native float[] predict(Bitmap image, int width, int height);

    public native int deinit();
}
