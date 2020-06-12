package com.yeyupiaoling.tfliteclassification;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;

public class TFLiteClassificationUtil {
    private static final String TAG = TFLiteClassificationUtil.class.getName();
    private Interpreter tflite;
    private TensorImage inputImageBuffer;
    private final TensorBuffer outputProbabilityBuffer;
    private static final int NUM_THREADS = 4;
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private final ImageProcessor imageProcessor;


    /**
     * @param modelPath model path
     */
    public TFLiteClassificationUtil(String modelPath) throws Exception {


        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }

        try {
            Interpreter.Options options = new Interpreter.Options();
            // 使用多线程预测
            options.setNumThreads(NUM_THREADS);
            // 使用Android自带的API或者GPU加速
            NnApiDelegate delegate = new NnApiDelegate();
//            GpuDelegate delegate = new GpuDelegate();
            options.addDelegate(delegate);
            tflite = new Interpreter(file, options);
            // 获取输入，shape为{1, height, width, 3}
            int[] imageShape = tflite.getInputTensor(tflite.getInputIndex("mobilenetv2_1.00_224_input")).shape();
            DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("mobilenetv2_1.00_224_input")).dataType();
            inputImageBuffer = new TensorImage(imageDataType);
            // 获取输入，shape为{1, NUM_CLASSES}
            int[] probabilityShape = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).shape();
            DataType probabilityDataType = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).dataType();
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            // 添加图像预处理方式
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(imageShape[1], imageShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                    .build();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    // 重载方法，根据图片路径转Bitmap预测
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

    // 重载方法，直接使用Bitmap预测
    public float[] predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }


    // 数据预处理
    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);
        return imageProcessor.process(inputImageBuffer);
    }

    // 执行预测
    private float[] predict(Bitmap bmp) throws Exception {
        inputImageBuffer = loadImage(bmp);

        try {
            tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
        float[] results = outputProbabilityBuffer.getFloatArray();
        int l = getMaxResult(results);
        return new float[]{l, results[l]};
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
}
