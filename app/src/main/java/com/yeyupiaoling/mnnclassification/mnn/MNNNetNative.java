package com.yeyupiaoling.mnnclassification.mnn;

import android.graphics.Bitmap;
import android.util.Log;


public class MNNNetNative {
    private static final String TAG = MNNNetNative.class.getName();
    // load libraries
    static {
        System.loadLibrary("MNN");
        try {
            System.loadLibrary("MNN_CL");
            System.loadLibrary("MNN_GL");
            System.loadLibrary("MNN_Vulkan");
        } catch (Throwable ce) {
            Log.w(TAG, "load MNN GPU so exception=%s", ce);
        }
        System.loadLibrary("mnncore");
    }

    //Net
    protected static native long nativeCreateNetFromFile(String modelName);

    protected static native long nativeReleaseNet(long netPtr);


    //Session
    protected static native long nativeCreateSession(long netPtr, int forwardType, int numThread, String[] saveTensors, String[] outputTensors);

    protected static native void nativeReleaseSession(long netPtr, long sessionPtr);

    protected static native int nativeRunSession(long netPtr, long sessionPtr);

    protected static native int nativeRunSessionWithCallback(long netPtr, long sessionPtr, String[] nameArray, long[] tensorAddr);

    protected static native int nativeReshapeSession(long netPtr, long sessionPtr);

    protected static native long nativeGetSessionInput(long netPtr, long sessionPtr, String name);

    protected static native long nativeGetSessionOutput(long netPtr, long sessionPtr, String name);


    //Tensor
    protected static native void nativeReshapeTensor(long netPtr, long tensorPtr, int[] dims);

    protected static native int[] nativeTensorGetDimensions(long tensorPtr);

    protected static native void nativeSetInputIntData(long netPtr, long tensorPtr, int[] data);

    protected static native void nativeSetInputFloatData(long netPtr, long tensorPtr, float[] data);


    //If dest is null, return length
    protected static native int nativeTensorGetData(long tensorPtr, float[] dest);

    protected static native int nativeTensorGetIntData(long tensorPtr, int[] dest);

    protected static native int nativeTensorGetUINT8Data(long tensorPtr, byte[] dest);


    //ImageProcess
    protected static native boolean nativeConvertBitmapToTensor(Bitmap srcBitmap, long tensorPtr, int destFormat, int filterType, int wrap, float[] matrixValue, float[] mean, float[] normal);

    protected static native boolean nativeConvertBufferToTensor(byte[] bufferData, int width, int height, long tensorPtr,
                                                                int srcFormat, int destFormat, int filterType, int wrap, float[] matrixValue, float[] mean, float[] normal);

}
