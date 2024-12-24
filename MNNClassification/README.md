# 前言

MNN是一个轻量级的深度神经网络推理引擎，在端侧加载深度神经网络模型进行推理预测。目前，MNN已经在阿里巴巴的手机淘宝、手机天猫、优酷等20多个App中使用，覆盖直播、短视频、搜索推荐、商品图像搜索、互动营销、权益发放、安全风控等场景。此外，IoT等场景下也有若干应用。

下面就介绍如何使用MNN在Android设备上实现图像分类。


# 编译库和转换模型

## 编译MNN的Android动态库

1. 在 `https://developer.android.com/ndk/downloads/`下载安装NDK，建议使用最新稳定版本
2. 在 .bashrc 或者 .bash_profile 中设置 NDK 环境变量，例如：`export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b`
3. `cd /path/to/MNN`
4. `./schema/generate.sh`
5. `cd project/android`
6. 编译armv7动态库：`mkdir build_32 && cd build_32 && ../build_32.sh`
7. 编译armv8动态库：`mkdir build_64 && cd build_64 && ../build_64.sh`

## 模型转换

执行下面命令，得到模型转换工具 `MNNConvert`。

```bash
cd MNN/
./schema/generate.sh
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true && make -j4
```

通过以下命令可以把其他框架的模型转换为MNN模型。

**TensorFlow -> MNN**

把Tensorflow的冻结图模型转换为MNN模型，bizCode指定标记码，这个随便吧。如果冻结图转换不成功，可以使用下面的Tensorflow Lite模型，这个通常会成功。

```bash
./MNNConvert -f TF --modelFile XXX.pb --MNNModel XXX.mnn --bizCode biz
```

**TensorFlow Lite -> MNN**

把Tensorflow Lite的模型转换为MNN模型，bizCode指定标记码。

```bash
./MNNConvert -f TFLITE --modelFile XXX.tflite --MNNModel XXX.mnn --bizCode biz
```

**Caffe -> MNN**

把Caffe的模型转换为MNN模型，bizCode指定标记码。

```bash
./MNNConvert -f CAFFE --modelFile XXX.caffemodel --prototxt XXX.prototxt --MNNModel XXX.mnn --bizCode biz
```

**ONNX -> MNN**

把ONNX 的模型转换为MNN模型，bizCode指定标记码。

```bash
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```

# Android应用开发

把生成的C++的头文件放在 `app/include/MNN/`目录下，把生成的动态库文件放在 `app/src/main/jniLibs/`目录下，在 `app/src/main/cpp/`目录下编写JNI的C++代码，`com.yeyupiaoling.mnnclassification.mnn`包下放JNI的java代码和MNN的相关工具类，将转换的模型放在`assets`目录下。

## MNN工具

编写一个[MNNClassification.java](https://github.com/yeyupiaoling/ClassificationForAndroid/blob/master/MNNClassification/app/src/main/java/com/yeyupiaoling/mnnclassification/mnn/MNNClassification.java)工具类，关于MNN的操作都在这里完成，如加载模型、预测。在构造方法中，通过参数传递的模型路径加载模型，在加载模型的时候配置预测信息，例如是否使用CPU或者GPU，同时获取网络的输入输出层。同时MNN还提供了很多的图像预处理工具，对图像的预处理非常简单。要注意的是图像的均值 `dataConfig.mean`和标准差 `dataConfig.normal`，还有图片的输入通道顺序 `dataConfig.dest`，因为在训练的时候图像预处理可能不一样的，有些读者出现在电脑上准确率很高，但在手机上准确率很低，多数情况下就是这个图像预处理做得不对。

```java
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
```

为了兼容图片路径和Bitmap格式的图片预测，这里创建了两个重载方法，它们都是通过调用 `predict()`

```java
public int predictImage(String image_path) throws Exception {
    if (!new File(image_path).exists()) {
        throw new Exception("image file is not exists!");
    }
    FileInputStream fis = new FileInputStream(image_path);
    Bitmap bitmap = BitmapFactory.decodeStream(fis);
    int result = predictImage(bitmap);
    if (bitmap.isRecycled()) {
        bitmap.recycle();
    }
    return result;
}

public int predictImage(Bitmap bitmap) throws Exception {
    return predict(bitmap);
}
```

这里创建一个获取最大概率值，并把下标返回的方法，其实就是获取概率最大的预测标签。

```java
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
```

这个方法就是MNN执行预测的最后一步，通过执行 `mSession.run()`对输入的数据进行预测并得到预测结果，通过解析获取到最大的概率的预测标签，并返回。到这里MNN的工具就完成了。

```java
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
    Log.d(TAG, Arrays.toString(result));
    int l = getMaxResult(result);
    return new float[]{l, result[l]};
}
```

## 选择图片预测

本教程会有两个页面，一个是选择图片进行预测的页面，另一个是使用相机实时预测并显示预测结果。以下为 `activity_main.xml`的代码，通过按钮选择图片，并在该页面显示图片和预测结果。

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    tools:context=".MainActivity">

    <ImageView
        android:id="@+id/image_view"
        android:layout_width="match_parent"
        android:layout_height="400dp" />

    <TextView
        android:id="@+id/result_text"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_below="@id/image_view"
        android:text="识别结果"
        android:textSize="16sp" />


    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:orientation="horizontal">

        <Button
            android:id="@+id/select_img_btn"
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="选择照片" />


        <Button
            android:id="@+id/open_camera"
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="实时预测" />

    </LinearLayout>

</RelativeLayout>
```

在 `MainActivity.java`中，进入到页面我们就要先加载模型，我们是把模型放在Android项目的assets目录的，我们需要把模型复制到一个缓存目录，然后再从缓存目录加载模型，同时还有读取标签名，标签名称按照训练的label顺序存放在assets的 `label_list.txt`，以下为实现代码。

```java
classNames = Utils.ReadListFromFile(getAssets(), "label_list.txt");
String classificationModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2.mnn";
Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2.mnn", classificationModelPath);
try {
    mnnClassification = new MNNClassification(classificationModelPath);
    Toast.makeText(MainActivity.this, "模型加载成功！", Toast.LENGTH_SHORT).show();
} catch (Exception e) {
    Toast.makeText(MainActivity.this, "模型加载失败！", Toast.LENGTH_SHORT).show();
    e.printStackTrace();
    finish();
}
```

添加两个按钮点击事件，可以选择打开相册读取图片进行预测，或者打开另一个Activity进行调用摄像头实时识别。

```java
Button selectImgBtn = findViewById(R.id.select_img_btn);
Button openCamera = findViewById(R.id.open_camera);
imageView = findViewById(R.id.image_view);
textView = findViewById(R.id.result_text);
selectImgBtn.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // 打开相册
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, 1);
    }
});
openCamera.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // 打开实时拍摄识别页面
        Intent intent = new Intent(MainActivity.this, CameraActivity.class);
        startActivity(intent);
    }
});
```

当打开相册选择照片之后，回到原来的页面，在下面这个回调方法中获取选择图片的Uri，通过Uri可以获取到图片的绝对路径。如果Android8以上的设备获取不到图片，需要在 `AndroidManifest.xml`配置文件中的 `application`添加 `android:requestLegacyExternalStorage="true"`。拿到图片路径之后，调用 `TFLiteClassificationUtil`类中的 `predictImage()`方法预测并获取预测值，在页面上显示预测的标签、对应标签的名称、概率值和预测时间。

```java
@Override
protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    String image_path;
    if (resultCode == Activity.RESULT_OK) {
        if (requestCode == 1) {
            if (data == null) {
                Log.w("onActivityResult", "user photo data is null");
                return;
            }
            Uri image_uri = data.getData();
            image_path = getPathFromURI(MainActivity.this, image_uri);
            try {
                // 预测图像
                FileInputStream fis = new FileInputStream(image_path);
                imageView.setImageBitmap(BitmapFactory.decodeStream(fis));
                long start = System.currentTimeMillis();
                float[] result = mnnClassification.predictImage(image_path);
                long end = System.currentTimeMillis();
                String show_text = "预测结果标签：" + (int) result[0] +
                        "\n名称：" +  classNames[(int) result[0]] +
                        "\n概率：" + result[1] +
                        "\n时间：" + (end - start) + "ms";
                textView.setText(show_text);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

上面获取的Uri可以通过下面这个方法把Url转换成绝对路径。

```java
// get photo from Uri
public static String getPathFromURI(Context context, Uri uri) {
    String result;
    Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
    if (cursor == null) {
        result = uri.getPath();
    } else {
        cursor.moveToFirst();
        int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
        result = cursor.getString(idx);
        cursor.close();
    }
    return result;
}
```

## 摄像头实时预测

在调用相机实时预测我就不再介绍了，原理都差不多，具体可以查看[https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/TFLiteClassification](https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/TFLiteClassification)中的源代码。核心代码如下，创建一个子线程，子线程中不断从摄像头预览的 `AutoFitTextureView`上获取图像，并执行预测，并在页面上显示预测的标签、对应标签的名称、概率值和预测时间。每一次预测完成之后都立即获取图片继续预测，只要预测速度够快，就可以看成实时预测。

```java
private Runnable periodicClassify =
        new Runnable() {
            @Override
            public void run() {
                synchronized (lock) {
                    if (runClassifier) {
                        // 开始预测前要判断相机是否已经准备好
                        if (getApplicationContext() != null && mCameraDevice != null && mnnClassification != null) {
                            predict();
                        }
                    }
                }
                if (mInferThread != null && mInferHandler != null && mCaptureHandler != null && mCaptureThread != null) {
                    mInferHandler.post(periodicClassify);
                }
            }
        };

// 预测相机捕获的图像
private void predict() {
    // 获取相机捕获的图像
    Bitmap bitmap = mTextureView.getBitmap();
    try {
        // 预测图像
        long start = System.currentTimeMillis();
        float[] result = mnnClassification.predictImage(bitmap);
        long end = System.currentTimeMillis();
        String show_text = "预测结果标签：" + (int) result[0] +
                "\n名称：" +  classNames[(int) result[0]] +
                "\n概率：" + result[1] +
                "\n时间：" + (end - start) + "ms";
        textView.setText(show_text);
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```

本项目中使用的了读取图片的权限和打开相机的权限，所以不要忘记在 `AndroidManifest.xml`添加以下权限申请。

```bash
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
```

如果是Android 6 以上的设备还要动态申请权限。

```java
    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }
```

**效果图：**
![在这里插入图片描述](https://s1.ax1x.com/2020/09/05/wVLG6g.jpg)
