

# 前言
Paddle Lite是飞桨基于Paddle Mobile全新升级推出的端侧推理引擎，在多硬件、多平台以及硬件混合调度的支持上更加完备，为包括手机在内的端侧场景的AI应用提供高效轻量的推理能力，有效解决手机算力和内存限制等问题，致力于推动AI应用更广泛的落地。


# 模型转换
Paddle Lite使用的是PaddlePaddle保存的预测模型，如果不了解PaddlePaddle的模型保存，可以参考[《模型的保存与使用》](https://blog.doiduoyi.com/articles/1584974792165.html)这篇文章。下面简单介绍一下保存模型的方式。通过使用`fluid.io.save_inference_model()`接口可以保存预测模型，预测模型值保存推所需的网络，不会保存损失函数等。当使用`model_filename`和`params_filename`指定参数之后，保存的预测模型只有两个文件，这种称为合并模型，否则会以网络结构命名将大量的参数文件保存在`dirname`指定的路径下，这种叫做非合并模型。例如通过以下的代码片段保存的预测模型为`model`和`params`，这两个模型将会用于下一步的模型转换。
```python
import paddle.fluid as fluid

# 定义网络
image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=image, size=10, act='softmax')

loss = fluid.layers.cross_entropy(input=predict, label=label)
avg_loss = fluid.layers.mean(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

# 数据输入及训练过程

# 保存预测模型
fluid.io.save_inference_model(dirname="mobilenet_v2/",
                              feeded_var_names=[image.name],
                              target_vars=[predict],
                              executor=exe,
                              model_filename="model", 
                              params_filename="params")
```


## opt转换
使用`fluid.io.save_inference_model()`接口可以保存预测模型并不能直接使用，还需要通过opt工具转换，这个工具可以下载Paddle Lite预编译的，或者通过源码编译，opt下载地址：[https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html#opt](https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html#opt)，关于如何编译opt请看下一部分。

通过以下命令即即可把预测模型转变成Paddle Lite使用的模型，其中输出的`mobilenet_v2.nb`就是所需的模型文件，因为转换之后，模型可以在`valid_targets`指定的环境上加速预测，所以变得非常牛B，因此后缀名为`nb`（开个玩笑）。
```
./opt \
    --model_file=mobilenet_v2/model \
    --param_file=mobilenet_v2/params \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v2 \
    --valid_targets=arm opencl \
    --record_tailoring_info=false
```

上面参数的说明如下表所示，其中需要关注的是`valid_targets`参数，要看模型用着上面设备上，通过指定backend可以使用更好的加速方式。有些读取可能会出现这样的疑问，上面使用的是合并的模型，没合并的模型怎样用呢，其实很简单，只有设置`--model_dir`，忽略`--model_file`和`--param_file`就可以了。

| 参数 |	说明 |
|--|:--|
| --model_dir | 待优化的PaddlePaddle模型（非combined形式）的路径 |
| --model_file	 | 待优化的PaddlePaddle模型（combined形式）的网络结构文件路径。 |
| --param_file	 | 待优化的PaddlePaddle模型（combined形式）的权重文件路径。 |
| --optimize_out_type | 	输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。 |
| --optimize_out	 | 优化模型的输出路径。 |
| --valid_targets	 | 指定模型可执行的backend，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm。 |
| --record_tailoring_info	 | 当使用 根据模型裁剪库文件 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。 |


## 源码编译opt
上面所使用的opt工具是通过下载得到的，如果读者喜欢折腾，可以尝试自行源码编译编译，首先是环境搭建，环境搭建有两种方式，第一种是使用Docker，第二种是本地搭建环境。

1. 使用Docker容器，只要3条命令即可搭建环境，这个也是最简单的方式。
```bash
# 拉取Paddle Lite镜像
docker pull paddlepaddle/paddle-lite:2.0.0_beta
# 克隆源码
git clone https://github.com/PaddlePaddle/Paddle-Lite.git

# 进行Paddle Lite容器
docker run -it \
  --name paddlelite_docker \
  -v $PWD/Paddle-Lite:/Paddle-Lite \
  --net=host \
  paddlepaddle/paddle-lite /bin/bash
```

2. 在Ubuntu本地搭建Paddle Lite编译环境，通过执行以下命令在Ubuntu本地完成环境搭建。

```bash
# 1. 安装基本环境
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip adb curl

# 2. 安装Java环境
apt-get install -y default-jdk

# 3. 安装Cmake
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
    mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \  
    ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

# 4. 安装NDK
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

# 5. 添加环境变量
echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
source ~/.bashrc
```

在以上的环境中编译opt工具，执行以下命令即可完成编译，编译完成之后，在`build.opt/lite/api/`下的可执行文件`opt`。
```
cd Paddle-Lite && ./lite/tools/build.sh build_optimize_tool
```

# Paddle Lite的Android预测库
Paddle Lite的Android预测库也可以通过下载预编译的，或者通过源码编译。下载地址为：，注意本教程使用的是静态库的方式，而且使用的是图像识别的，所以需要选择的下载库为with_extra=ON，arm_stl=c++_static，with_cv=ON的armv7和armv8库。下载解压之后得到的目录结构如下，其中我们所需的在`java`的jar和so动态库，注意32位的so动态库放在Android的`armeabi-v7a`目录，64位的so动态库放在Android的`arm64-v8a`目录，jar包只取一个就好。
```
inference_lite_lib.android.armv8/
|-- cxx                           C++ 预测库和头文件
|   |-- include                                C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                    C++预测库
|       |-- libpaddle_api_light_bundled.a             C++静态库
|       `-- libpaddle_light_api_shared.so             C++动态库
|-- java                          Java预测库
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                          C++和Java示例代码
|   |-- cxx                                  C++  预测库demo
|   `-- java                                 Java 预测库demo
```

同样如果读者喜欢折腾，可以尝试自行源码编译编译，在上面编译opt工具时搭建的环境上编译Paddle Lite的Android预测库。在Paddle Lite源码的根目录下执行以下两条命令编译Paddle Lite的Android预测库。
```
./lite/tools/build_android.sh --arch=armv7 --with_extra=ON
./lite/tools/build_android.sh --arch=armv8 --with_extra=ON
```
完成编译之后，会在`Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7`和`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8`目录生成所以的jar和动态库，所在位置和使用查看上面的下载Android预测库的介绍。


# 开发Android项目
创建一个Android项目，在`app/libs`目录下存放上一步编译得到的`PaddlePredictor.jar`，并添加到`app`库中，添加方式可以是选择这个jar包，右键选择`add as Librarys`，或者在`app/build.gradle`添加以下代码结果都是一样的。
```
implementation files('libs\\PaddlePredictor.jar')
```

然后在`app/src/main/jniLibs`下存放下载或者编译得到的动态库，最好把32位和64为的动态库`libpaddle_lite_jni.so`都添加进去，分别是`armeabi-v7a`目录和`arm64-v8a`目录。

复制转换的预测模型到`app/src/main/assets`目录下，还有类别的标签，每一行对应一个标签名称。

## Paddle Lite工具
编写一个[PaddleLiteClassification](https://github.com/yeyupiaoling/ClassificationForAndroid/blob/master/PaddleLiteClassification/app/src/main/java/com/yeyupiaoling/paddleliteclassification/PaddleLiteClassification.java)工具类，关于Paddle Lite的操作都在这里完成，如加载模型、预测。在构造方法中，通过参数传递的模型路径加载模型，在加载模型的时候配置预测信息，如预测时使用的线程数量，使用计算资源的模式，要注意的是图像预处理的缩放比例`scale`，均值`inputMean`和标准差`inputStd`，因为在训练的时候图像预处理可能不一样的，有些读者出现在电脑上准确率很高，但在手机上准确率很低，多数情况下就是这个图像预处理做得不对。
```java
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
```

为了兼容图片路径和Bitmap格式的图片预测，这里创建了两个重载方法，它们都是通过调用`predict()`

```java
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

在数据输入之前，需要对数据进行预处理，输入的数据是一个浮点数组，但是目前输入的是一个Bitmap的图片，所以需要把Bitmap转换为浮点数组，在转换过程中需要对图像做相应的预处理，如乘比例，减均值，除以方差。为了避免输入的图像过大，图像预处理变慢，通常在元数据预处理之前，需要对图像进行压缩，使用`getScaleBitmap()`方法可以压缩等比例压缩图像。
```java
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
```

这个方法就是Paddle Lite执行预测的最后一步，使用`inputTensor.setData(inputData)`输入预测图像数据，通过执行`paddlePredictor.run()`对输入的数据进行预测并得到预测结果，预测结果通过`paddlePredictor.getOutput(0)`提前出来，最后通过解析获取到最大的概率的预测标签。到这里Paddle Lite的工具就完成了。

```java
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
```


## 选择图片预测
本教程会有两个页面，一个是选择图片进行预测的页面，另一个是使用相机实时预测并显示预测结果。以下为`activity_main.xml`的代码，通过按钮选择图片，并在该页面显示图片和预测结果。
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

在`MainActivity.java`中，进入到页面我们就要先加载模型，我们是把模型放在Android项目的assets目录的，但是Tensorflow Lite并不建议直接在assets读取模型，所以我们需要把模型复制到一个缓存目录，然后再从缓存目录加载模型，同时还有读取标签名，标签名称按照训练的label顺序存放在assets的`label_list.txt`，以下为实现代码。
```java
classNames = Utils.ReadListFromFile(getAssets(), "label_list.txt");
String classificationModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2.nb";
Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2.nb", classificationModelPath);
try {
    paddleLiteClassification = new PaddleLiteClassification(classificationModelPath);
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

当打开相册选择照片之后，回到原来的页面，在下面这个回调方法中获取选择图片的Uri，通过Uri可以获取到图片的绝对路径。如果Android8以上的设备获取不到图片，需要在`AndroidManifest.xml`配置文件中的`application`添加`android:requestLegacyExternalStorage="true"`。拿到图片路径之后，调用`PaddleLiteClassification`类中的`predictImage()`方法预测并获取预测值，在页面上显示预测的标签、对应标签的名称、概率值和预测时间。
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
                float[] result = paddleLiteClassification.predictImage(image_path);
                long end = System.currentTimeMillis();
                String show_text = "预测结果标签：" + (int) result[0] +
                        "\n名称：" +  classNames.get((int) result[0]) +
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
在调用相机实时预测我就不再介绍了，原理都差不多，具体可以查看[https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/PaddleLiteClassification](https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/PaddleLiteClassification)中的源代码。核心代码如下，创建一个子线程，子线程中不断从摄像头预览的`AutoFitTextureView`上获取图像，并执行预测，并在页面上显示预测的标签、对应标签的名称、概率值和预测时间。每一次预测完成之后都立即获取图片继续预测，只要预测速度够快，就可以看成实时预测。
```java
private Runnable periodicClassify =
        new Runnable() {
            @Override
            public void run() {
                synchronized (lock) {
                    if (runClassifier) {
                        // 开始预测前要判断相机是否已经准备好
                        if (getApplicationContext() != null && mCameraDevice != null && tfLiteClassificationUtil != null) {
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
        float[] result = paddleLiteClassification.predictImage(bitmap);
        long end = System.currentTimeMillis();
        String show_text = "预测结果标签：" + (int) result[0] +
                "\n名称：" +  classNames.get((int) result[0]) +
                "\n概率：" + result[1] +
                "\n时间：" + (end - start) + "ms";
        textView.setText(show_text);
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```

本项目中使用的了读取图片的权限和打开相机的权限，所以不要忘记在`AndroidManifest.xml`添加以下权限申请。
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

**选择图片识别效果图：**

![选择图片识别效果图](https://s1.ax1x.com/2020/07/24/UvE511.jpg)



**相机实时识别效果图：**

![相机实时识别效果图](https://s1.ax1x.com/2020/07/24/UvEoX6.jpg)
