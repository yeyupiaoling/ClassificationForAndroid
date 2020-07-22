> 原文博客：[Doi技术团队](http://blog.doiduoyi.com)<br/>
> 链接地址：[https://blog.doiduoyi.com/authors/1584446358138](https://blog.doiduoyi.com/authors/1584446358138)<br/>
> 初心：记录优秀的Doi技术团队学习经历<br/>
>本文链接：[基于Tensorflow2 Lite在Android手机上实现图像分类](https://blog.doiduoyi.com/articles/1595399632765.html)<br/>


# 前言
Tensorflow2之后，训练保存的模型也有所变化，基于Keras接口搭建的网络模型默认保存的模型是h5格式的，而之前的模型格式是pb。Tensorflow2的h5格式的模型转换成tflite格式模型非常方便。本教程就是介绍如何使用Tensorflow2的Keras接口训练分类模型并使用Tensorflow Lite部署到Android设备上。



# 训练和转换模型
以下是使用Tensorflow2的keras搭建的一个MobileNetV2模型并训练自定义数据集，本教程主要是介绍如何在Android设备上使用Tensorflow Lite部署分类模型，所以关于训练模型只是简单介绍，代码并不完整。通过下面的训练模型，我们最终会得到一个`mobilenet_v2.h5`模型。
```python
import os
import tensorflow as tf
import reader
import config as cfg

# 获取模型
input_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL)
model = tf.keras.Sequential(
    [tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, pooling='max'),
     tf.keras.layers.Dense(units=cfg.CLASS_DIM, activation='softmax')])
model.summary()

# 获取训练数据
train_data = reader.train_reader(data_list_path=cfg.TRAIN_LIST_PATH, batch_size=cfg.BATCH_SIZE)

# 定义训练参数
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 开始训练
model.fit(train_data, epochs=cfg.EPOCH_SUM, workers=4)

# 保存h5模型
if not os.path.exists(os.path.dirname(cfg.H5_MODEL_PATH)):
    os.makedirs(os.path.dirname(cfg.H5_MODEL_PATH))
model.save(filepath=cfg.H5_MODEL_PATH)
print('saved h5 model!')
```

通过上面得到的`mobilenet_v2.h5`模型，我们需要转换为tflite格式的模型，在Tensorflow2之后，这个转换就变动很简单了，通过下面的几行代码即可完成转换，最终我们会得到一个`mobilenet_v2.tflite`模型。
```python
import tensorflow as tf
import config as cfg

# 加载模型
model = tf.keras.models.load_model(cfg.H5_MODEL_PATH)

# 生成非量化的tflite模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(cfg.TFLITE_MODEL_FILE, 'wb').write(tflite_model)
print('saved tflite model!')
```

如果保存的模型格式不是h5，而是tf格式的，如下代码，保存的模型是tf格式的。
```python
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3))

model.save(filepath='mobilenet_v2', save_format='tf')
```

如果是tf格式的模型，那需要使用以下转换模型的方式。
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('mobilenet_v2')
tflite_model = converter.convert()
open("mobilenet_v2.tflite", "wb").write(tflite_model)
```

在部署到Android中可能需要到输入输出层的名称，通过下面代码可以获取到输入输出层的名称和shape。
```python
import tensorflow as tf

model_path = 'models/mobilenet_v2.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)
```

# 部署到Android设备
首先要在`build.gradle`导入这三个库，如果不使用GPU可以只导入两个库。
```bash
implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
```

在以前还需要在`android`下添加以下代码，避免在打包apk的是对模型有压缩操作，损坏模型。现在好像不加也没有关系，但是为了安全起见，还是添加上去。
```
    aaptOptions {
        noCompress "tflite"
    }
```

## Tensorflow Lite工具
编写一个[TFLiteClassificationUtil](https://github.com/yeyupiaoling/ClassificationForAndroid/blob/master/TFLiteClassification/app/src/main/java/com/yeyupiaoling/tfliteclassification/TFLiteClassificationUtil.java)工具类，关于Tensorflow Lite的操作都在这里完成，如加载模型、预测。在构造方法中，通过参数传递的模型路径加载模型，在加载模型的时候配置预测信息，例如是否使用Android底层神经网络API`NnApiDelegate`或者是否使用GPU`GpuDelegate`，同时获取网络的输入输出层。有了`tensorflow-lite-support`库，数据预处理就变得非常简单，通过`ImageProcessor`创建一个数据预处理的工具，之后在预测之前使用这个工具对图像进行预处理，处理速度还是挺快的，要注意的是图像的均值`IMAGE_MEAN`和标准差`IMAGE_STD`，因为在训练的时候图像预处理可能不一样的，有些读者出现在电脑上准确率很高，但在手机上准确率很低，多数情况下就是这个图像预处理做得不对。

```java
private static final float[] IMAGE_MEAN = new float[]{128.0f, 128.0f, 128.0f};
private static final float[] IMAGE_STD = new float[]{128.0f, 128.0f, 128.0f};

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
        int[] imageShape = tflite.getInputTensor(tflite.getInputIndex("input_1")).shape();
        DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("input_1")).dataType();
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
```

为了兼容图片路径和Bitmap格式的图片预测，这里创建了两个重载方法，它们都是通过调用`predict()`
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

这个方法就是Tensorflow Lite执行预测的最后一步，通过执行`tflite.run()`对输入的数据进行预测并得到预测结果，通过解析获取到最大的概率的预测标签，并返回。到这里Tensorflow Lite的工具就完成了。
```java
private int predict(Bitmap bmp) throws Exception {
    inputImageBuffer = loadImage(bmp);

    try {
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
    } catch (Exception e) {
        throw new Exception("predict image fail! log:" + e);
    }

    float[] results = outputProbabilityBuffer.getFloatArray();
    Log.d(TAG, Arrays.toString(results));
    return getMaxResult(results);
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
String classificationModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2.tflite";
Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2.tflite", classificationModelPath);
try {
    tfLiteClassificationUtil = new TFLiteClassificationUtil(classificationModelPath);
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

当打开相册选择照片之后，回到原来的页面，在下面这个回调方法中获取选择图片的Uri，通过Uri可以获取到图片的绝对路径。如果Android8以上的设备获取不到图片，需要在`AndroidManifest.xml`配置文件中的`application`添加`android:requestLegacyExternalStorage="true"`。拿到图片路径之后，调用`TFLiteClassificationUtil`类中的`predictImage()`方法预测并获取预测值，在页面上显示预测的标签、对应标签的名称、概率值和预测时间。
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
                float[] result = tfLiteClassificationUtil.predictImage(image_path);
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
在调用相机实时预测我就不再介绍了，原理都差不多，具体可以查看[https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/TFLiteClassification](https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/TFLiteClassification)中的源代码。核心代码如下，创建一个子线程，子线程中不断从摄像头预览的`AutoFitTextureView`上获取图像，并执行预测，并在页面上显示预测的标签、对应标签的名称、概率值和预测时间。每一次预测完成之后都立即获取图片继续预测，只要预测速度够快，就可以看成实时预测。
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
        float[] result = tfLiteClassificationUtil.predictImage(bitmap);
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

![选择图片识别效果图](https://s1.ax1x.com/2020/07/22/UHVKG8.jpg)

**相机实时识别效果图：**

![相机实时识别效果图](https://s1.ax1x.com/2020/07/22/UHVuPf.jpg)


