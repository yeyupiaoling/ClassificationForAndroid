> 原文博客：[Doi技术团队](http://blog.doiduoyi.com)<br/>
> 链接地址：[https://blog.doiduoyi.com/authors/1584446358138](https://blog.doiduoyi.com/authors/1584446358138)<br/>
> 初心：记录优秀的Doi技术团队学习经历<br/>
>本文链接：[基于TNN在Android手机上实现图像分类](https://blog.doiduoyi.com/articles/1599362726469.html)<br/>



# 前言

TNN：由腾讯优图实验室打造，移动端高性能、轻量级推理框架，同时拥有跨平台、高性能、模型压缩、代码裁剪等众多突出优势。TNN框架在原有Rapidnet、ncnn框架的基础上进一步加强了移动端设备的支持以及性能优化，同时也借鉴了业界主流开源框架高性能和良好拓展性的优点。


# 编译Android库

1. 安装cmake 3.12
```bash
# 卸载旧的cmake
sudo apt-get autoremove cmake

# 下载cmake3.12
wget https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.tar.gz
tar zxvf cmake-3.12.2-Linux-x86_64.tar.gz

# 移动目录并添加软连接
sudo mv cmake-3.12.2-Linux-x86_64 /opt/cmake-3.12.2
sudo ln -sf /opt/cmake-3.12.2/bin/*  /usr/bin/
```

2. 添加Android NDK

```bash
wget https://dl.google.com/android/repository/android-ndk-r21b-linux-x86_64.zip
unzip android-ndk-r21b-linux-x86_64.zip
# 添加环境变量，留意你实际下载地址
export ANDROID_NDK=/mnt/d/android-ndk-r21b
```

3. 安装编译环境

```bash
sudo apt-get install attr
```

4. 开始编译

```bash
git clone https://github.com/Tencent/TNN.git
cd TNN/scripts

vim build_android.sh
```


```bash
 ABIA32="armeabi-v7a"
 ABIA64="arm64-v8a"
 STL="c++_static"
 SHARED_LIB="ON"                # ON表示编译动态库，OFF表示编译静态库
 ARM="ON"                       # ON表示编译带有Arm CPU版本的库
 OPENMP="ON"                    # ON表示打开OpenMP
 OPENCL="ON"                    # ON表示编译带有Arm GPU版本的库
 SHARING_MEM_WITH_OPENGL=0      # 1表示OpenGL的Texture可以与OpenCL共享
```

执行编译
```bash
./build_android.sh
```

编译完成后，会在当前目录的`release`目录下生成对应的`armeabi-v7a`库，`arm64-v8a`库和`include`头文件，这些文件在下一步的Android开发都需要使用到。

# 模型转换
接下来我们需要把Tensorflow，onnx等其他的模型转换为TNN的模型。目前 TNN 支持业界主流的模型文件格式，包括ONNX、PyTorch、TensorFlow 以及 Caffe 等。TNN 将 ONNX 作为中间层，借助于ONNX 开源社区的力量，来支持多种模型文件格式。如果要将PyTorch、TensorFlow 以及 Caffe 等模型文件格式转换为 TNN，首先需要使用对应的模型转换工具，统一将各种模型格式转换成为 ONNX 模型格式，然后将 ONNX 模型转换成 TNN 模型。

```bash
sudo docker pull turandotkay/tnn-convert
sudo docker tag turandotkay/tnn-convert:latest tnn-convert:latest
sudo docker rmi turandotkay/tnn-convert:latest
```

针对不同的模型转换，有不同的命令，如onnx2tnn，caffe2tnn，tf2tnn。
```bash
docker run --volume=$(pwd):/workspace -it tnn-convert:latest  python3 ./converter.py tf2tnn \
    -tp /workspace/mobilenet_v1.pb \
    -in "input[1,224,224,3]" \
    -on MobilenetV1/Predictions/Reshape_1 \
    -v v1.0 \
    -optimize
```

通过上面的输出，可以发现针对 TF 模型的转换，convert2tnn 工具提供了很多参数，我们一次对下面的参数进行解释：

- tp 参数（必须）
    通过 “-tp” 参数指定需要转换的模型的路径。目前只支持单个 TF模型的转换，不支持多个 TF 模型的一起转换。
- in 参数（必须）
    通过 “-in” 参数指定模型输入的名称，输入的名称需要放到“”中，例如，-in "name"。如果模型有多个输入，请使用 “;”进行分割。有的 TensorFlow 模型没有指定 batch 导致无法成功转换为 ONNX 模型，进而无法成功转换为 TNN 模型。你可以通过在名称后添加输入 shape 进行指定。shape 信息需要放在 [] 中。例如：-in "name[1,28,28,3]"。
- on 参数（必须）
    通过 “-on” 参数指定模型输入的名称，如果模型有多个输出，请使用 “;”进行分割
- output_dir 参数：
    可以通过 “-o <path>” 参数指定输出路径，但是在 docker 中我们一般不使用这个参数，默认会将生成的 TNN 模型放在当前和 TF 模型相同的路径下。
- optimize 参数（可选）
    可以通过 “-optimize” 参数来对模型进行优化，**我们强烈建议你开启这个选项，只有在开启这个选项模型转换失败时，我们才建议你去掉 “-optimize” 参数进行重新尝试**。
- v 参数（可选）
    可以通过 -v 来指定模型的版本号，以便于后期对模型进行追踪和区分。
- half 参数（可选）
    可以通过 -half 参数指定，模型数据通过 FP16 进行存储，减少模型的大小，默认是通过 FP32 的方式进行存储模型数据的。
- align 参数（可选）
    可以通过 -align 参数指定，将 转换得到的 TNN 模型和原模型进行对齐，确定 TNN 模型是否转换成功。__当前仅支持单输入单输出模型和单输入多输出模型。 align 只支持 FP32 模型的校验，所以使用 align 的时候不能使用 half__
- input_file 参数（可选）
    可以通过 -input_file 参数指定模型对齐所需要的输入文件的名称，输入需要遵循如下[格式](#输入)。
- ref_file 参数（可选）
    可以通过 -ref_file 参数指定待对齐的输出文件的名称，输出需遵循如下[格式](#输出)。生成输出的代码可以[参考](#生成输出示例代码)。


成功转换会输出以下的日志。
```
----------  convert model, please wait a moment ----------

Converter Tensorflow to TNN model

Convert TensorFlow to ONNX model succeed!

Converter ONNX to TNN Model

Converter ONNX to TNN model succeed!
```

最终会得到这两个模型文件，`mobilenet_v1.opt.tnnmodel` `mobilenet_v1.opt.tnnproto`。

# 开发Android项目

1. 将转换的模型放在`assets`目录下。
2. 把上一步编译得到的`include`目录复制到Android项目的`app`目录下。
3. 把上一步编译得到的`armeabi-v7a`和`arm64-v8a`目录复制到`main/jniLibs`下。
4. 在`app/src/main/cpp/`目录下编写JNI的C++代码。


## TNN工具
编写一个[ImageClassifyUtil.java](https://github.com/yeyupiaoling/ClassificationForAndroid/blob/master/TNNClassification/app/src/main/java/com/yeyupiaoling/tnnclassification/tnn/ImageClassifyUtil.java)工具类，关于TNN的操作都在这里完成，如加载模型、预测。


下面三个就是TNN的JNI接口，通过这个接口完成模型加载，预测，当不使用的时候和可以调用`deinit()`清空对象。
```java
public native int init(String modelPath, String protoPath, int computeUnitType);

public native float[] predict(Bitmap image, int width, int height);

public native int deinit();
```

通过上面的JNI接口，下面就可以实现图像识别了，`WIDTH`和`HEIGHT`是模型输入图片的大小。为了兼容图片路径和Bitmap格式的图片预测，这里创建了两个重载方法。
```java
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

不同的模型，训练的预处理方式可能不一样，TNN 的图像预处理在C++中完成，[代码片段](https://github.com/yeyupiaoling/ClassificationForAndroid/blob/8b673db08f0a97b8ba1578f8ac049c2748c407cb/TNNClassification/app/src/main/cpp/imageclassify_jni.cc#L76-L80)。
```cpp
TNN_NS::MatConvertParam input_cvt_param;
input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
auto status = instance_->SetInputMat(input_mat, input_cvt_param);
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

在`MainActivity.java`中，进入到页面我们就要先加载模型，我们是把模型放在Android项目的assets目录的，我们需要把模型复制到一个缓存目录，然后再从缓存目录加载模型，同时还有读取标签名，标签名称按照训练的label顺序存放在assets的`label_list.txt`，以下为实现代码。
```java
classNames = Utils.ReadListFromFile(getAssets(), "label_list.txt");
String protoContent = getCacheDir().getAbsolutePath() + File.separator + "squeezenet_v1.1.tnnproto";
Utils.copyFileFromAsset(MainActivity.this, "squeezenet_v1.1.tnnproto", protoContent);
String modelContent = getCacheDir().getAbsolutePath() + File.separator + "squeezenet_v1.1.tnnmodel";
Utils.copyFileFromAsset(MainActivity.this, "squeezenet_v1.1.tnnmodel", modelContent);

imageClassifyUtil = new ImageClassifyUtil();
int status = imageClassifyUtil.init(modelContent, protoContent, USE_GPU ? 1 : 0);
if (status == 0){
    Toast.makeText(MainActivity.this, "模型加载成功！", Toast.LENGTH_SHORT).show();
}else {
    Toast.makeText(MainActivity.this, "模型加载失败！", Toast.LENGTH_SHORT).show();
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
                float[] result = imageClassifyUtil.predictImage(image_path);
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
        float[] result = imageClassifyUtil.predictImage(bitmap);
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

**效果图：** 
![在这里插入图片描述](https://s1.ax1x.com/2020/09/06/wZvzN9.jpg)
