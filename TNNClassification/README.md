

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

# 开发Android项目


2. 把上一步编译得到的`include`目录复制到Android项目的`app`目录下。

3. 把上一步编译得到的`armeabi-v7a`和`arm64-v8a`目录复制到`main/jniLibs`下