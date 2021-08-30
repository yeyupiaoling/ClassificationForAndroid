package com.yeyupiaoling.mnnclassification;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.yeyupiaoling.mnnclassification.mnn.MNNClassification;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private MNNClassification mnnClassification;
    private MNNClassification mnnClassificationPrune;
    private MNNClassification mnnClassificationQat;
    private ImageView imageView;
    private TextView textView;
    private TextView textViewPrune;
    private TextView textViewQat;
    private ArrayList<String> classNames;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!hasPermission()) {
            requestPermission();
        }

        // 加载模型和标签
        classNames = Utils.ReadListFromFile(getAssets(), "label_list.txt");
        String classificationModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2_new.mnn";
        Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2_new.mnn", classificationModelPath);
        String classificationPruneModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2_prune_final.mnn";
        Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2_prune_final.mnn", classificationPruneModelPath);
        String classificationQatModelPath = getCacheDir().getAbsolutePath() + File.separator + "mobilenet_v2_prune_final_quan0.mnn";
        Utils.copyFileFromAsset(MainActivity.this, "mobilenet_v2_prune_final_quan0.mnn", classificationQatModelPath);
        try {
            mnnClassification = new MNNClassification(classificationModelPath);
            mnnClassificationPrune = new MNNClassification(classificationPruneModelPath);
            mnnClassificationQat = new MNNClassification(classificationQatModelPath);
            Toast.makeText(MainActivity.this, "模型加载成功！", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(MainActivity.this, "模型加载失败！", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            finish();
        }

        // 获取控件
        Button selectImgBtn = findViewById(R.id.select_img_btn);
        Button openCamera = findViewById(R.id.open_camera);
        imageView = findViewById(R.id.image_view);
        textView = findViewById(R.id.result_text);
        textViewPrune = findViewById(R.id.result_text_prune);
        textViewQat = findViewById(R.id.result_text_qat);
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
    }

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
                    float[] result2 = mnnClassificationQat.predictImage(image_path);
                    long end1 = System.currentTimeMillis();

                    float[] result0 = mnnClassification.predictImage(image_path);

                    long end2 = System.currentTimeMillis();
                    float[] result1 = mnnClassificationPrune.predictImage(image_path);
                    long end3 = System.currentTimeMillis();
                    String show_text = "预测结果标签：" + (int) result0[0] +
                            "\n名称：" +  classNames.get((int) result0[0]) +
                            "\n时间：" + (end1 - start) + "ms";
                    textView.setText(show_text);
                    show_text = "预测结果标签：" + (int) result1[0] +
                            "\n名称：" +  classNames.get((int) result1[0]) +
                            "\n时间：" + (end2 - end1) + "ms";
                    textViewPrune.setText(show_text);
                    show_text = "预测结果标签：" + (int) result2[0] +
                            "\n名称：" +  classNames.get((int) result2[0]) +
                            "\n时间：" + (end3 - end2) + "ms";
                    textViewQat.setText(show_text);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // 根据相册的Uri获取图片的路径
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
}