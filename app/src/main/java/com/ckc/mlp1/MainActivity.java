package com.ckc.mlp1;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.TextView;

import com.ckc.mlp1.ml.DogCat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView;
    private Button select,predict;
    private TextView tv;
    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String[] lables = new String[1000];
        int cnt=0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line = bufferedReader.readLine();
            while (line != null && cnt < 1000) {
                lables[cnt] = line;
                cnt++;
                line = bufferedReader.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        imgView = findViewById(R.id.selectedimg);
        select = findViewById(R.id.selectIMFG);
        tv = findViewById(R.id.resulttv);
        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);

            }
        });

        predict = findViewById(R.id.predict);
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


                try {
                    DogCat model = DogCat.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
                    bitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);


                    TensorImage tensorImage = new TensorImage(DataType.UINT8);
                    tensorImage.load(bitmap);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    DogCat.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();

                    tv.setText(lables[getMax(outputFeature0.getFloatArray())]+" ");
                } catch (IOException e) {
                    // TODO Handle the exception
                }


            }
        });
    }

    int getMax(float[] arr)
    {
        int max =0;
        for (int i =0;i<arr.length;i++)
        {
            if (arr[i]>arr[max]) max = i;
        }
        return max;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode==100)
        {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
    }
}