/*
 * Created by ishaanjav
 * github.com/ishaanjav
 */

package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;


public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Call the superclass method
        super.onCreate(savedInstanceState);
        // Set the content view to the main activity layout
        setContentView(R.layout.activity_main);

        // Initialize the button and view references
        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Set a click listener on the "Take Picture" button
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Check if camera permission is granted
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    // Create an intent to capture an image using the camera
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    // Start the camera activity and wait for the result with request code 3
                    startActivityForResult(cameraIntent, 3);
                } else {
                    // Request camera permission if not granted
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        // Set a click listener on the "Launch Gallery" button
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Create an intent to pick an image from the gallery
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                // Start the gallery activity and wait for the result with request code 1
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            // Create an instance of the model using the application context
            Model model = Model.newInstance(getApplicationContext());

            // Create input tensor for the model with the specified size and data type
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            // Create a byte buffer to store the image data (32x32 pixels, 3 channels: RGB)
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Retrieve pixel values from the input image
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            // Loop through each pixel and extract the red, green, and blue values. Then, add each of these values separately to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }
            // Load the byte buffer into the input tensor
            inputFeature0.loadBuffer(byteBuffer);

            // Run model inference using the input tensor and get the output
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Get the confidence scores for each class
            float[] confidences = outputFeature0.getFloatArray();

            // Find the index of the class with the highest confidence score
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            // Define the class names corresponding to the model's output
            String[] classes = {"Apple", "Banana", "Orange"};
            // Set the text of the result view to the class with the highest confidence
            result.setText(classes[maxPos]);

            // Iterate through each class name, reversing each string and printing the results
            for (String fruit : classes) {
                // Call reverseString() method for each element
                String reversed = reverseString(fruit);

                // Print the reversed string
                System.out.println("Original: " + fruit + " | Reversed: " + reversed);
            }

            // Release model resources when done using them
            model.close();
        } catch (IOException e) {
            // Handle the exception
        }
    }

    public static String reverseString(String input) {
        // Check if the input is null or empty
        if (input == null || input.isEmpty()) {
            return input;
        }

        // Create a StringBuilder with the input string
        StringBuilder sb = new StringBuilder(input);

        // Reverse the string using the reverse() method
        sb.reverse();

        // Convert the reversed StringBuilder back to a String and return
        return sb.toString();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // Check if the result is successful
        if(resultCode == RESULT_OK){
            // Check the request code to determine the source of the activity result
            if(requestCode == 3){
                // For request code 3 (camera intent)
                // Retrieve the bitmap image from the intent extras
                Bitmap image = (Bitmap) data.getExtras().get("data");
                // Find the minimum dimension (width or height) of the image
                int dimension = Math.min(image.getWidth(), image.getHeight());
                // Create a thumbnail of the image using the minimum dimension
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                // Set the thumbnail image to the ImageView
                imageView.setImageBitmap(image);

                // Scale the image to the desired size (imageSize x imageSize)
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                // Classify the image using the `classifyImage` method
                classifyImage(image);
            }else{
                // For other request codes (e.g., gallery intent)
                // Retrieve the URI of the selected image
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    // Load the bitmap image from the URI using the content resolver
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    // Handle any IOExceptions that may occur during image loading
                    e.printStackTrace();
                }
                // Set the loaded image to the ImageView
                imageView.setImageBitmap(image);

                // Scale the image to the desired size (imageSize x imageSize)
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                // Classify the image using the `classifyImage` method
                classifyImage(image);
            }
        }
        // Call the superclass's onActivityResult method
        super.onActivityResult(requestCode, resultCode, data);
    }
}