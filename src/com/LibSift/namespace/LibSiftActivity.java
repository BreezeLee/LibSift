package com.LibSift.namespace;


//import org.opencv.android.LoaderCallbackInterface;

import java.io.File;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;

public class LibSiftActivity extends Activity {
    private static final String TAG = null;

	/** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        File sdDir = null; 
        boolean sdCardExist = Environment.getExternalStorageState().equals(android.os.Environment.MEDIA_MOUNTED);
        if(sdCardExist)   
        {                               
          sdDir = Environment.getExternalStorageDirectory();
       }   
        
 String fileName1 = sdDir.toString() +"/classroom1.jpg" ;
 File f1= new File(sdDir.toString() +"/classroom1.jpg");
 boolean temp1 = f1.exists();
 String fileName2 = sdDir.toString() +"/classroom2.jpg" ;
 File f2= new File(sdDir.toString() +"/classroom2.jpg");
 boolean temp2 = f2.exists();
//        char[] namea=a.toCharArray();
//        char[] nameb=b.toCharArray();
        SiftFun.siftConjunction(fileName1, fileName2);
//        
    }

}