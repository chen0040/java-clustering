package com.github.chen0040.clustering.utils;


import java.io.InputStream;


/**
 * Created by xschen on 30/5/2017.
 */
public class FileUtils {
   public static InputStream getResource(String filename){
      return FileUtils.class.getClassLoader().getResourceAsStream(filename);
   }
}
