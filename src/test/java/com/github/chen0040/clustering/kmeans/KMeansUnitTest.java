package com.github.chen0040.clustering.kmeans;


import com.github.chen0040.clustering.utils.FileUtils;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.image.ImageDataFrameFactory;
import com.github.chen0040.data.image.ImageDataRow;
import org.testng.annotations.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.testng.Assert.*;


/**
 * Created by xschen on 30/5/2017.
 */
public class KMeansUnitTest {

   private static Random rand = new Random();

   @Test
   public void test_image_segmentation(){
      BufferedImage img=null;
      try{
         img= ImageIO.read(FileUtils.getResource("1.jpg"));
      }catch(IOException ie)
      {
         ie.printStackTrace();
      }

      DataFrame data = ImageDataFrameFactory.dataFrame(img);

      KMeans cluster = new KMeans();
      cluster.setMaxIters(200);
      cluster.setClusterCount(25);

      DataFrame learnedData = cluster.fitAndTransform(data);
      for(int i=0; i <learnedData.rowCount(); ++i) {
         ImageDataRow row = (ImageDataRow)learnedData.row(i);
         int x = row.getPixelX();
         int y = row.getPixelY();
         String clusterId = row.getCategoricalTargetCell("cluster");
         System.out.println("cluster id for pixel (" + x + "," + y + ") is " + clusterId);
      }

      List<Integer> classColors = new ArrayList<>();
      for(int i=0; i < 5; ++i){
         for(int j=0; j < 5; ++j){
            classColors.add(ImageDataFrameFactory.get_rgb(255, rand.nextInt(255), rand.nextInt(255), rand.nextInt(255)));
         }
      }

      BufferedImage class_img = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
      for(int x=0; x < img.getWidth(); x++)
      {
         for(int y=0; y < img.getHeight(); y++)
         {
            int rgb = img.getRGB(x, y);

            DataRow tuple = ImageDataFrameFactory.getPixelTuple(x, y, rgb);

            int clusterIndex = cluster.transform(tuple);

            rgb = classColors.get(clusterIndex);

            class_img.setRGB(x, y, rgb);
         }
      }
   }
}
