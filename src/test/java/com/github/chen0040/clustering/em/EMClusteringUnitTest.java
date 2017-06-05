package com.github.chen0040.clustering.em;


import com.github.chen0040.clustering.onelink.SingleLinkageClustering;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.frame.Sampler;
import org.testng.annotations.Test;

import java.util.Random;

import static org.testng.Assert.*;


/**
 * Created by xschen on 5/6/2017.
 */
public class EMClusteringUnitTest {
   private static Random random = new Random();

   public static double rand(){
      return random.nextDouble();
   }

   public static double rand(double lower, double upper){
      return rand() * (upper - lower) + lower;
   }

   public static double randn(){
      double u1 = rand();
      double u2 = rand();
      double r = Math.sqrt(-2.0 * Math.log(u1));
      double theta = 2.0 * Math.PI * u2;
      return r * Math.sin(theta);
   }


   // unit testing based on example from http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#
   @Test
   public void testSimple(){


      DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
              .newInput("c1")
              .newInput("c2")
              .newOutput("designed")
              .end();

      Sampler.DataSampleBuilder negativeSampler = new Sampler()
              .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? 2 : 4))
              .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? 2 : 4))
              .forColumn("designed").generate((name, index) -> 0.0)
              .end();

      Sampler.DataSampleBuilder positiveSampler = new Sampler()
              .forColumn("c1").generate((name, index) -> rand(-4, -2))
              .forColumn("c2").generate((name, index) -> rand(-2, -4))
              .forColumn("designed").generate((name, index) -> 1.0)
              .end();

      DataFrame data = schema.build();

      data = negativeSampler.sample(data, 50);
      data = positiveSampler.sample(data, 50);

      System.out.println(data.head(10));

      EMClustering algorithm = new EMClustering();
      algorithm.setSigma0(1.5);
      algorithm.setClusterCount(2);

      DataFrame learnedData = algorithm.fitAndTransform(data);

      for(int i = 0; i < learnedData.rowCount(); ++i){
         DataRow tuple = learnedData.row(i);
         String clusterId = tuple.getCategoricalTargetCell("cluster");
         System.out.println("learned: " + clusterId +"\tknown: "+tuple.target());
      }


   }
}
