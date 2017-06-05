package com.github.chen0040.clustering;


/**
 * Created by xschen on 5/6/2017.
 */
public class Edge implements Comparable<Edge> {
   private int v;
   private int w;
   private double weight;


   public Edge(int v, int w, double weight) {
      this.v = v;
      this.w = w;
      this.weight = weight;
   }

   public int either() {
      return v;
   }

   public int other(int x) {
      return x == v ? w : v;
   }

   public double getWeight() {
      return weight;
   }


   @Override public int compareTo(Edge that) {
      return Double.compare(this.getWeight(), that.getWeight());
   }
}
