package com.github.chen0040.clustering;


/**
 * Created by xschen on 5/6/2017.
 */
public class QuickUnion {
   private int N;
   private int[] id;
   public QuickUnion(int N) {
      this.N = N;
      this.id = new int[N];
      for(int i=0; i < N; ++i) {
         id[i]  = i;
      }

   }

   public void union(int v, int w){
      int vroot = root(v);
      int wroot = root(w);
      if(vroot != wroot) {
         id[wroot] = vroot;
      }
   }

   public boolean connected(int v, int w) {
      return root(v) == root(w);
   }

   private int root(int v) {
      while(id[v] != v){
         v = id[v];
      }
      return v;
   }

   public int id(int v){
      return root(v);
   }


}
