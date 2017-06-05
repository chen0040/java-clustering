package com.github.chen0040.clustering;


/**
 * Created by xschen on 5/6/2017.
 */
public class MinPQ<T extends Comparable<T>>  {

   private T[] s;
   private int N = 0;
   public MinPQ(){
      s = (T[])new Comparable[20];
   }

   public void enqueue(T item) {
      if(this.N + 1 == this.s.length){
         this.resize(this.s.length * 2);
      }
      this.s[++this.N] = item;
      this.swim(this.N);
   }

   public T delMin() {
      if(this.N == 0) {
         return null;
      }

      T item = this.s[1];
      exchange(s, 1, N--);
      if(N == s.length / 4) {
         resize(s.length / 2);
      }
      sink(1);
      return item;
   }

   public boolean isEmpty() {
      return N == 0;
   }

   private void sink(int k) {
      while(k * 2 <= N) {
         int child = k * 2;
         if(child < N && less(s[child+1], s[child])) {
            child++;
         }
         if(less(s[child], s[k])) {
            exchange(s, child, k);
            k = child;
         } else {
            break;
         }
      }
   }


   private void swim(int k) {
      while(k > 1) {
         int parent = k / 2;
         if(less(s[k], s[parent])){
            exchange(s, k, parent);
            k = parent;
         } else {
            break;
         }
      }
   }

   private static <T> void exchange(T[] a, int i, int j){
      T temp = a[i];
      a[i] = a[j];
      a[j] = temp;
   }

   private boolean less(T a1, T a2) {
      return a1.compareTo(a2) < 0;
   }

   private void resize(int len) {
      T[] temp = (T[])new Comparable[len];
      for(int i=0; i < Math.min(len, s.length); ++i) {
         temp[i] = this.s[i];
      }
      s = temp;
   }
}
