package com.github.chen0040.clustering.onelink;

import com.github.chen0040.clustering.DistanceMeasureService;
import com.github.chen0040.clustering.Edge;
import com.github.chen0040.clustering.MinPQ;
import com.github.chen0040.clustering.QuickUnion;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.Getter;
import lombok.Setter;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;


/**
 * Created by xschen on 19/8/15.
 */
@Getter
@Setter
public class SingleLinkageClustering {
    private int clusterCount = 10;
    private BiFunction<DataRow, double[], Double> distanceMeasure;

    private double getDistance(Cluster c1, Cluster c2){
        List<DataRow> ct1 = c1.getPoints();
        List<DataRow> ct2 = c2.getPoints();
        int m1 = ct1.size();
        int m2 = ct2.size();
        double min_distance = Double.MAX_VALUE;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2.toArray(), distanceMeasure);
                min_distance = Math.min(min_distance, distance);
            }
        }

        return min_distance;
    }

    public DataFrame fitAndTransform(DataFrame dataFrame) {

        DataFrame batch = dataFrame.makeCopy();
        int m = batch.rowCount();
        Cluster[] clusters = new Cluster[m];
        for(int i = 0; i < m; ++i){
            DataRow tuple = batch.row(i);
            clusters[i] = new Cluster(tuple, i);
        }

        MinPQ<Edge> pq = new MinPQ<>();
        QuickUnion uf =new QuickUnion(clusters.length);
        for(int j=0; j < clusters.length; ++j){
            Cluster cluster_j = clusters[j];
            for(int k=j+1; k < clusters.length; ++k){
                Cluster cluster_k = clusters[k];
                double distance = getDistance(cluster_j, cluster_k);
                pq.enqueue(new Edge(j, k,distance));
            }
        }

        List<Edge> mst = new ArrayList<>();
        while(!pq.isEmpty() && mst.size() < (m-clusterCount)){

            Edge e = pq.delMin();
            int select_j = e.either();
            int select_k = e.other(select_j);

            if(!uf.connected(select_j, select_k)) {
                uf.union(select_j, select_k);
                mst.add(e);
            }
        }


        Set<Integer> set = new HashSet<>();
        for(int i=0; i < mst.size(); ++i) {
            Edge e = mst.get(i);
            int j = e.either();
            set.add(uf.id(j));
        }

        Map<Integer, Integer> clusterIds = new HashMap<>();
        for(Integer i : set) {
            clusterIds.put(i, clusterIds.size());
        }


        for(int i=0; i < clusters.length; ++i){
            clusters[i].setIndex(clusterIds.get(uf.id(i)));
        }

        return batch;
    }

    private class Cluster{
        private int index;
        private final List<DataRow> points = new ArrayList<>();;

        public Cluster(DataRow tuple, int index){
            this.index = index;
            add(tuple);
        }

        public void setIndex(int index){
            this.index = index;
            for(DataRow row : points){
                row.setCategoricalTargetCell("cluster", "" + index);
            }
        }

        @Override
        public int hashCode(){
            return this.index;
        }

        @Override
        public boolean equals(Object rhs){
            if(rhs instanceof Cluster){
                Cluster cast_rhs = (Cluster) rhs;
                return cast_rhs.index == index;
            }
            return false;
        }

        public void add(Cluster cluster){
            if(index == cluster.index) return;

            for(DataRow tuple : cluster.getPoints()){
                add(tuple);
            }
            cluster.getPoints().clear();
        }

        private void add(DataRow tuple){
            this.points.add(tuple);
            tuple.setCategoricalTargetCell("cluster", "" + index);
        }

        public List<DataRow> getPoints(){
            return points;
        }
    }
}
