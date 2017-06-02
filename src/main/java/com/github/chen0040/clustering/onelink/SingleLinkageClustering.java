package com.github.chen0040.clustering.onelink;

import com.github.chen0040.clustering.DistanceMeasureService;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;


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

        int remainingClusterCount = m;
        for(int i=0; i < (m-clusterCount); ++i){
            remainingClusterCount--;
            Cluster[] newClusters = new Cluster[remainingClusterCount];

            int select_j = -1;
            int select_k = -1;
            double min_distance = Double.MAX_VALUE;
            for(int j=0; j < clusters.length; ++j){
                Cluster cluster_j = clusters[j];
                for(int k=j+1; k < clusters.length; ++k){
                    Cluster cluster_k = clusters[k];
                    double distance = getDistance(cluster_j, cluster_k);
                    if(distance < min_distance){
                        select_j = j;
                        select_k = k;
                        min_distance = distance;
                    }
                }
            }

            int newIndex = 0;
            for(int l=0; l < clusters.length; ++l){
                if(l != select_j && l != select_k){
                    newClusters[newIndex++] = clusters[l];
                }
            }

            clusters[select_j].add(clusters[select_k]);
            newClusters[newIndex] = clusters[select_j];

            clusters = newClusters;
        }

        for(int i=0; i < clusters.length; ++i){
            clusters[i].setIndex(i);
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
