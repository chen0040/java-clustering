package com.github.chen0040.clustering.hierarchical;


import com.github.chen0040.clustering.DistanceMeasureService;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;


/**
 * Created by xschen on 23/8/15.
 */
@Getter
@Setter
public class HierarchicalClustering {

    private int clusterCount = 1;

    private LinkageCriterion linkage = LinkageCriterion.AverageLinkage;

    private BiFunction<DataRow, DataRow, Double> distanceMeasure;


    private double getDistance(DataFrame context, Cluster c1, Cluster c2) {
        switch (linkage){

            case CompleteLinkage:
                return getDistance_CompleteLinkage(context, c1, c2);
            case SingleLinkage:
                return getDistance_SingleLinkage(context, c1, c2);
            case AverageLinkage:
                return getDistance_AverageLinkage(context, c1, c2);
            case CentroidLinkage:
                return getDistance_CentroidLinkage(context, c1, c2);
            case MinimumEnergyClustering:
                return getDistance_MinimuEnergyClustering(context, c1, c2);
        }

        return Double.MAX_VALUE;
    }

    private double getDistance_MinimuEnergyClustering(DataFrame context, Cluster c1, Cluster c2){
        List<DataRow> ct1 = c1.getTuples();
        List<DataRow> ct2 = c2.getTuples();
        int m1 = ct1.size();
        int m2 = ct2.size();

        double part1 = 0;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
                double norm2_distance = Math.pow(distance, 2);
                part1 += norm2_distance;
            }
        }

        double part2 = 0;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m1; ++j){
                DataRow tuple2 = ct1.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
                double norm2_distance = Math.pow(distance, 2);
                part2 += norm2_distance;
            }
        }

        double part3 = 0;
        for(int i=0; i < m2; ++i){
            DataRow tuple1 = ct2.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
                double norm2_distance = Math.pow(distance, 2);
                part3 += norm2_distance;
            }
        }

        return 2 * part1 / (m1 * m2) - part2 / (m1 * m1) - part3 / (m2 * m2);
    }

    private double getDistance_CentroidLinkage(DataFrame context, Cluster c1, Cluster c2){
        return DistanceMeasureService.euclideanDistance(c1.getCentroid(context), c2.getCentroid(context));
    }

    private double getDistance_CompleteLinkage(DataFrame context, Cluster c1, Cluster c2){
        List<DataRow> ct1 = c1.getTuples();
        List<DataRow> ct2 = c2.getTuples();
        int m1 = ct1.size();
        int m2 = ct2.size();
        double max_distance = Double.MIN_VALUE;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
                max_distance = Math.max(max_distance, distance);
            }
        }

        return max_distance;
    }

    private double getDistance_AverageLinkage(DataFrame context, Cluster c1, Cluster c2){
        List<DataRow> ct1 = c1.getTuples();
        List<DataRow> ct2 = c2.getTuples();
        int m1 = ct1.size();
        int m2 = ct2.size();
        double avg_distance = 0;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                avg_distance += DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
            }
        }

        avg_distance /= (m1*m2);

        return avg_distance;
    }

    private double getDistance_SingleLinkage(DataFrame context, Cluster c1, Cluster c2){
        List<DataRow> ct1 = c1.getTuples();
        List<DataRow> ct2 = c2.getTuples();
        int m1 = ct1.size();
        int m2 = ct2.size();
        double min_distance = Double.MAX_VALUE;
        for(int i=0; i < m1; ++i){
            DataRow tuple1 = ct1.get(i);
            for(int j=0; j < m2; ++j){
                DataRow tuple2 = ct2.get(j);
                double distance = DistanceMeasureService.getDistance(tuple1, tuple2, distanceMeasure);
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
                    double distance = getDistance(batch, cluster_j, cluster_k);
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

    public enum LinkageCriterion {
        CompleteLinkage,
        SingleLinkage,
        AverageLinkage,
        CentroidLinkage, // aka UPGMC
        MinimumEnergyClustering
    }

    private class Cluster{
        private int index;
        private List<DataRow> tuples;

        public Cluster(DataRow tuple, int index){
            this.index = index;
            tuples = new ArrayList<>();
            add(tuple);
        }

        double[] getCentroid(DataFrame context){


            int m = tuples.size();
            int n = context.row(0).toArray().length;
            double[] centroid = new double[n];

            for(int i=0; i < m; ++i){
                DataRow tuple = tuples.get(i);
                double[] x = tuple.toArray();
                for(int j=0; j < n; ++j) {
                    centroid[j] += x[j];
                }
            }

            for(int i=0; i < n; ++i){
                centroid[i] /= m;
            }
            return centroid;
        }


        void setIndex(int index){
            this.index = index;
            for(DataRow row : tuples){
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
            for(DataRow tuple : cluster.getTuples()){
                add(tuple);
            }
            cluster.getTuples().clear();
        }

        private void add(DataRow tuple){
            this.tuples.add(tuple);
            tuple.setCategoricalTargetCell("cluster", "" + index);
        }

        public List<DataRow> getTuples(){
            return tuples;
        }
    }
}
