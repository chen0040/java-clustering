package com.github.chen0040.clustering.kmeans;

import com.github.chen0040.clustering.DistanceMeasureService;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.*;
import java.util.function.BiFunction;


/**
 * Created by xschen on 17/8/15.
 */
@Getter
@Setter
public class KMeans {
    private static final Random random = new Random();

    @Setter(AccessLevel.NONE)
    private final Map<Integer, double[]> clusters = new HashMap<>();

    private int maxIters = 2000;
    private int clusterCount = 5;

    private BiFunction<DataRow, double[], Double> distanceMeasure;

    public int transform(DataRow tuple)
    {
        double minDistance = Double.MAX_VALUE;
        int closestClusterIndex = -1;
        for (int j = 0; j < clusterCount; ++j) {
            double[] clusterCenter = clusters.get(j);
            double distance = DistanceMeasureService.getDistance(tuple, clusterCenter, distanceMeasure);
            if (minDistance > distance) {
                minDistance = distance;
                closestClusterIndex = j;
            }
        }
        return closestClusterIndex;
    }

    private void initializeCluster(DataFrame batch){
        if(clusters.size() != clusterCount) {
            HashSet<Integer> indexList = new HashSet<Integer>();
            int m = batch.rowCount();
            if (m < clusterCount * 3) {
                clusterCount = Math.min(m, clusterCount);
                for (int i = 0; i < clusterCount; ++i) {
                    indexList.add(i);
                }
            } else {

                while (indexList.size() < clusterCount) {
                    int r = random.nextInt(m);
                    if (!indexList.contains(r)) {
                        indexList.add(r);
                    }
                }
            }

            clusters.clear();

            // initialize cluster centers
            int clusterIndex = 0;
            for (Integer i : indexList) {

                DataRow t = batch.row(i);
                clusters.put(clusterIndex++, t.toArray());
            }
        }
    }

    public DataFrame fitAndTransform(DataFrame dataFrame) {

        DataFrame batch = dataFrame.makeCopy();

        initializeCluster(batch);
        int m = batch.rowCount();

        for(int iter = 0; iter < maxIters; ++iter) {
            Cluster[] cg = new Cluster[clusterCount];
            for (int i = 0; i < clusterCount; ++i) {
                cg[i] = new Cluster();
            }

            // do clustering
            for (int i = 0; i < m; ++i) {
                DataRow tuple = batch.row(i);

                double minDistance = Double.MAX_VALUE;
                int closestClusterIndex = -1;
                for (int j = 0; j < clusterCount; ++j) {
                    double[] clusterCenter = clusters.get(j);
                    double distance = DistanceMeasureService.getDistance(tuple, clusterCenter, distanceMeasure);
                    if (minDistance > distance) {
                        minDistance = distance;
                        closestClusterIndex = j;
                    }
                }

                cg[closestClusterIndex].append(tuple);
                tuple.setCategoricalTargetCell("cluster", String.format("%d", closestClusterIndex));
            }

            //readjust cluster center
            for(int i=0; i < clusterCount; ++i){
                double[] newCenter = cg[i].calcCenter(batch);
                if(newCenter != null){
                    clusters.put(i, newCenter);
                }
            }

        }

        return batch;
    }

    private class Cluster {
        private final List<DataRow> elements = new ArrayList<>();

        void append(DataRow tuple){
            elements.add(tuple);
        }

        double[] calcCenter(DataFrame context){
            if(elements.isEmpty()) return null;
            int n = context.row(0).toArray().length;
            double[] newCenter = new double[n];
            int m = elements.size();
            for(int j=0; j < m; ++j){
                double[] x = context.row(j).toArray();
                for(int i=0; i < n; ++i) {
                    newCenter[i] += x[i];
                }
            }

            for(int i=0; i < n; ++i){

                newCenter[i] /= m;
            }
            return newCenter;
        }
    }
}
