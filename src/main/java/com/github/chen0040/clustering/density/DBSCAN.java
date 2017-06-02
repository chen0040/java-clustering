package com.github.chen0040.clustering.density;

import com.github.chen0040.clustering.DistanceMeasureService;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.StringUtils;
import lombok.Getter;
import lombok.Setter;

import java.util.HashSet;
import java.util.Iterator;
import java.util.function.BiFunction;


/**
 * Created by xschen on 23/8/15.
 */
@Getter
@Setter
public class DBSCAN  {
    private BiFunction<DataRow, DataRow, Double> distanceMeasure;
    private double epsilon;
    private int minPts;
    private DataFrame model;

    public DBSCAN(){
        epsilon = 0.1;
        minPts = 10;
    }

    public int getMinPts() {
        return minPts;
    }

    public void setMinPts(int minPts) {
        this.minPts = minPts;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public DataFrame getModel(){
        return model;
    }

    public DataFrame fitAndTransform(DataFrame batch) {
        batch = batch.makeCopy();
        this.model = batch;

        int m = model.rowCount();

        boolean[] visited = new boolean[m];
        double[][] distanceMatrix = new double[m][];
        for(int i=0; i < m; ++i) {
            distanceMatrix[i] = new double[m];
        }

        for(int i=0; i < m; ++i){
            DataRow tuple_i = model.row(i);
            for(int j=i+1; j < m; ++j){
                DataRow tuple_j = model.row(j);
                double distance = DistanceMeasureService.getDistance(tuple_i, tuple_j, distanceMeasure);
                distanceMatrix[i][j] = distance;
                distanceMatrix[j][i] = distance;
            }
        }

        int C = -1;

        for(int i=0; i < m; ++i){
            if(visited[i]){
                continue;
            }
            visited[i] = true;
            HashSet<Integer> neighbors = regionQuery(i, epsilon, distanceMatrix);
            if(neighbors.size() < minPts){
                // mark i as NOISE
            }else{
                C++;
                expandCluster(i, neighbors, C, epsilon, minPts, visited, distanceMatrix, model);
            }
        }

        return batch;
    }

    private void add_to_cluster(int i, int clusterId, DataFrame batch){
        DataRow tuple = batch.row(i);
        tuple.setCategoricalTargetCell("cluster", "" + clusterId);
    }

    private boolean is_member_of_a_cluster(DataRow row){
       return !StringUtils.isEmpty(row.getCategoricalTargetCell("cluster"));
    }

    private void expandCluster(int i, HashSet<Integer> neighbors_i, int clusterId, double eps, int MinPts, boolean[] visited, double[][] distanceMatrix, DataFrame batch) {
        add_to_cluster(i, clusterId, batch);
        Iterator<Integer> piter = neighbors_i.iterator();
        while (piter.hasNext()) {
            int p = piter.next();
            if (!visited[p]) {
                visited[p] = true;
                HashSet<Integer> neighbor_pts_prime = regionQuery(p, eps, distanceMatrix);
                if (neighbor_pts_prime.size() >= MinPts) {
                    neighbors_i.addAll(neighbor_pts_prime);
                    piter = neighbors_i.iterator();
                }
            }
            if (!is_member_of_a_cluster(batch.row(p))) {
                add_to_cluster(p, clusterId, batch);
            }
        }
    }

    private HashSet<Integer> regionQuery(int i, double eps, double[][] distanceMatrix){
        int m = distanceMatrix.length;
        HashSet<Integer> neighbors = new HashSet<Integer>();
        for(int j = 0; j < m; ++j){
            if(i == j) continue;
            double distance = distanceMatrix[i][j];
            if(distance < eps){
                neighbors.add(j);
            }
        }

        return neighbors;
    }

}
