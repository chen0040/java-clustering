package com.github.chen0040.clustering.em;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.Getter;
import lombok.Setter;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;


/**
 * Created by xschen on 18/8/15.
 * Expectation Maximization Clustering, Note that this is a soft clustering method
 */

public class EMClustering {

    private static final Random random = new Random();

    @Getter
    @Setter
    protected double sigma0 = 0.1;

    @Getter
    @Setter
    protected int clusterCount = 10;

    protected double[][] expectactionMatrix;
    protected double[][] clusters;

    @Getter
    @Setter
    private int maxIters = 2000;


    public EMClustering()
    {

    }

    public double getDistance(double[] x1, double[] x2){
        int n = x1.length;
        double sum = 0;
        for(int i=0; i < n; ++i){
            sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
        }

        return Math.sqrt(sum);
    }

    public double calcExpectation(DataRow tuple, int clusterIndex)
    {
        double[] P=new double[clusterCount];
        double sum = 0;
        for (int k = 0; k < clusterCount; ++k)
        {
            double distance = getDistance(tuple.toArray(), clusters[k]);


            P[k] = Math.exp(-Math.sqrt(distance) / (2 * sigma0 * sigma0));
            sum += P[k];
        }
        return P[clusterIndex] / sum;
    }

    private void initializeCluster(DataFrame batch, int n){
        Set<Integer> indexList = new HashSet<>();
        int m = batch.rowCount();
        if(m < clusterCount * 3){
            clusterCount = Math.min(m, clusterCount);
            for(int i=0; i < clusterCount; ++i){
                indexList.add(i);
            }
        }else {

            while (indexList.size() < clusterCount) {
                int r = random.nextInt(m);
                if (!indexList.contains(r)) {
                    indexList.add(r);
                }
            }
        }

        clusters = new double[clusterCount][];
        for(int i=0; i < clusterCount; ++i){
            clusters[i] = new double[n];
        }

        // initialize cluster centers
        int clusterIndex = 0;
        for(Integer i : indexList) {
            double[] center = clusters[clusterIndex];
            DataRow t = batch.row(i);
            double[] x = t.toArray();
            for(int j=0; j < n; ++j){
                center[j] = x[j];
            }
            clusterIndex++;
        }
    }

    private void initializeEM(DataFrame batch){
        int m = batch.rowCount();
        expectactionMatrix = new double[m][];
        for(int i=0; i< m; ++i){
            expectactionMatrix[i] = new double[clusterCount];
        }
    }


    public DataFrame fitAndTransform(DataFrame batch)
    {
        batch = batch.makeCopy();
        int m = batch.rowCount();

        int n = batch.row(0).toArray().length;

        initializeCluster(batch, n);
        initializeEM(batch);

        for (int iteration = 0; iteration < maxIters; ++iteration)
        {
            //expectation step
            for (int i = 0; i < m; ++i)
            {
                for (int k = 0; k < clusterCount; ++k)
                {
                    expectactionMatrix[i][k] = calcExpectation(batch.row(i), k);
                }
            }

            //maximization step
            for (int k = 0; k < clusterCount; ++k)
            {
                for (int d = 0; d < n; ++d)
                {
                    double denom = 0;
                    double num = 0;
                    for (int i = 0; i < m; ++i)
                    {
                        DataRow tuple = batch.row(i);
                        double[] x = tuple.toArray();
                        num += expectactionMatrix[i][k] * x[d];
                        denom += expectactionMatrix[i][k];
                    }
                    double mu = num / denom;
                    clusters[k][d] = mu;
                }
            }

            break;

        }

        for(int i=0; i < m; ++i){
            int max_k = -1;

            double max_expectation = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < clusterCount; ++k)
            {
                double expectation = expectactionMatrix[i][k];
                if (expectation > max_expectation)
                {
                    max_expectation = expectation;
                    max_k = k;
                }
            }
            DataRow tuple = batch.row(i);
            tuple.setCategoricalTargetCell("cluster", "" + max_k);
        }

        return batch;
    }


}


