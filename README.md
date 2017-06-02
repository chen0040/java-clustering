# java-clustering
Package provides java implementation of various clustering algorithms

[![Build Status](https://travis-ci.org/chen0040/java-clustering.svg?branch=master)](https://travis-ci.org/chen0040/java-clustering) [![Coverage Status](https://coveralls.io/repos/github/chen0040/java-clustering/badge.svg?branch=master)](https://coveralls.io/github/chen0040/java-clustering?branch=master)
 
# Features

* Hierarchical Clustering
* KMeans Clustering
* DBSCAN
 
# Install

Add the following dependency to your POM file:

```xml
<dependency>
  <groupId>com.github.chen0040</groupId>
  <artifactId>java-clustering</artifactId>
  <version>1.0.1</version>
</dependency>
```

### Image Segmentation (Clustering) using KMeans

The following sample code shows how to use FuzzyART to perform image segmentation:

```java
BufferedImage img= ImageIO.read(FileUtils.getResource("1.jpg"));

DataFrame dataFrame = ImageDataFrameFactory.dataFrame(img);

KMeans cluster = new KMeans();
DataFrame learnedData = cluster.fitAndTransform(dataFrame);

for(int i=0; i <learnedData.rowCount(); ++i) {
 ImageDataRow row = (ImageDataRow)learnedData.row(i);
 int x = row.getPixelX();
 int y = row.getPixelY();
 String clusterId = row.getCategoricalTargetCell("cluster");
 System.out.println("cluster id for pixel (" + x + "," + y + ") is " + clusterId);
}
```

The segmented image can be generated using the trained KMeans from above as illustrated by the following sample code:

```java

List<Integer> classColors = new ArrayList<Integer>();
for(int i=0; i < 5; ++i){
 for(int j=0; j < 5; ++j){
    classColors.add(ImageDataFrameFactory.get_rgb(255, rand.nextInt(255), rand.nextInt(255), rand.nextInt(255)));
 }
}

BufferedImage segmented_image = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
for(int x=0; x < img.getWidth(); x++)
{
 for(int y=0; y < img.getHeight(); y++)
 {
    int rgb = img.getRGB(x, y);

    DataRow tuple = ImageDataFrameFactory.getPixelTuple(x, y, rgb);

    int clusterIndex = cluster.transform(tuple);

    rgb = classColors.get(clusterIndex % classColors.size());

    segmented_image.setRGB(x, y, rgb);
 }
}
```
