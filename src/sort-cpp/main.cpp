/**
 * SORT: A Simple, Online and Realtime Tracker
 */

#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>

#include "sort-cpp/tracker.h"

bool VIS_CLUSTERS = true;

typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> PointCloud;

std::string map_frame = "base_link";
std::string base_frame = "base_link";
float voxel_size = 0.05;
float z_min = 0.4;
float z_max = 2.5;
float clustering_tolerance = 0.15;
float min_pts_in_cluster = 50;

// create SORT tracker
Tracker tracker;

std::vector<ros::Publisher> clusters_pubs_;
ros::Publisher proccessed_pub_;

void
filterGround(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // TODO can be improved using normals and height
  // currently just cut between z_min and z_max

  pcl::PassThrough<PointXYZI> pass_filter;
  pass_filter.setInputCloud(input);
  pass_filter.setFilterFieldName("z");
  pass_filter.setFilterLimits(z_min, z_max);
  // pass_filter.setFilterLimitsNegative(true);
  pass_filter.filter(*processed);
}

void
processPointCloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // remove invalid pts (NaN, Inf)
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*input, *processed, indices);

  // transform_pointcloud to map frame  // TODO actually needed?
  tf::TransformListener tf_listener_;
  tf_listener_.waitForTransform(
      processed->header.frame_id, map_frame, fromPCL(processed->header).stamp, ros::Duration(5.0));
  pcl_ros::transformPointCloud<PointXYZI>(map_frame, *processed, *processed, tf_listener_);

  // voxel filter
  pcl::VoxelGrid<PointXYZI> vox_filter;
  vox_filter.setInputCloud(processed);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  vox_filter.filter(*processed);

  // remove ground
  filterGround(processed, processed);
}

void
cloud_cb(const PointCloud::ConstPtr& input_cloud)
{
  PointCloud::Ptr processed_cloud(new PointCloud);
  processPointCloud(input_cloud, processed_cloud);
  processed_cloud->header = input_cloud->header;
  processed_cloud->header.frame_id = map_frame;

  // publish processed pointcloud
  proccessed_pub_.publish(processed_cloud);

  // Creating the KdTree from input point cloud
  pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
  tree->setInputCloud(processed_cloud);

  // clustering
  /* vector of PointIndices, which contains the actual index information in a vector<int>.
  * The indices of each detected cluster are saved here.
  * Cluster_indices is a vector containing one instance of PointIndices for each detected cluster. */
  std::vector<pcl::PointIndices> clusters_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> eucl_clustering;
  eucl_clustering.setClusterTolerance(clustering_tolerance);
  eucl_clustering.setMinClusterSize(min_pts_in_cluster);
  // eucl_clustering.setMaxClusterSize(600);
  eucl_clustering.setSearchMethod(tree);
  eucl_clustering.setInputCloud(processed_cloud);
  /* Extract the clusters out of pc and save indices in clusters_indices.*/
  eucl_clustering.extract(clusters_indices);

  std::cout << "cluster nu: " << clusters_indices.size() << std::endl;

  // get cluster centroid
  std::vector<PointCloud::Ptr> clusters;
  std::vector<Eigen::VectorXd> clusters_centroids;

  for (const pcl::PointIndices& cluster_ids : clusters_indices)
  // for (auto cluster_it = clusters_indices.begin(); cluster_it != clusters_indices.end(); ++cluster_it)
  {
    Eigen::Vector4f centroid_homogenous;
    pcl::compute3DCentroid(*processed_cloud, cluster_ids, centroid_homogenous); // in homogenous coords
    std::cout << "centroid homo: " << centroid_homogenous << std::endl;

    Eigen::VectorXd centroid_xyz;
    centroid_xyz << centroid_homogenous(0), centroid_homogenous(1), centroid_homogenous(2);
    std::cout << "centroid xyz: " << centroid_xyz << std::endl;

    clusters_centroids.push_back(centroid_xyz);

    // Extract the cluster pointcloud
    PointCloud::Ptr cluster_pointcloud(new PointCloud);
    pcl::copyPointCloud(*processed_cloud, cluster_ids.indices, *cluster_pointcloud);

    clusters.push_back(cluster_pointcloud);
  }

  /*** Run SORT tracker ***/
  tracker.Run(clusters_centroids);
  const auto tracks = tracker.GetTracks();
  /*** Tracker update done ***/

  if (VIS_CLUSTERS)
  {
    for (long unsigned int i = 0; i < clusters_pubs_.size(); i++)
    {
      clusters[i]->header.frame_id = map_frame;
      pcl_conversions::toPCL(ros::Time::now(), clusters[i]->header.stamp);
      clusters_pubs_[i].publish(clusters[i]);
    }
  }
}

int
main(int argc, char** argv)
{
  // ROS init
  ros::init(argc, argv, "kf_tracker");
  ros::NodeHandle nh;

  // Publishers to publish the state of the objects (pos and vel)
  // objState1=nh.advertise<geometry_msgs::Twist> ("obj_1",1);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber pointcloud_sub = nh.subscribe("pointcloud", 1, cloud_cb);

  for (int i = 0; i < 10; i++)
  {
    std::string topic = "cluster_" + std::to_string(i);
    ros::Publisher pub = nh.advertise<PointCloud>(topic, 1);
    clusters_pubs_.push_back(pub);
  }

  // cc_pos = nh.advertise<std_msgs::Float32MultiArray>("ccs", 100);
  // markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);

  proccessed_pub_ = nh.advertise<PointCloud>("processed_pointcloud", 1);

  ros::spin();
}
