/**
 * SORT: A Simple, Online and Realtime Tracker
 */

#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

// #include <pcl/filters/extract_indices.h>
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
#include <visualization_msgs/MarkerArray.h>

#include "sort_cpp/tracker.h"
#include "sort_cpp/utils.h"

bool VIS_CLUSTERS = true;
bool PRINT_TRACKS = true;

typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> PointCloud;

std::string tracking_frame = "base_link";
std::string base_frame = "base_link";
float voxel_size = 0.05;
float z_min = 0.5;
float z_max = 2.5;
float clustering_tolerance = 0.5;
float min_pts_in_cluster = 50;
float distance_thresh = 1.0;

// create SORT tracker
Tracker tracker;

std::vector<ros::Publisher> clusters_pubs_;
ros::Publisher proccessed_pub_;
ros::Publisher marker_pub_;

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
      processed->header.frame_id, tracking_frame, fromPCL(processed->header).stamp, ros::Duration(5.0));
  pcl_ros::transformPointCloud<PointXYZI>(tracking_frame, *processed, *processed, tf_listener_);

  // voxel filter
  pcl::VoxelGrid<PointXYZI> vox_filter;
  vox_filter.setInputCloud(processed);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  vox_filter.filter(*processed);

  // remove ground
  filterGround(processed, processed);
}

void
publishTrackAsMarker(const std::string& frame_id, const std::map<int, Track> tracks)
{
  visualization_msgs::MarkerArray array;
  std_msgs::Header header;
  header.frame_id = frame_id;
  header.stamp = ros::Time::now();

  for (const auto& track : tracks)
  {
    if (track.second.coast_cycles_ < kMaxCoastCycles && track.second.hit_streak_ >= kMinHits)
    {
      const auto state = track.second.GetState();

      visualization_msgs::Marker heading;
      heading.header = header;
      heading.ns = "Dynamic obstacle headings";
      heading.action = visualization_msgs::Marker::ADD;
      heading.id = track.first;
      heading.type = visualization_msgs::Marker::ARROW;
      heading.color.r = 1.0;
      heading.color.a = 1.0;

      constexpr double k_marker_lifetime_sec = 0.5;
      heading.lifetime = ros::Duration(k_marker_lifetime_sec);

      heading.pose.position.x = state(0);
      heading.pose.position.y = state(1);
      heading.pose.position.z = state(2);

      tf::Vector3 speed_direction (state(3), state(4), state(5));
      tf::Vector3 origin (1, 0, 0);

      // q.w = sqrt((origin.length() ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
      auto w = (origin.length() * speed_direction.length()) + tf::tfDot(origin, speed_direction);

      tf::Vector3 a = origin.cross(speed_direction);

      tf::Quaternion q(a.x(), a.y(), a.z(), w);
      q.normalize();

      heading.pose.orientation.w = q.w();
      heading.pose.orientation.x = q.x();
      heading.pose.orientation.y = q.y();
      heading.pose.orientation.z = q.z();

      const double speed = std::sqrt(state(3) * state(3)
                                      + state(4) * state(4)
                                      + state(5) * state(5));

      constexpr double k_arrow_shaft_diameter = 0.15;
      heading.scale.x = speed;
      heading.scale.y = k_arrow_shaft_diameter;
      heading.scale.z = k_arrow_shaft_diameter;

      array.markers.push_back(heading);

      visualization_msgs::Marker text;
      text.header = header;
      text.ns = "Dynamic obstacle texts";
      text.action = visualization_msgs::Marker::ADD;
      text.id = track.first;
      text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.color.g = 1.0;
      text.color.a = 1.0;
      text.lifetime = ros::Duration(k_marker_lifetime_sec);

      text.text = std::to_string(track.first);

      text.pose.position.x = state(0);
      text.pose.position.y = state(1);
      text.pose.position.z = state(2);

      text.pose.orientation.w = q.w();
      text.pose.orientation.x = q.x();
      text.pose.orientation.y = q.y();
      text.pose.orientation.z = q.z();

      text.scale.z = 3;

      array.markers.push_back(text);
    }
  }

  marker_pub_.publish(array);
}

void
cloud_cb(const PointCloud::ConstPtr& input_cloud)
{
  PointCloud::Ptr processed_cloud(new PointCloud);
  processPointCloud(input_cloud, processed_cloud);
  processed_cloud->header = input_cloud->header;
  processed_cloud->header.frame_id = tracking_frame;
  // std::cout << "process frame: " << processed_cloud->header.frame_id << std::endl;

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

  std::cout << "------------------------------" << std::endl;
  std::cout << "cluster nu: " << clusters_indices.size() << std::endl;

  // get cluster centroid
  std::map<int, PointCloud::Ptr> clusters;
  // std::vector<Eigen::VectorXd> clusters_centroids;
  std::vector<Tracker::Detection> clusters_centroids;

  int i = 0;
  for (const pcl::PointIndices& cluster_ids : clusters_indices)
  // for (auto cluster_it = clusters_indices.begin(); cluster_it != clusters_indices.end(); ++cluster_it)
  {
    Eigen::Vector4f centroid_homogenous;
    pcl::compute3DCentroid(*processed_cloud, cluster_ids, centroid_homogenous); // in homogenous coords

    Eigen::VectorXd centroid_xyz(3);
    centroid_xyz << centroid_homogenous(0), centroid_homogenous(1), centroid_homogenous(2);

    // clusters_centroids.push_back(centroid_xyz);
    Tracker::Detection det;
    det.centroid = centroid_xyz;
    det.cluster_id = i;
    clusters_centroids.push_back(det);

    // Extract the cluster pointcloud
    PointCloud::Ptr cluster_pointcloud(new PointCloud);
    pcl::copyPointCloud(*processed_cloud, cluster_ids.indices, *cluster_pointcloud);

    // clusters.push_back(cluster_pointcloud);
    clusters[i] = cluster_pointcloud;
    i++;
  }

  /*** Run SORT tracker ***/
  std::map<int, Tracker::Detection> track_to_detection_associations = tracker.Run(clusters_centroids, distance_thresh);
  const auto tracks = tracker.GetTracks();
  /*** Tracker update done ***/

  if (PRINT_TRACKS)
  {
    for (auto &trk : tracks)
    {
      const auto state = trk.second.GetState();
      // Note that we will not export coasted tracks
      // If we export coasted tracks, the total number of false negative will decrease (and maybe ID switch)
      // However, the total number of false positive will increase more (from experiments),
      // which leads to MOTA decrease
      // Developer can export coasted cycles if false negative tracks is critical in the system
      if (trk.second.coast_cycles_ < kMaxCoastCycles && trk.second.hit_streak_ >= kMinHits)
      {
          // Print to terminal for debugging
          std::cout << "track id: " << trk.first
                    << ", cluster id: " << track_to_detection_associations[trk.first].cluster_id
                    << ", state: " << state(0) << ", " << state(1) << ", " << state(2)
                    << ", v: " << std::sqrt(state(3) * state(3) + state(4) * state(4) + state(5) * state(5))
                    << " (" << state(3) << ", " << state(4) << ", " << state(5) << ")"
                    << " Hit Streak = " << trk.second.hit_streak_
                    << " Coast Cycles = " << trk.second.coast_cycles_ << std::endl;
      }
    }
  }

  if (marker_pub_.getNumSubscribers() > 0)
  {
    publishTrackAsMarker(processed_cloud->header.frame_id, tracks);
  }

  if (VIS_CLUSTERS)
  {
    int i = 0;
    for (auto &trk : tracks)
    {
      if (trk.second.coast_cycles_ < kMaxCoastCycles && trk.second.hit_streak_ >= kMinHits)
      {
        if (i >= clusters_pubs_.size())
        {
          break;
        }

        int cluster_id = track_to_detection_associations[trk.first].cluster_id;
        PointCloud::Ptr cluster = clusters[cluster_id];

        clusters_pubs_[i].publish(cluster);

        i++;
      }
    }
  }
}

int
main(int argc, char** argv)
{
  // ROS init
  ros::init(argc, argv, "sort_cpp_tracker");
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

  marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("tracks", 1);

  proccessed_pub_ = nh.advertise<PointCloud>("processed_pointcloud", 1);

  ros::spin();
}
