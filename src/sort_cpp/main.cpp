/**
 * SORT: A Simple, Online and Realtime Tracker
 */

#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

// #include <pcl/filters/extract_indices.h>
#include <pcl/features/don.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include "sort_cpp/tracker.h"
#include "sort_cpp/utils.h"

bool VIS_CLUSTERS_BY_TRACKS = false;
bool PRINT_TRACKS = true;

typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> PointCloud;

std::string processing_frame = "base_link";
std::string tracking_frame = "map";

float voxel_size = 0.05;
float z_min = 0.45;
float z_max = 2.5;
float clustering_tolerance = 2.5;
float min_pts_in_cluster = 5;
float clustering_dist_2d_thresh = 0.7;

bool don_filter = false;
double don_small_scale = 0.5;   // The small scale to use in the DoN filter.
double don_large_scale = 2.0;   // The large scale to use in the DoN filter.
double don_angle_thresh = 0.1; // The minimum DoN magnitude to threshold by

float tracking_distance_thresh = 1.0;
float tracking_max_distance = 1.0;

static tf2_ros::Buffer tf_buffer_;

// create SORT tracker
Tracker tracker;

std::vector<ros::Publisher> clusters_pubs_;
ros::Publisher proccessed_pub_;
ros::Publisher marker_pub_;

void publishTrackAsMarker(const std::string& frame_id, const std::map<int, Track> tracks, double dt);

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

void differneceOfNormalsFiltering(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  std::cout << "input: " << input->points.size() << std::endl;

  pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
  tree->setInputCloud(input);

  // Compute normals using both small and large scales at each point
  pcl::NormalEstimationOMP<PointXYZI, pcl::PointNormal> normal_estimation;
  normal_estimation.setInputCloud(input);
  normal_estimation.setSearchMethod(tree);

  // setting viewpoint is very important, so that we can ensure normals are all pointed in the same direction!
  normal_estimation.setViewPoint(std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

  // calculate normals with the small scale
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
  normal_estimation.setRadiusSearch(don_small_scale);
  normal_estimation.compute(*normals_small_scale);

  // calculate normals with the large scale
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);
  normal_estimation.setRadiusSearch(don_large_scale);
  normal_estimation.compute(*normals_large_scale);

  // Create output cloud for DoN results
  pcl::PointCloud<pcl::PointNormal>::Ptr doncloud(new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*input, *doncloud);

  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<PointXYZI, pcl::PointNormal, pcl::PointNormal> diffnormals_estimator;
  diffnormals_estimator.setInputCloud(input);
  diffnormals_estimator.setNormalScaleLarge(normals_large_scale);
  diffnormals_estimator.setNormalScaleSmall(normals_small_scale);

  if (!diffnormals_estimator.initCompute())
  {
    std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
    exit (EXIT_FAILURE);
  }

  // Compute DoN
  diffnormals_estimator.computeFeature(*doncloud);

  // filter based on DoN
  pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
  // filter curvature (= l2 norm of the normal ?) > thresh => keep only points with high curvature
  range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr (
                              new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, don_angle_thresh))
                            );
  // Build the filter
  pcl::ConditionalRemoval<pcl::PointNormal> conditional_filter;
  conditional_filter.setCondition(range_cond);
  conditional_filter.setInputCloud(doncloud);

  pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

  // Apply filter
  conditional_filter.filter(*doncloud_filtered);

  pcl::copyPointCloud<pcl::PointNormal, PointXYZI>(*doncloud_filtered, *processed);
  std::cout << "dof: " << processed->points.size() << std::endl;
}

void
processPointCloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // remove invalid pts (NaN, Inf)
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*input, *processed, indices);

  std::cout << "input frame_id: " << input->header.frame_id << std::endl;

  // voxel filter
  pcl::VoxelGrid<PointXYZI> vox_filter;
  vox_filter.setInputCloud(processed);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  vox_filter.filter(*processed);

  // transform_pointcloud to processing frame
  tf2_ros::TransformListener tf_listener_(tf_buffer_);
  try
  {
    constexpr double transform_wait_time {0.2};
    geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        processing_frame, processed->header.frame_id, fromPCL(processed->header).stamp, ros::Duration{transform_wait_time});

    pcl_ros::transformPointCloud<PointXYZI>(*processed, *processed, transform.transform);
    processed->header.frame_id = processing_frame;
    std::cout << "process frame_id: " << processed->header.frame_id << std::endl;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR_STREAM(ex.what());
    return;
  }

  // remove ground
  filterGround(processed, processed);

  if (don_filter)
  {
    differneceOfNormalsFiltering(processed, processed);
  }
}

bool
clusterCondition(const PointXYZI& a, const PointXYZI& b, float  /*dist*/)
{
  // NOTE very similar results between 2D and 3D

  float dist_2d_sq = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
  return (dist_2d_sq < (clustering_dist_2d_thresh * clustering_dist_2d_thresh));

  // float dist_3d_sq = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
  // return (dist_3d_sq < (1.6 * 1.6));
}

void
clusterPointcloud(const PointCloud::Ptr& input, std::vector<pcl::PointIndices>& clusters_indices)
{
  // Cluster_indices is a vector containing one instance of PointIndices for each detected cluster.
  clusters_indices.clear();

  // transform to tracking frame
  tf2_ros::TransformListener tf_listener_(tf_buffer_);
  try
  {
    constexpr double transform_wait_time {0.2};
    geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        tracking_frame, input->header.frame_id, fromPCL(input->header).stamp, ros::Duration{transform_wait_time});

    pcl_ros::transformPointCloud<PointXYZI>(*input, *input, transform.transform);
    input->header.frame_id = tracking_frame;
    std::cout << "tracking frame_id: " << input->header.frame_id << std::endl;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR_STREAM(ex.what());
    return;
  }

  // Creating the KdTree from input point cloud
  pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
  tree->setInputCloud(input);

  // clustering
  pcl::ConditionalEuclideanClustering<PointXYZI> clusters_extractor;
  clusters_extractor.setClusterTolerance(clustering_tolerance);
  clusters_extractor.setMinClusterSize(min_pts_in_cluster);    // TODO change to percentage of the cluster size
  // clusters_extractor.setMaxClusterSize(600);
  // clusters_extractor.setSearchMethod(tree);  // Doesn't exist for conditional clustering

  clusters_extractor.setConditionFunction(&clusterCondition);//[](const PointXYZI& a, const PointXYZI& b, float  /*dist*/)
                                          // {
                                          //   return clusterCondition(a, b);
                                          // });

  clusters_extractor.setInputCloud(input);
  /* Extract the clusters out of pc and save indices in clusters_indices.*/
  clusters_extractor.segment(clusters_indices);   // extract for notmal clustering
}

void
cloud_cb(const PointCloud::ConstPtr& input_cloud)
{
  std::cout << "------------------------------" << std::endl;
  PointCloud::Ptr processed_cloud(new PointCloud);
  processPointCloud(input_cloud, processed_cloud);

  // publish processed pointcloud
  proccessed_pub_.publish(processed_cloud);

  std::vector<pcl::PointIndices> clusters_indices;
  clusterPointcloud(processed_cloud, clusters_indices);

  std::cout << "cluster no.: " << clusters_indices.size() << std::endl;

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
  std::map<int, Tracker::Detection> track_to_detection_associations = tracker.Run(clusters_centroids, input_cloud->header.stamp, tracking_distance_thresh * tracking_distance_thresh, tracking_max_distance * tracking_max_distance);
  /*** Tracker update done ***/

  const auto tracks = tracker.GetTracks();
  const double dt = static_cast<double>(tracker.GetDT()) * 1e-6;

  std::cout << "dt: " << dt << std::endl;

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
        double vkf = std::sqrt(state(3) * state(3) + state(4) * state(4) + state(5) * state(5));
        double v = vkf / dt;
        // Print to terminal for debugging
        std::cout << "track id: " << trk.first
                  << ", cluster id: " << track_to_detection_associations[trk.first].cluster_id
                  << ", state: " << state(0) << ", " << state(1) << ", " << state(2)
                  << ", v: " << v
                  << ", vkf: " << vkf
                  << " (" << state(3) << ", " << state(4) << ", " << state(5) << ")"
                  << " Hit Streak = " << trk.second.hit_streak_
                  << " Coast Cycles = " << trk.second.coast_cycles_ << std::endl;
      }
    }
  }

  if (marker_pub_.getNumSubscribers() > 0)
  {
    publishTrackAsMarker(processed_cloud->header.frame_id, tracks, dt);
  }

  if (VIS_CLUSTERS_BY_TRACKS)
  {
    long unsigned int i = 0;
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
  else
  {
    for (long unsigned int i = 0; i < clusters.size(); i++)
    {
      if (i >= clusters_pubs_.size())
      {
        break;
      }
      clusters_pubs_[i].publish(clusters[i]);
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

  // get 1st transform
  tf2_ros::TransformListener tf_listener_(tf_buffer_);
  try
  {
    constexpr double transform_wait_time {10.0};
    geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        tracking_frame, "front_mid", ros::Time{0.0}, ros::Duration{transform_wait_time});
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR_STREAM(ex.what());
    throw ex;
  }
  std::cout << "got 1st transform!" << std::endl;

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


void
publishTrackAsMarker(const std::string& frame_id, const std::map<int, Track> tracks, double dt)
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

      const double vkf = std::sqrt(state(3) * state(3) + state(4) * state(4) + state(5) * state(5));
      const double v = vkf / dt;

      constexpr double k_arrow_shaft_diameter = 0.15;
      heading.scale.x = v;
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

      text.scale.z = 2;

      array.markers.push_back(text);
    }
  }

  marker_pub_.publish(array);
}
