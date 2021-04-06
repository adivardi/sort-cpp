/**
 * SORT: A Simple, Online and Realtime Tracker
 */

#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

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

#include <nav_msgs/OccupancyGrid.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include "sort_cpp/tracker.h"
#include "sort_cpp/utils.h"

#include <enway_msgs/ObstacleArray.h>

bool k_filter_drivable = true;
std::string drivable_map_topic = "/navigation/enway_map/map_drivable_region";
std::unique_ptr<grid_map::GridMap> drivable_region_;

bool VIS_CLUSTERS_BY_TRACKS = false;
bool PRINT_TRACKS = true;

typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> PointCloud;

std::string processing_frame = "base_link";
std::string tracking_frame = "odom";

float voxel_size = 0.05;
float z_min = 0.45;
float z_max = 2.5;
float clustering_tolerance = 2.5;
float min_pts_in_cluster = 5;
float clustering_dist_2d_thresh = 2.0;
float clustering_dist_2d_thresh_sq = clustering_dist_2d_thresh * clustering_dist_2d_thresh;

bool don_filter = false;
double don_small_scale = 0.5;   // The small scale to use in the DoN filter.
double don_large_scale = 2.0;   // The large scale to use in the DoN filter.
double don_angle_thresh = 0.1; // The minimum DoN magnitude to threshold by

float tracking_distance_thresh = 2.5;
float tracking_max_distance = 3.0;  // must not be smaller than tracking_distance_thresh, otherwise tracker will
                                    // accept association with max distance (=> no distance threshold applied)
                                    // in order to avoid floating point error on equal, should set to a bit bigger

static tf2_ros::Buffer tf_buffer_;
// create SORT tracker
Tracker tracker;

std::vector<ros::Publisher> clusters_pubs_;
ros::Publisher proccessed_pub_;
ros::Publisher marker_pub_;
ros::Publisher obstacles_pub_;

void publishTrackAsMarker(const std_msgs::Header& header, const std::map<int, Track>& tracks);

void publishObstacles(const std_msgs::Header& header, const std::map<int, Track>& tracks);

bool
transformPointcloud(PointCloud& cloud, std::string frame)
{
  // transform_pointcloud to processing frame
  tf2_ros::TransformListener tf_listener_(tf_buffer_);
  try
  {
    constexpr double transform_wait_time {0.2};
    geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        frame, cloud.header.frame_id, fromPCL(cloud.header).stamp, ros::Duration{transform_wait_time});

    pcl_ros::transformPointCloud<PointXYZI>(cloud, cloud, transform.transform);
    cloud.header.frame_id = frame;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR_STREAM(ex.what());
    return false;
  }
  return true;
}

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

bool
filterDrivable(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // TODO Can merge remove nans and the filtering together. This way can save the copy Pointcloud inside the filtering

  // TODO check if inpout and processed point on the same object, then can skip copy
  PointCloud::Ptr temp_cloud(new PointCloud);
  pcl::copyPointCloud(*input, *temp_cloud);

  if (!drivable_region_)
  {
    ROS_WARN("Drivable region is not available. Skip filtering");
    processed = temp_cloud;
    return true;
  }

  // transform to map frame
  if (!transformPointcloud(*temp_cloud, drivable_region_->getFrameId()))
  {
    ROS_ERROR_STREAM("Failed to transform to " << drivable_region_->getFrameId());
    return false;
  }
  std::cout << "map frame_id: " << temp_cloud->header.frame_id << std::endl;

  // we cannot just iterate and erase points, as this will invalidate the iterator
  // method 1: mark all elements to be deleted with a special value. then use std::remove_if(begin, end, lambda: value==specialValue)
  // complexity: O(N) for first loop to mark elements + O(N) for remove_if  => O(2*N)
  // method 2: copy pointcloud into a temp cloud. Then delete processed->points. then go over all points in temp and copy only relevant ones intoprocessed
  // complexity: O(N) for 1st copy (+ copy!) + O(N) for nd copy.
  // maybe can be merged with the NaN removal, so should save a copy

  std::vector<int> keep_indices;
  keep_indices.reserve(temp_cloud->size());

  // for (const auto& point : temp_cloud->points)
  for (size_t i = 0; i < temp_cloud->size(); ++i)
  {
    const PointXYZI point = temp_cloud->points[i];
    const grid_map::Position position(point.x, point.y);

    // if outside drivable region, keep it
    if (!drivable_region_->isInside(position))
    {
      keep_indices.push_back(i);
      continue;
    }

    const float value = drivable_region_->atPosition("drivable_region", position);
    if (!std::isnan(value)) // not NaN => drivable => keep
    {
      keep_indices.push_back(i);
    }
  }

  pcl::copyPointCloud(*temp_cloud, keep_indices, *processed);

  std::cout << "input pts: " << temp_cloud->points.size() << std::endl;
  std::cout << "processed pts: " << processed->points.size() << std::endl;

  return true;
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

bool
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
  if (!transformPointcloud(*processed, processing_frame))
  {
    ROS_ERROR_STREAM("Failed to transform to " << processing_frame);
    return false;
  }
  std::cout << "processing frame_id: " << processed->header.frame_id << std::endl;

  // remove ground
  filterGround(processed, processed);

  if (don_filter)
  {
    differneceOfNormalsFiltering(processed, processed);
  }

  // filter non-drivable parts
  if (k_filter_drivable)
  {
    if (!filterDrivable(processed, processed))
    {
      return false;
    }
  }

  return true;
}

bool
clusterCondition(const PointXYZI& a, const PointXYZI& b, float  /*dist*/)
{
  // NOTE very similar results between 2D and 3D

  float dist_2d_sq = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
  return (dist_2d_sq < (clustering_dist_2d_thresh_sq));

  // float dist_3d_sq = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
  // return (dist_3d_sq < (1.6 * 1.6));
}

bool
clusterPointcloud(const PointCloud::Ptr& input, std::vector<pcl::PointIndices>& clusters_indices)
{
  // Cluster_indices is a vector containing one instance of PointIndices for each detected cluster.
  clusters_indices.clear();

  // transform to tracking frame
  if (!transformPointcloud(*input, tracking_frame))
  {
    ROS_ERROR_STREAM("Failed to transform to " << tracking_frame);
    return false;
  }
  std::cout << "tracking frame_id: " << input->header.frame_id << std::endl;

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

  return true;
}

void
cluster_and_track(const PointCloud::Ptr& processed_cloud)
{
  auto t2 = std::chrono::high_resolution_clock::now();

  std::vector<pcl::PointIndices> clusters_indices;
  if (!clusterPointcloud(processed_cloud, clusters_indices))
  {
    return;
  }

  std::cout << "cluster no.: " << clusters_indices.size() << std::endl;
  auto t3 = std::chrono::high_resolution_clock::now();

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

  auto t4 = std::chrono::high_resolution_clock::now();

  /*** Run SORT tracker ***/
  std::map<int, Tracker::Detection> track_to_detection_associations =
    tracker.Run(clusters_centroids, processed_cloud->header.stamp,
    tracking_distance_thresh * tracking_distance_thresh,
    tracking_max_distance * tracking_max_distance);
  /*** Tracker update done ***/

  auto t5 = std::chrono::high_resolution_clock::now();

  const auto tracks = tracker.GetTracks();

  std::cout << "dt: " << tracker.GetDT() << std::endl;

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
        double v = std::sqrt(state(3) * state(3) + state(4) * state(4) + state(5) * state(5));
        // Print to terminal for debugging
        std::cout << "track id: " << trk.first
                  << ", cluster id: " << track_to_detection_associations[trk.first].cluster_id
                  << ", state: " << state(0) << ", " << state(1) << ", " << state(2)
                  << ", v: " << v
                  << " (" << state(3) << ", " << state(4) << ", " << state(5) << ")"
                  << " Hit Streak = " << trk.second.hit_streak_
                  << " Coast Cycles = " << trk.second.coast_cycles_
                  // << " covariance = " << covar.transpose()
                  << std::endl;
      }
    }
  }

  auto t6 = std::chrono::high_resolution_clock::now();

  std_msgs::Header input_header = fromPCL(processed_cloud->header);

  if (marker_pub_.getNumSubscribers() > 0)
  {
    publishTrackAsMarker(input_header, tracks);
  }

  auto t7 = std::chrono::high_resolution_clock::now();

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

  auto t8 = std::chrono::high_resolution_clock::now();

  if (obstacles_pub_.getNumSubscribers() > 0)
  {
    publishObstacles(input_header, tracks);
  }

  auto t9 = std::chrono::high_resolution_clock::now();

  std::cout << "cluster      : " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;
  std::cout << "centroids    : " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;
  std::cout << "track        : " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " ms" << std::endl;
  std::cout << "print tracks : " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms" << std::endl;
  std::cout << "pub tracks   : " << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count() << " ms" << std::endl;
  std::cout << "pub clusters : " << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << " ms" << std::endl;
  std::cout << "pub obstacles: " << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count() << " ms" << std::endl;
  std::cout << "loop         : " << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t2).count() << " ms" << std::endl;
  std::cout << "cloud size : " << processed_cloud->points.size() << std::endl;
}

void
cloud_cb(const PointCloud::ConstPtr& input_cloud)
{
  std::cout << "++++++++++++++++++ cloud_cb +++++++++++++++++++++" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  PointCloud::Ptr processed_cloud(new PointCloud);
  if(!processPointCloud(input_cloud, processed_cloud))
  {
    return;
  }

  // publish processed pointcloud
  proccessed_pub_.publish(processed_cloud);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "process      : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
  std::cout << "input size   : " << input_cloud->points.size() << std::endl;

  cluster_and_track(processed_cloud);
}

void
map_callback(const nav_msgs::OccupancyGrid& input_map)
{
  std::cout << "============== map_callback ==============" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();

  // convert msg to GridMap
  drivable_region_ = std::make_unique<grid_map::GridMap>();
  bool success = grid_map::GridMapRosConverter::fromOccupancyGrid(input_map, "drivable_region", *drivable_region_);
  if (!success)
  {
    drivable_region_.reset();
    ROS_ERROR("Failed to convert OccupancyGrid msg to GridMap");
    return;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "got map      : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
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

  ros::Subscriber input_sub = nh.subscribe("pointcloud", 1, cloud_cb);
  ros::Subscriber map_sub = nh.subscribe(drivable_map_topic, 1, map_callback);

  for (int i = 0; i < 10; i++)
  {
    std::string topic = "cluster_" + std::to_string(i);
    ros::Publisher pub = nh.advertise<PointCloud>(topic, 1);
    clusters_pubs_.push_back(pub);
  }

  marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("tracks", 1);

  proccessed_pub_ = nh.advertise<PointCloud>("processed_pointcloud", 1);

  obstacles_pub_ = nh.advertise<enway_msgs::ObstacleArray>("obstacles", 1);

  ros::spin();
}

void
publishObstacles(const std_msgs::Header& header, const std::map<int, Track>& tracks)
{
  auto obstacles = boost::make_shared<enway_msgs::ObstacleArray>();

  for (const auto& track : tracks)
  {
    if (track.second.coast_cycles_ < kMaxCoastCycles && track.second.hit_streak_ >= kMinHits)
    {
      const auto state = track.second.GetState();

      enway_msgs::Obstacle obstacle;
      obstacle.header = header;
      obstacle.type = enway_msgs::Obstacle::OTHER;

      obstacle.pose.position.x = state(0);
      obstacle.pose.position.y = state(1);
      obstacle.pose.position.z = state(2);

      // get orientation from velocity direction
      tf::Vector3 velocity_vector (state(3), state(4), state(5));
      tf::Vector3 origin (1, 0, 0);

      // q.w = sqrt((origin.length() ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
      auto w = (origin.length() * velocity_vector.length()) + tf::tfDot(origin, velocity_vector);
      tf::Vector3 a = origin.cross(velocity_vector);
      tf::Quaternion q(a.x(), a.y(), a.z(), w);
      q.normalize();

      if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || !std::isfinite(q.w()))
      {
        q.setX(0);
        q.setY(0);
        q.setZ(0);
        q.setW(1);
      }
      // TODO add covariance to pose so can add the KF covariance ?

      obstacle.pose.orientation.w = q.w();
      obstacle.pose.orientation.x = q.x();
      obstacle.pose.orientation.y = q.y();
      obstacle.pose.orientation.z = q.z();

      obstacle.velocities.twist.linear.x = state(3);
      obstacle.velocities.twist.linear.y = state(4);
      obstacle.velocities.twist.linear.z = state(5);
      obstacle.velocities.twist.angular.x = 0;
      obstacle.velocities.twist.angular.y = 0;
      obstacle.velocities.twist.angular.z = 0;

      obstacle.size.x = 0;
      obstacle.size.y = 0;
      obstacle.size.z = 0;

      obstacles->obstacles.push_back(obstacle);
    }
  }

  obstacles_pub_.publish(obstacles);
}

void
publishTrackAsMarker(const std_msgs::Header& header, const std::map<int, Track>& tracks)
{
  visualization_msgs::MarkerArray array;

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

      // convert rpy to quaternion
      tf::Vector3 velocity_vector (state(3), state(4), state(5));
      tf::Vector3 origin (1, 0, 0);

      // q.w = sqrt((origin.length() ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
      auto w = (origin.length() * velocity_vector.length()) + tf::tfDot(origin, velocity_vector);
      tf::Vector3 a = origin.cross(velocity_vector);
      tf::Quaternion q(a.x(), a.y(), a.z(), w);
      q.normalize();

      heading.pose.orientation.w = q.w();
      heading.pose.orientation.x = q.x();
      heading.pose.orientation.y = q.y();
      heading.pose.orientation.z = q.z();

      const double v = std::sqrt(state(3) * state(3) + state(4) * state(4) + state(5) * state(5));

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
