#include "sort_cpp/tracker.h"

#include <Eigen/Dense>
#include <map>

Tracker::Tracker() : dt_{0.0}
{
    id_ = 0;
}

// float Tracker::CalculateIou(const cv::Rect& det, const Track& track) {
//     auto trk = track.GetStateAsBbox();
//     // get min/max points
//     auto xx1 = std::max(det.tl().x, trk.tl().x);
//     auto yy1 = std::max(det.tl().y, trk.tl().y);
//     auto xx2 = std::min(det.br().x, trk.br().x);
//     auto yy2 = std::min(det.br().y, trk.br().y);
//     auto w = std::max(0, xx2 - xx1);
//     auto h = std::max(0, yy2 - yy1);

//     // calculate area of intersection and union
//     float det_area = det.area();
//     float trk_area = trk.area();
//     auto intersection_area = w * h;
//     float union_area = det_area + trk_area - intersection_area;
//     auto iou = intersection_area / union_area;
//     return iou;
// }

double Tracker::CalculateDistSquared(const Eigen::VectorXd& det, const Track& track) {
    auto track_state = track.GetState();

    double delta_x = track_state(0) - det(0);
    double delta_y = track_state(1) - det(1);
    double delta_z = track_state(2) - det(2);

    return (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}


/*
void Tracker::HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
                                size_t nrows, size_t ncols,
                                std::vector<std::vector<float>>& association) {
    Matrix<float> matrix(nrows, ncols);
    // Initialize matrix with IOU values
    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            // Multiply by -1 to find max cost
            if (iou_matrix[i][j] != 0) {
                matrix(i, j) = -iou_matrix[i][j];
            }
            else {
                // TODO: figure out why we have to assign value to get correct result
                matrix(i, j) = 1.0f;
            }
        }
    }

//    // Display begin matrix state.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(10);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;


    // Apply Kuhn-Munkres algorithm to matrix.
    Munkres<float> m;
    m.solve(matrix);

//    // Display solved matrix.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(2);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            association[i][j] = matrix(i, j);
        }
    }
}
*/
void Tracker::HungarianMatching(const std::vector<std::vector<float>>& dist_matrix,
                                size_t nrows, size_t ncols,
                                std::vector<std::vector<float>>& association) {
    Matrix<float> matrix(nrows, ncols);
    // Initialize matrix with dist values
    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            matrix(i, j) = dist_matrix[i][j];   // we want to find minimum distance
        }
    }

//    // Display begin matrix state.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(10);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;


    // Apply Kuhn-Munkres algorithm to matrix.
    Munkres<float> m;
    m.solve(matrix);

//    // Display solved matrix.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(2);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            association[i][j] = matrix(i, j);
        }
    }
}

// void Tracker::AssociateDetectionsToTrackers(const std::vector<cv::Rect>& detection,
//                                             std::map<int, Track>& tracks,
//                                             std::map<int, cv::Rect>& matched,
//                                             std::vector<cv::Rect>& unmatched_det,
                                            // float iou_threshold)
// {
//     // Set all detection as unmatched if no tracks existing
//     if (tracks.empty()) {
//         for (const auto& det : detection) {
//             unmatched_det.push_back(det);
//         }
//         return;
//     }

//     std::vector<std::vector<float>> iou_matrix;
//     // resize IOU matrix based on number of detection and tracks
//     iou_matrix.resize(detection.size(), std::vector<float>(tracks.size()));

//     std::vector<std::vector<float>> association;
//     // resize association matrix based on number of detection and tracks
//     association.resize(detection.size(), std::vector<float>(tracks.size()));


//     // row - detection, column - tracks
//     for (size_t i = 0; i < detection.size(); i++) {
//         size_t j = 0;
//         for (const auto& trk : tracks) {
//             iou_matrix[i][j] = CalculateIou(detection[i], trk.second);
//             j++;
//         }
//     }

//     // Find association
//     HungarianMatching(iou_matrix, detection.size(), tracks.size(), association);

//     for (size_t i = 0; i < detection.size(); i++) {
//         bool matched_flag = false;
//         size_t j = 0;
//         for (const auto& trk : tracks) {
//             if (0 == association[i][j]) {
//                 // Filter out matched with low IOU
//                 if (iou_matrix[i][j] >= iou_threshold) {
//                     matched[trk.first] = detection[i];
//                     matched_flag = true;
//                 }
//                 // It builds 1 to 1 association, so we can break from here
//                 break;
//             }
//             j++;
//         }
//         // if detection cannot match with any tracks
//         if (!matched_flag) {
//             unmatched_det.push_back(detection[i]);
//         }
//     }
// }

void Tracker::AssociateDetectionsToTrackers(const std::vector<Detection>& detection,
                                            std::map<int, Track>& tracks,
                                            std::map<int, Detection>& matched,
                                            std::vector<Detection>& unmatched_det,
                                            const float dist_threshold_sq,
                                            const float max_distance_sq)
{
    // Set all detection as unmatched if no tracks existing
    if (tracks.empty()) {
        for (const auto& det : detection) {
            unmatched_det.push_back(det);
        }
        return;
    }

    // row - detection, column - tracks
    std::vector<std::vector<float>> dist_matrix;
    // resize IOU matrix based on number of detection and tracks
    dist_matrix.resize(detection.size(), std::vector<float>(tracks.size()));

    std::vector<std::vector<float>> association;
    // resize association matrix based on number of detection and tracks
    association.resize(detection.size(), std::vector<float>(tracks.size()));


    // row - detection, column - tracks
    for (size_t i = 0; i < detection.size(); i++) {
        size_t j = 0;
        for (const auto& trk : tracks) {
            double dist_sq = CalculateDistSquared(detection[i].centroid, trk.second);
            if (dist_sq > max_distance_sq)
            {
                dist_sq = max_distance_sq;
            }
            dist_matrix[i][j] = dist_sq;
            j++;
        }
    }

    // Find association
    HungarianMatching(dist_matrix, detection.size(), tracks.size(), association);

    for (size_t i = 0; i < detection.size(); i++) {
        bool matched_flag = false;
        size_t j = 0;
        for (const auto& trk : tracks) {
            if (0 == association[i][j]) {
                // Filter out matched with high distance
                if (dist_matrix[i][j] < dist_threshold_sq) {
                    matched[trk.first] = detection[i];
                    matched_flag = true;
                }
                // It builds 1 to 1 association, so we can break from here
                break;
            }
            j++;
        }
        // if detection cannot match with any tracks
        if (!matched_flag) {
            unmatched_det.push_back(detection[i]);
        }
    }
}


// void Tracker::Run(const std::vector<cv::Rect>& detections) {
std::map<int, Tracker::Detection> Tracker::Run(const std::vector<Detection>& detections, uint64_t timestamp,
                                               const float dist_threshold_sq, const float max_distance_sq,
                                               std::map<int, Eigen::VectorXd>& predicted_states)
{
    if (prev_update_time_)
    {
        const auto microsec = timestamp - prev_update_time_.value();
        dt_ = static_cast<double>(microsec) * 1e-6;
    }

    prev_update_time_ = std::make_optional(timestamp);

    predicted_states.clear();
    /*** Predict internal tracks from previous frame ***/
    for (auto &track : tracks_) {
        track.second.setDtInModel(dt_);
        track.second.Predict();
        predicted_states.emplace(track.first, track.second.GetState()); // keep predicted states for debugging
    }

    // Hash-map between track ID and associated detection
    std::map<int, Detection> matched;
    // vector of unassociated detections
    std::vector<Detection> unmatched_det;

    // return values - matched, unmatched_det
    if (!detections.empty()) {
        AssociateDetectionsToTrackers(detections, tracks_, matched, unmatched_det, dist_threshold_sq, max_distance_sq);
    }

    std::map<int, Detection> track_to_detection_associations;

    /*** Update tracks with associated detection ***/
    for (const auto &match : matched) {
        const auto &ID = match.first;
        tracks_[ID].Update(match.second.centroid);

        track_to_detection_associations[ID] = match.second;
    }

    /*** Create new tracks for unmatched detections ***/
    for (const auto &det : unmatched_det) {
        Track tracker;
        tracker.Init(det.centroid);
        // Create new track and generate new ID
        tracks_[id_] = tracker;

        track_to_detection_associations[id_] = det;
        id_++;
    }

    /*** Delete lose tracked tracks ***/
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (it->second.coast_cycles_ > kMaxCoastCycles) {
            it = tracks_.erase(it);
        } else {
            it++;
        }
    }

    return track_to_detection_associations;
}


std::map<int, Track> Tracker::GetTracks() {
    return tracks_;
}

double Tracker::GetDT()
{
    return dt_;
}
