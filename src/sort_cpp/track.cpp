#include "sort_cpp/track.h"

// constexpr unsigned int num_states = 8; // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
constexpr unsigned int num_states = 6;    // state - center_x, center_y, center_z, v_cx, v_cy, v_cz
constexpr unsigned int num_obs = 3;       // observation - center_x, center_y, center_z

Track::Track() : kf_(num_states, num_obs) {

    /*** Define constant velocity model ***/
    // x_k+1 = x_k + v_k
    // v_k+1 = v_k
    // no input (drop BkUk from prediction state estimate)

    // // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    // kf_.F_ <<
    //         1, 0, 0, 0, 1, 0, 0, 0,
    //         0, 1, 0, 0, 0, 1, 0, 0,
    //         0, 0, 1, 0, 0, 0, 1, 0,
    //         0, 0, 0, 1, 0, 0, 0, 1,
    //         0, 0, 0, 0, 1, 0, 0, 0,
    //         0, 0, 0, 0, 0, 1, 0, 0,
    //         0, 0, 0, 0, 0, 0, 1, 0,
    //         0, 0, 0, 0, 0, 0, 0, 1;

    // state - state - center_x, center_y, center_z, v_cx, v_cy, v_cz
    kf_.F_ <<
            1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    // Error covariance matrix P
    // Give high uncertainty to the unobservable initial velocities

    // kf_.P_ <<
    //        10, 0, 0, 0, 0, 0, 0, 0,
    //         0, 10, 0, 0, 0, 0, 0, 0,
    //         0, 0, 10, 0, 0, 0, 0, 0,
    //         0, 0, 0, 10, 0, 0, 0, 0,
    //         0, 0, 0, 0, 10000, 0, 0, 0,
    //         0, 0, 0, 0, 0, 10000, 0, 0,
    //         0, 0, 0, 0, 0, 0, 10000, 0,
    //         0, 0, 0, 0, 0, 0, 0, 10000;
    kf_.P_ <<
           10, 0,  0,  0,     0,     0,
            0, 10, 0,  0,     0,     0,
            0, 0,  10, 0,     0,     0,
            0, 0,  0,  10000, 0,     0,
            0, 0,  0,  0,     10000, 0,
            0, 0,  0,  0,     0,     10000;

    // Observation matrix   num_obs, num_states
    kf_.H_ <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0;

    // Covariance matrix of process noise   num_states, num_states
    kf_.Q_ <<
            1, 0, 0, 0,    0,    0,
            0, 1, 0, 0,    0,    0,
            0, 0, 1, 0,    0,    0,
            0, 0, 0, 0.01, 0,    0,
            0, 0, 0, 0,    0.01, 0,
            0, 0, 0, 0,    0,    0.01;

    // Covariance matrix of observation noise   num_obs, num_obs
    kf_.R_ <<
            1, 0, 0,
            0, 1, 0,
            0, 0, 1;
}


// Get predicted locations from existing trackers
// dt is time elapsed between the current and previous measurements
void Track::Predict() {
    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0) {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}


// // Update matched trackers with assigned detections
// void Track::Update(const cv::Rect& bbox) {

//     // get measurement update, reset coast cycle count
//     coast_cycles_ = 0;
//     // accumulate hit streak count
//     hit_streak_++;

//     // observation - center_x, center_y, area, ratio
//     Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
//     kf_.Update(observation);
// }

// Update matched trackers with assigned detections
void Track::Update(const Eigen::VectorXd observation)
{
    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, center_z
    kf_.Update(observation);
}

// // Create and initialize new trackers for unmatched detections, with initial bounding box
// void Track::Init(const cv::Rect &bbox) {
//     kf_.x_.head(4) << ConvertBboxToObservation(bbox);
//     hit_streak_++;
// }

// Create and initialize new trackers for unmatched detections, with initial observation
void Track::Init(const Eigen::VectorXd observation)
{
    kf_.x_.head(3) << observation;
    hit_streak_++;
}

// /**
//  * Returns the current bounding box estimate
//  * @return
//  */
// cv::Rect Track::GetStateAsBbox() const {
//     return ConvertStateToBbox(kf_.x_);
// }


Eigen::VectorXd Track::GetState() const
{
    return kf_.x_;
}

Eigen::MatrixXd Track::GetCovariance() const
{
    return kf_.P_;
}

float Track::GetNIS() const {
    return kf_.NIS_;
}


/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
Eigen::VectorXd Track::ConvertBboxToObservation(const cv::Rect& bbox) const{
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
    auto width = static_cast<float>(bbox.width);
    auto height = static_cast<float>(bbox.height);
    float center_x = bbox.x + width / 2;
    float center_y = bbox.y + height / 2;
    observation << center_x, center_y, width, height;
    return observation;
}


/**
 * Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
 * [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
 *
 * @param state
 * @return
 */
cv::Rect Track::ConvertStateToBbox(const Eigen::VectorXd &state) const {
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    auto width = static_cast<int>(state[2]);
    auto height = static_cast<int>(state[3]);
    auto tl_x = static_cast<int>(state[0] - width / 2.0);
    auto tl_y = static_cast<int>(state[1] - height / 2.0);
    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
    return rect;
}
