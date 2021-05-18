#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include "kalman_filter.h"

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void setDtInModel(double dt);

    // void Init(const cv::Rect& bbox);
    void Predict();
    // void Update(const cv::Rect& bbox);
    // cv::Rect GetStateAsBbox() const;

    void Init(const Eigen::VectorXd observation);
    void Update(const Eigen::VectorXd observation);

    Eigen::VectorXd GetState() const;
    Eigen::MatrixXd GetCovariance() const;

    std::tuple<float, int, float> GetNIS() const;
    Eigen::MatrixXd GetS() const;
    Eigen::VectorXd GetY() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect& bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;

    KalmanFilter kf_;

    Eigen::MatrixXd F_;
};
