#include "sort_cpp/kalman_filter.h"


KalmanFilter::KalmanFilter(unsigned int num_states, unsigned int num_obs) :
        num_states_(num_states), num_obs_(num_obs) {
    /*** Predict ***/
    // State vector
    x_ = Eigen::VectorXd::Zero(num_states);
    // Predicted(a prior) state vector
    x_predict_ = Eigen::VectorXd::Zero(num_states);

    // State transition matrix F_
    F_ = Eigen::MatrixXd::Zero(num_states, num_states);

    // Error covariance matrix P
    P_ = Eigen::MatrixXd::Zero(num_states, num_states);

    // Predicted(a prior) error covariance matrix
    P_predict_ = Eigen::MatrixXd::Zero(num_states, num_states);

    // Covariance matrix of process noise
    q_std_ = 0.0;
    Q_ = Eigen::MatrixXd::Zero(num_states, num_states);

    /*** Update ***/
    // Observation matrix
    H_ = Eigen::MatrixXd::Zero(num_obs, num_states);

    // Covariance matrix of observation noise
    R_ = Eigen::MatrixXd::Zero(num_obs, num_obs);

    metrics_.log_likelihood_delta_ = 0.0;
    metrics_.NIS_average_ = 0.0;
    metrics_.NIS_vector_.clear();
}


void KalmanFilter::Coast() {
    x_predict_ = F_ * x_;
    P_predict_ = F_ * P_ * F_.transpose() + Q_;
}


void KalmanFilter::Predict() {
    Coast();
    x_ = x_predict_;
    P_ = P_predict_;
}


Eigen::VectorXd KalmanFilter::PredictionToObservation(const Eigen::VectorXd &state) {
    return (H_*state);
}


void KalmanFilter::Update(const Eigen::VectorXd& z) {
    Eigen::VectorXd z_predict = PredictionToObservation(x_predict_);

    // y - innovation, z - real observation, z_predict - predicted observation
    y_ = z - z_predict;

    Eigen::MatrixXd Ht = H_.transpose();

    // S - innovation covariance matrix
    S_ = H_ * P_predict_ * Ht + R_;

//    std::cout << "P_predict = " << P_predict_.diagonal().transpose() << std::endl;
//
//    std::cout << "Z = " << std::endl;
//    std::cout << z << std::endl;
//
//    std::cout << "Z_pred = " << std::endl;
//    std::cout << z_predict << std::endl;
//
//    std::cout << "y = " << std::endl;
//    std::cout << y << std::endl;
//
//    std::cout << "S = " << std::endl;
//    std::cout << S_ << std::endl;

    // K - Kalman gain
    Eigen::MatrixXd K = P_predict_ * Ht * S_.inverse();

    // Updated state estimation
    x_ = x_predict_ + K * y_;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_states_, num_states_);
    // Joseph form
    //P_ = (I - K * H_) * P_predict_ * (I - K * H_).transpose() + K * R_ * K.transpose();
    // Optimal gain
    P_ = (I - K * H_) * P_predict_;

    // evaluation metrics
    float NIS = y_.transpose() * S_.inverse() * y_;     // normalized innovation error squared

    // metrics_.N_ += 1;
    metrics_.NIS_vector_.push_back(NIS);

    if (metrics_.NIS_vector_.size() <= max_size_metrics)
    {
        metrics_.NIS_average_ = 0;
        for (const auto& q : metrics_.NIS_vector_)
        {
            metrics_.NIS_average_ += q;
        }
        metrics_.NIS_average_ = metrics_.NIS_average_ / metrics_.NIS_vector_.size();
    }
    else
    {
        float first = metrics_.NIS_vector_.front();
        metrics_.NIS_vector_.pop_front();
        metrics_.NIS_average_ = metrics_.NIS_average_ + (NIS - first) / max_size_metrics;
    }

    // std::cout << "NIS = " << metrics_.NIS_average_ << std::endl;
}


float KalmanFilter::CalculateLogLikelihood(const Eigen::VectorXd& y, const Eigen::MatrixXd& S) {
    float log_likelihood;

    // Note: Computing log(M.determinant()) in Eigen C++ is risky for large matrices since it may overflow or underflow.
    // compute the Cholesky decomposition of the innovation covariance matrix, because it is symmetric
    // S = L * L^T = U^T * U
    // then retrieve factor L in the decomposition
    auto& L = S.llt().matrixL();

    // find log determinant of innovation covariance matrix
    float log_determinant = 0;
    for (unsigned int i = 0; i < S.rows(); i++)
        log_determinant += log(L(i, i));
    log_determinant *= 2;

    // log-likelihood expression for current iteration
    log_likelihood = -0.5 * (y.transpose() * S.inverse() * y + num_obs_ * log(2 * M_PI) + log_determinant);

    if (std::isnan(log_likelihood)) {
    	log_likelihood = -1e50;
	}

	return log_likelihood;
}
